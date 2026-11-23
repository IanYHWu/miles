"""Global trainer rank 0's side of vLLM's NCCL weight-transfer protocol.

Routes (vllm/entrypoints/serve/dev/rlhf): ``/pause``, ``/resume``, ``/init_weight_transfer_engine``,
``/start_weight_update``, ``/update_weights``, ``/finish_weight_update``, ``/weight_info``.

* Rendezvous by NCCL unique id, vLLM's path for trainers that do not speak its TCPStore protocol:
  the trainer mints one id per server and is rank 0 of that server's communicator, the server's
  GPUs join with ``rank_offset`` 1. After ``ncclCommInitRank`` every vLLM worker issues a
  one-element fp32 sum all-reduce (PyNcclCommunicator's warm-up), which the trainer matches.
* One update: pause every server with ``mode=keep`` (in-flight requests are frozen and continue
  under the new weights), ``start_weight_update``, then for every batch of HF-named tensors
  ``update_weights`` with the batch's names/dtypes/shapes on every server while rank 0 broadcasts
  the tensors one by one (vLLM's non-packed receive), ``finish_weight_update`` with the version,
  reset the prefix cache, ``resume``.

Weights the trainer does not hold (e.g. a frozen MTP draft or vision tower) are not sent; vLLM's
layerwise reload keeps the old tensors of modules that received nothing.
"""

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from miles.backends.training_utils.weight_update.vllm_nccl.nccl import NcclLib

logger = logging.getLogger(__name__)

NamedTensors = list[tuple[str, torch.Tensor]]


@dataclass
class VllmEngine:
    url: str  # http://host:port of one vLLM server
    world_size: int  # its GPUs (TP x PP x DP), as vLLM's /get_world_size reports


def _http(method: str, url: str, body=None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode() or "{}"
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {url}: HTTP {error.code} {error.read()[:500]!r}") from error
    return json.loads(text)


def chunk_metadata(named_tensors: NamedTensors) -> dict:
    """vLLM's ``NCCLWeightTransferUpdateInfo`` for one batch: names, dtype names, shapes."""
    return {
        "names": [name for name, _ in named_tensors],
        "dtype_names": [str(tensor.dtype).split(".")[-1] for _, tensor in named_tensors],
        "shapes": [list(tensor.shape) for _, tensor in named_tensors],
    }


class _Link:
    """One rollout server: its own NCCL communicator (trainer rank 0 + the server's GPUs) and stream."""

    def __init__(self, engine: VllmEngine):
        self.engine = engine
        self.comm = None
        self.stream = None
        self.error: str | None = None

    @property
    def url(self) -> str:
        return self.engine.url


class _LinkDropped(Exception):
    pass


class _Cuda:
    """The CUDA calls of the sender (replaced by a stub in tests)."""

    def new_stream(self):
        return torch.cuda.Stream()

    def zeros_fp32(self, n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=torch.float32, device="cuda")

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.is_cuda else tensor.cuda()

    def wait_current(self, stream) -> None:
        stream.wait_stream(torch.cuda.current_stream())

    def synchronize(self, stream) -> None:
        stream.synchronize()

    def wait_until_done(self, stream, check: Callable[[], None], poll: float = 0.005) -> None:
        """Wait for the stream's queued work while calling ``check`` (which raises to give up)."""
        done = torch.cuda.Event()
        done.record(stream)
        while not done.query():
            check()
            time.sleep(poll)


class VllmPusher:
    """Global trainer rank 0's side of the NCCL weight transfer, one communicator PER SERVER.

    With one group spanning every server, a server dying mid-broadcast leaves all the others blocked
    inside their own NCCL receive (vLLM never aborts it). With one group per server a failure only
    breaks that server's link: it is aborted and dropped, and the push completes on the others.

    Timeouts must nest below the trainer's distributed timeout: while rank 0 broadcasts, the other
    trainer ranks wait in the HF iterator's gathers, and a push that outlives the NCCL watchdog tears
    the trainer down. Every HTTP call of a push takes at most ``http_timeout`` < ``push_timeout``.

    ``extra_ranks`` and ``test_batch_delay`` are fault-injection knobs for validating recovery:
    ranks that never join (a connect that must time out), and a sleep per batch (a push slow enough
    to kill a server during it).
    """

    def __init__(
        self,
        engines: Sequence[VllmEngine | dict],
        nccl_so: str,
        http_timeout: float = 300.0,
        lib=None,
        http=_http,
        cuda=None,
        push_timeout: float = 900.0,
        clear_cache: bool = True,
        connect_timeout: float = 600.0,
        extra_ranks: int = 0,
        test_batch_delay: float = 0.0,
    ):
        self.engines = [e if isinstance(e, VllmEngine) else VllmEngine(**e) for e in engines]
        self.links = [_Link(e) for e in self.engines]
        self.http_timeout = http_timeout
        self.http = http
        self.nccl = lib if lib is not None else NcclLib(nccl_so)
        self.cuda = cuda if cuda is not None else _Cuda()
        self.pool = ThreadPoolExecutor(max_workers=max(8, 2 * len(self.engines)))
        self.metrics: dict = {}
        self.push_timeout = push_timeout
        # vLLM ignores /pause's clear_cache in mode=keep; the push resets the prefix cache itself
        self.clear_cache = clear_cache
        self.connect_timeout = connect_timeout
        self.extra_ranks = extra_ranks
        self.test_batch_delay = test_batch_delay
        self._guard = threading.Lock()
        self._pushing = False
        self.aborted = False
        self.dropped: list[str] = []  # urls dropped since connect (failed, dead or wedged)

    def live(self) -> list[_Link]:
        return [link for link in self.links if link.comm is not None]

    def _drop(self, link: _Link, reason: str) -> None:
        with self._guard:
            comm, link.comm = link.comm, None
        if comm is not None:
            logger.error("VLLM_PUSH aborting the link to %s: %s", link.url, reason)
            self.nccl.abort(comm)
        if link.error is None:
            link.error = reason
            self.dropped.append(link.url)

    def connect(self) -> dict:
        """One communicator per server, all rendezvous concurrently.

        A server that rejects the init, or whose rendezvous does not finish within
        ``connect_timeout``, is dropped; the timed-out ones are reported in ``wedged`` (stuck in their
        own ncclCommInitRank, never to be used again). Raises if no server could be connected, with
        the ``wedged`` urls on the exception."""
        if not self.links:
            logger.info("VLLM_PUSH no servers: nothing to connect")
            return {"world_size": 1, "servers": 0, "seconds": 0.0, "dropped": [], "wedged": []}
        start = time.time()
        pending = []
        for link in self.links:
            engine = link.engine
            try:
                reported = self.http("GET", f"{engine.url}/get_world_size", None, 60)["world_size"]
            except Exception as error:  # noqa: BLE001
                link.error = f"unreachable: {error!r}"
                self.dropped.append(engine.url)
                continue
            if int(reported) != int(engine.world_size):
                link.error = f"reports world size {reported}, expected {engine.world_size}"
                self.dropped.append(engine.url)
                continue
            world = 1 + int(engine.world_size) + self.extra_ranks
            uid = self.nccl.unique_id()
            init_info = {
                "nccl_unique_id_b64": base64.b64encode(bytes(uid.internal)).decode(),
                "rank_offset": 1,
                "world_size": world,
                "packed": False,
            }
            server_init = self.pool.submit(
                self.http,
                "POST",
                f"{engine.url}/init_weight_transfer_engine",
                {"init_info": init_info},
                self.http_timeout,
            )
            result = {}

            def init(result=result, world=world, uid=uid):
                try:
                    result["comm"] = self.nccl.init_rank(world, uid, 0)
                except BaseException as error:  # noqa: BLE001
                    result["error"] = error

            # a thread, not the pool: an init that never completes must not occupy a pool worker
            helper = threading.Thread(target=init, daemon=True)
            helper.start()
            pending.append((link, server_init, helper, result))
        wedged = []
        while pending:
            still = []
            for link, server_init, helper, result in pending:
                if server_init.done() and server_init.exception() is not None:
                    link.error = f"rejected the weight-transfer init: {server_init.exception()!r}"
                    self.dropped.append(link.url)
                    continue
                if not helper.is_alive():
                    if "error" in result:
                        link.error = repr(result["error"])
                        self.dropped.append(link.url)
                    else:
                        link.comm = result["comm"]
                    continue
                if time.time() - start > self.connect_timeout:
                    link.error = "NCCL rendezvous timed out"
                    self.dropped.append(link.url)
                    wedged.append(link.url)
                    continue
                still.append((link, server_init, helper, result))
            pending = still
            if pending:
                time.sleep(0.2)
        for link in self.live():
            link.stream = self.cuda.new_stream()
            warm = self.cuda.zeros_fp32(1)  # matches PyNcclCommunicator's warm-up all-reduce
            self.cuda.wait_current(link.stream)
            self.nccl.all_reduce_sum(link.comm, warm, link.stream)
            self.cuda.synchronize(link.stream)
        if not self.live():
            errors = [link.error for link in self.links]
            error_cls = TimeoutError if wedged else RuntimeError
            error = error_cls(f"no server connected ({errors})")
            error.wedged = wedged
            raise error
        info = {
            "servers": len(self.live()),
            "dropped": list(self.dropped),
            "wedged": wedged,
            "nccl_version": self.nccl.version(),
            "seconds": round(time.time() - start, 1),
        }
        logger.info("VLLM_PUSH connected %s", json.dumps(info))
        return info

    def close(self) -> None:
        """Abort every communicator (in daemon threads: an abort can block on a dead peer) and the pool."""
        for link in self.links:
            with self._guard:
                comm, link.comm = link.comm, None
            if comm is not None:
                threading.Thread(target=self.nccl.abort, args=(comm,), daemon=True).start()
        self.pool.shutdown(wait=False)

    def _watchdog(self) -> None:
        with self._guard:
            if not self._pushing:
                return
            self.aborted = True
        for link in self.live():
            self._drop(link, f"push exceeded {self.push_timeout:.0f}s")

    def _each(self, method: str, path: str, body=None, timeout: float | None = None) -> dict:
        """The same request to every live server concurrently; a server that fails is dropped."""
        futures = [
            (link, self.pool.submit(self.http, method, f"{link.url}{path}", body, timeout or self.http_timeout))
            for link in self.live()
        ]
        out = {}
        for link, future in futures:
            try:
                out[link.url] = future.result()
            except Exception as error:  # noqa: BLE001
                self._drop(link, f"{path}: {error!r}")
        return out

    def push(self, batches: Iterable[NamedTensors], version: str) -> dict:
        """Send every (name, tensor) batch to every live server, then publish ``version`` on them.

        ``batches`` is always consumed fully: the other trainer ranks iterate the same collective
        stream. A server that fails during the push is aborted and dropped, and its version is set
        to "poisoned" so it never serves partial weights under a version; the push succeeds on the
        others. No live server left raises."""
        if not self.links:
            for _ in batches:
                pass
            return {"version": str(version), "tensors": 0, "skipped": "no servers", "seconds": 0.0, "ok": False}
        if not self.live():
            raise RuntimeError("no live server link: reconnect first")
        watchdog = threading.Timer(self.push_timeout, self._watchdog)
        watchdog.daemon = True
        with self._guard:
            self._pushing = True
        watchdog.start()
        dropped_before = len(self.dropped)
        try:
            return self._push(batches, version, dropped_before)
        finally:
            with self._guard:
                self._pushing = False
            watchdog.cancel()
            for url in self.dropped[dropped_before:]:
                try:
                    self.http("POST", f"{url}/update_weight_version", {"new_version": "poisoned"}, 30)
                except Exception:  # noqa: BLE001 - a dead server
                    pass

    def _push(self, batches: Iterable[NamedTensors], version: str, dropped_before: int) -> dict:
        start = time.time()
        clear_cache = "true" if self.clear_cache else "false"
        self._each("POST", f"/pause?mode=keep&clear_cache={clear_cache}")
        paused = time.time()
        sent_bytes = tensors = 0
        self._each("POST", "/start_weight_update")
        for named in batches:
            named = [(name, tensor if tensor.is_contiguous() else tensor.contiguous()) for name, tensor in named]
            links = self.live()
            if not named or not links:
                continue  # keep consuming the stream (the other ranks are in its collectives)
            metadata = chunk_metadata(named)
            updates = {
                link.url: self.pool.submit(
                    self.http, "POST", f"{link.url}/update_weights", {"update_info": metadata}, self.http_timeout
                )
                for link in links
            }
            named = [(name, self.cuda.to_device(tensor)) for name, tensor in named]
            for link in links:
                # the iterator's gathers ran on the current stream: broadcast only after they finished
                self.cuda.wait_current(link.stream)
                for _, tensor in named:
                    if link.comm is None:
                        break
                    self.nccl.broadcast(link.comm, tensor, 0, link.stream)
            for _, tensor in named:
                sent_bytes += tensor.numel() * tensor.element_size()
                tensors += 1
            for link in links:
                update = updates[link.url]

                def check(link=link, update=update):
                    if self.aborted:
                        raise RuntimeError("push aborted")
                    if update.done() and update.exception() is not None:
                        self._drop(link, f"update_weights: {update.exception()!r}")
                        raise _LinkDropped()

                try:
                    self.cuda.wait_until_done(link.stream, check)
                    update.result()
                except _LinkDropped:
                    continue
                except RuntimeError:
                    if self.aborted:
                        raise
                    reason = f"update_weights: {update.exception()!r}" if update.done() else "update_weights failed"
                    self._drop(link, reason)
            if self.test_batch_delay:
                time.sleep(self.test_batch_delay)
        if self.aborted:
            raise RuntimeError("push aborted by the watchdog")
        self._each("POST", "/finish_weight_update", {"weight_version": str(version)})
        cache_reset = None
        if self.clear_cache:
            # Still paused: drop every cached KV block (pre-empting the frozen requests, which
            # re-prefill on resume) so nothing computed with the old weights is reused.
            reset = self._each("POST", "/reset_prefix_cache?reset_running_requests=true")
            cache_reset = all(r.get("success") for r in reset.values())
        if not self.live():
            raise RuntimeError(f"every server failed during the push: {self.dropped[dropped_before:]}")
        self._each("POST", "/resume")
        served = self._each("GET", "/weight_info", None, 60)
        wrong = {
            url: r.get("weight_version") for url, r in served.items() if str(r.get("weight_version")) != str(version)
        }
        if wrong or not served:
            raise RuntimeError(f"servers report weight versions {wrong}, expected {version}")
        self.metrics = {
            "version": str(version),
            "tensors": tensors,
            "gib": round(sent_bytes / 2**30, 2),
            "servers": len(self.live()),
            "dropped": self.dropped[dropped_before:],
            "cache_reset": cache_reset,
            "pause_seconds": round(paused - start, 1),
            "seconds": round(time.time() - start, 1),
        }
        logger.info("VLLM_PUSH %s", json.dumps(self.metrics))
        return self.metrics
