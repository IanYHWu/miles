"""VllmPusher's side of vLLM's NCCL weight-transfer protocol (one communicator per server), with fake
NCCL / HTTP / CUDA."""

import threading
import time

import pytest
import torch

from miles.backends.training_utils.weight_update.vllm_nccl.pusher import VllmPusher, chunk_metadata


class FakeNccl:
    def __init__(self):
        self.calls, self.n = [], 0

    def unique_id(self):
        self.n += 1
        n = self.n

        class Uid:
            internal = bytes([n] * 128)

        return Uid()

    def init_rank(self, world, uid, rank):
        comm = f"comm{bytes(uid.internal)[0]}"
        self.calls.append(("init", comm, world, rank))
        return comm

    def all_reduce_sum(self, comm, tensor, stream):
        self.calls.append(("all_reduce", comm, tensor.numel()))

    def abort(self, comm):
        self.calls.append(("abort", comm))

    def broadcast(self, comm, tensor, root, stream):
        self.calls.append(("broadcast", comm, tuple(tensor.shape)))

    def version(self):
        return 23007


class FakeCuda:
    def __init__(self):
        self.events = []

    def new_stream(self):
        return object()

    def zeros_fp32(self, n):
        return torch.zeros(n, dtype=torch.float32)

    def to_device(self, t):
        return t

    def wait_current(self, stream):
        self.events.append("wait")

    def synchronize(self, stream):
        self.events.append("sync")

    def wait_until_done(self, stream, check):
        check()
        self.events.append("sync")


class FakeHttp:
    def __init__(self, fail=None, hang=None):
        self.log, self.version = [], {}
        self.fail = fail or {}  # (host, path prefix) -> fail
        self.hang = hang or set()

    def __call__(self, method, url, body, timeout):
        host, path = url.split("/", 3)[2], "/" + url.split("/", 3)[3]
        self.log.append((host, method, path, body))
        for h, prefix in self.fail:
            if h == host and path.startswith(prefix):
                raise RuntimeError(f"{host} {prefix} failed")
        for h, prefix in self.hang:
            if h == host and path.startswith(prefix):
                time.sleep(5)
        if path == "/get_world_size":
            return {"world_size": 8}
        if path == "/finish_weight_update":
            self.version[host] = body["weight_version"]
        if path == "/update_weight_version":
            self.version[host] = body["new_version"]
        if path == "/weight_info":
            return {"weight_version": self.version.get(host)}
        if path.startswith("/reset_prefix_cache"):
            return {"success": True}
        return {}


ENGINES = [{"url": "http://a:1", "world_size": 8}, {"url": "http://b:1", "world_size": 8}]


def make(http, engines=ENGINES):
    nccl, cuda = FakeNccl(), FakeCuda()
    return VllmPusher(engines, "unused.so", lib=nccl, http=http, cuda=cuda), nccl, cuda


def batches():
    return iter(
        [
            [("model.a", torch.ones(2, 3, dtype=torch.bfloat16))],
            [("model.b", torch.ones(4, dtype=torch.bfloat16)), ("model.c", torch.ones(1, dtype=torch.bfloat16))],
        ]
    )


def test_connect_one_communicator_per_server():
    http = FakeHttp()
    pusher, nccl, _ = make(http)
    info = pusher.connect()
    inits = {h: b["init_info"] for h, m, p, b in http.log if p == "/init_weight_transfer_engine"}
    assert {h: (i["rank_offset"], i["world_size"]) for h, i in inits.items()} == {"a:1": (1, 9), "b:1": (1, 9)}
    assert inits["a:1"]["nccl_unique_id_b64"] != inits["b:1"]["nccl_unique_id_b64"]
    assert sorted(c[1] for c in nccl.calls if c[0] == "init") == ["comm1", "comm2"]
    assert len([c for c in nccl.calls if c[0] == "all_reduce"]) == 2  # a warm-up per communicator
    assert info["servers"] == 2 and info["dropped"] == []


def test_push_protocol_order_and_metadata():
    http = FakeHttp()
    pusher, nccl, cuda = make(http)
    pusher.connect()
    http.log.clear()
    metrics = pusher.push(batches(), "7")
    paths = [p for h, m, p, b in http.log if h == "a:1"]
    assert paths == [
        "/pause?mode=keep&clear_cache=true",
        "/start_weight_update",
        "/update_weights",
        "/update_weights",
        "/finish_weight_update",
        "/reset_prefix_cache?reset_running_requests=true",
        "/resume",
        "/weight_info",
    ]
    updates = [b["update_info"] for h, m, p, b in http.log if p == "/update_weights" and h == "a:1"]
    assert updates[1] == {
        "names": ["model.b", "model.c"],
        "dtype_names": ["bfloat16", "bfloat16"],
        "shapes": [[4], [1]],
    }
    per_comm = {}
    for c in nccl.calls:
        if c[0] == "broadcast":
            per_comm.setdefault(c[1], []).append(c[2])
    assert per_comm == {"comm1": [(2, 3), (4,), (1,)], "comm2": [(2, 3), (4,), (1,)]}
    assert metrics["version"] == "7" and metrics["tensors"] == 3 and metrics["servers"] == 2


def test_a_server_dying_mid_push_is_dropped_and_the_push_completes_on_the_others():
    http = FakeHttp()
    pusher, nccl, _ = make(http)
    pusher.connect()
    http.fail[("b:1", "/update_weights")] = True  # b died after the pause
    metrics = pusher.push(batches(), "3")
    assert metrics["dropped"] == ["http://b:1"] and metrics["servers"] == 1
    assert ("abort", "comm2") in nccl.calls and ("abort", "comm1") not in nccl.calls
    assert http.version == {"a:1": "3", "b:1": "poisoned"}  # b never serves partial weights as a version
    assert "/resume" in [p for h, m, p, b in http.log if h == "a:1"]
    assert "/resume" not in [p for h, m, p, b in http.log if h == "b:1"]
    pusher.close()
    assert nccl.calls.count(("abort", "comm2")) == 1  # never aborted twice


def test_a_server_failing_mid_batch_is_noticed_while_its_broadcast_drains():
    gate = threading.Event()

    class SlowFail(FakeHttp):
        def __call__(self, method, url, body, timeout):
            if url.endswith("/update_weights") and "b:1" in url:
                gate.wait(5)
                raise RuntimeError("server died")
            return super().__call__(method, url, body, timeout)

    http = SlowFail()
    pusher, nccl, cuda = make(http)
    pusher.connect()
    waits = []

    def waiting(stream, check):  # b's NCCL work never completes (peer gone): only check() can end its wait
        waits.append(stream)
        gate.set()
        t = time.time()
        while time.time() - t < 10:
            check()
            if len(waits) % 2 == 1 and waits.count(stream) == 1 and stream is pusher.links[0].stream:
                return  # a's work completes
            time.sleep(0.01)
        raise AssertionError("not aborted")

    cuda.wait_until_done = waiting
    t = time.time()
    metrics = pusher.push(iter([[("model.a", torch.ones(1, dtype=torch.bfloat16))]]), "1")
    assert time.time() - t < 5 and metrics["dropped"] == ["http://b:1"]
    assert nccl.calls.count(("abort", "comm2")) == 1


def test_every_server_failing_is_an_error_and_nothing_resumes():
    http = FakeHttp(fail={("a:1", "/update_weights"): 1, ("b:1", "/update_weights"): 1})
    pusher, nccl, _ = make(http)
    pusher.connect()
    with pytest.raises(RuntimeError, match="every server failed"):
        pusher.push(batches(), "1")
    assert "/resume" not in [p for h, m, p, b in http.log]
    assert http.version == {"a:1": "poisoned", "b:1": "poisoned"}


def test_connect_drops_a_rejecting_server_and_reports_a_timed_out_one_as_wedged():
    http = FakeHttp(fail={("b:1", "/init_weight_transfer_engine"): 1})
    pusher, nccl, _ = make(http)
    info = pusher.connect()
    assert info["servers"] == 1 and info["dropped"] == ["http://b:1"] and info["wedged"] == []
    # a rendezvous that never completes on one server: only that one is wedged
    http2 = FakeHttp(hang={("b:1", "/init_weight_transfer_engine")})
    pusher2, nccl2, _ = make(http2)
    pusher2.connect_timeout = 0.5
    never = threading.Event()
    real = nccl2.init_rank
    nccl2.init_rank = lambda world, uid, rank: (
        never.wait(10) if bytes(uid.internal)[0] == 2 else real(world, uid, rank)
    )
    info = pusher2.connect()
    assert info["servers"] == 1 and info["wedged"] == ["http://b:1"]


def test_connect_with_every_server_timing_out_raises_with_all_wedged():
    http = FakeHttp(hang={("a:1", "/init_weight_transfer_engine"), ("b:1", "/init_weight_transfer_engine")})
    pusher, nccl, _ = make(http)
    pusher.connect_timeout = 0.5
    never = threading.Event()
    nccl.init_rank = lambda world, uid, rank: never.wait(10)
    with pytest.raises(TimeoutError) as info:
        pusher.connect()
    assert sorted(info.value.wedged) == ["http://a:1", "http://b:1"]


def test_watchdog_aborts_a_hung_push_and_never_after_success():
    http = FakeHttp()
    pusher, nccl, cuda = make(http)
    pusher.connect()
    pusher.push(batches(), "1")
    pusher._watchdog()  # fired just after a successful push: nothing happens
    assert not [c for c in nccl.calls if c[0] == "abort"] and not pusher.aborted
    pusher.push_timeout = 0.2

    def hang(stream, check):
        t = time.time()
        while time.time() - t < 5:
            check()
            time.sleep(0.01)

    cuda.wait_until_done = hang
    with pytest.raises(RuntimeError, match="aborted"):
        pusher.push(batches(), "2")
    assert sorted(c[1] for c in nccl.calls if c[0] == "abort") == ["comm1", "comm2"]
    assert "/resume" not in [p for h, m, p, b in http.log[-6:]]


def test_no_servers_is_not_ok():
    pusher = VllmPusher([], "unused.so", lib=FakeNccl(), http=FakeHttp(), cuda=FakeCuda())
    assert pusher.connect()["servers"] == 0
    consumed = []
    out = pusher.push((consumed.append(i) or [("x", torch.ones(1))] for i in range(3)), "2")
    assert out["skipped"] == "no servers" and out["ok"] is False and consumed == [0, 1, 2]


def test_chunk_metadata():
    assert chunk_metadata([("x", torch.zeros(2, 2, dtype=torch.float32))]) == {
        "names": ["x"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
    }
