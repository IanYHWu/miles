"""ctypes bindings for the NCCL calls of the vLLM weight sender.

The trainer's torch links its own NCCL, usually not the version the vLLM servers run, and NCCL
refuses communicators whose ranks run different versions. The sender therefore loads a copy of
the serving image's ``libnccl.so`` explicitly and drives it directly.
"""

import ctypes
import os

import torch

NCCL_UNIQUE_ID_BYTES = 128
NCCL_DTYPES = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}
NCCL_SUM = 0


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * NCCL_UNIQUE_ID_BYTES)]


class NcclLib:
    """The NCCL entry points the sender needs, from one explicit libnccl.so."""

    def __init__(self, so_path: str):
        # RTLD_DEEPBIND: this copy's internal calls must bind to its own symbols. Otherwise they
        # resolve to torch's libnccl, already in the global scope, and that version then reads a
        # communicator of this one ("NCCL WARN Error: corrupted comm object detected").
        self.lib = ctypes.CDLL(so_path, mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p
        self.lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
        self.lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            NcclUniqueId,
            ctypes.c_int,
        ]
        collective_argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.ncclBroadcast.argtypes = collective_argtypes
        self.lib.ncclAllReduce.argtypes = collective_argtypes
        self.lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self.lib.ncclCommAbort.argtypes = [ctypes.c_void_p]

    def _check(self, result: int, what: str) -> None:
        if result != 0:
            raise RuntimeError(f"NCCL {what} failed: {self.lib.ncclGetErrorString(result).decode()}")

    def version(self) -> int:
        version = ctypes.c_int()
        self._check(self.lib.ncclGetVersion(ctypes.byref(version)), "ncclGetVersion")
        return version.value

    def unique_id(self) -> NcclUniqueId:
        uid = NcclUniqueId()
        self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        return uid

    def init_rank(self, world_size: int, uid: NcclUniqueId, rank: int) -> ctypes.c_void_p:
        comm = ctypes.c_void_p()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(comm), world_size, uid, rank), "ncclCommInitRank")
        return comm

    def broadcast(self, comm, tensor: torch.Tensor, root: int, stream: torch.cuda.Stream) -> None:
        ptr = ctypes.c_void_p(tensor.data_ptr())
        result = self.lib.ncclBroadcast(
            ptr, ptr, tensor.numel(), NCCL_DTYPES[tensor.dtype], root, comm, ctypes.c_void_p(stream.cuda_stream)
        )
        self._check(result, "ncclBroadcast")

    def all_reduce_sum(self, comm, tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
        ptr = ctypes.c_void_p(tensor.data_ptr())
        result = self.lib.ncclAllReduce(
            ptr, ptr, tensor.numel(), NCCL_DTYPES[tensor.dtype], NCCL_SUM, comm, ctypes.c_void_p(stream.cuda_stream)
        )
        self._check(result, "ncclAllReduce")

    def destroy(self, comm) -> None:
        self.lib.ncclCommDestroy(comm)

    def abort(self, comm) -> None:
        """Abort a communicator: unblocks its pending operations, also with dead peers."""
        self.lib.ncclCommAbort(comm)
