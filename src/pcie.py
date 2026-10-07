"""Cumulative PCIe byte counters of GPU 0, read through the driver's NVML (ctypes, no extra package).

NVML fields 197 / 198 (NVML_FI_DEV_PCIE_COUNT_TX_BYTES / RX_BYTES) count bytes since
driver load, from the GPU's side: TX = GPU -> host, RX = host -> GPU. Reading them at
phase boundaries, after a device sync, gives each phase's exact bus traffic.

Validated on an RTX 2000 Ada, driver 580: a 400 MB host->device copy reads ~430 MB RX,
so counts include ~8-10 % PCIe protocol overhead. The counters are *per device*, not
per process: other processes on the GPU (a display server, co-tenants) are counted too,
so the numbers only mean something on an exclusive GPU.

ponytail: this belongs in denet, which already samples NVML. See docs/infrastructure.typ
(delegate to denet). Then this file goes away.
"""

import ctypes

_TX, _RX = 197, 198


class _Val(ctypes.Union):
    _fields_ = [("d", ctypes.c_double), ("ui", ctypes.c_uint), ("ul", ctypes.c_ulong),
                ("ull", ctypes.c_ulonglong), ("sll", ctypes.c_longlong)]


class _Field(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_uint),
                ("nvmlReturn", ctypes.c_uint), ("value", _Val)]


def _open():
    try:
        nv = ctypes.CDLL("libnvidia-ml.so.1")
        h = ctypes.c_void_p()
        if nv.nvmlInit_v2() or nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)):
            return None
        return nv, h
    except OSError:
        return None


_nvml = _open()


def counters():
    """(tx_bytes, rx_bytes) since driver load, or None if NVML or the fields are unavailable."""
    if _nvml is None:
        return None
    nv, h = _nvml
    f = (_Field * 2)()
    f[0].fieldId, f[1].fieldId = _TX, _RX
    if nv.nvmlDeviceGetFieldValues(h, 2, f) or f[0].nvmlReturn or f[1].nvmlReturn:
        return None
    return f[0].value.ull, f[1].value.ull
