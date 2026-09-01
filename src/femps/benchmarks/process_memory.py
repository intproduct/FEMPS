"""Dependency-free process resident-memory sampling for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
import sys
import threading


def _windows_rss_bytes() -> int:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
            ("quota_nonpaged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not success:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.working_set_size)


def current_process_rss_bytes() -> int:
    """Return current resident set size without an optional runtime dependency."""

    if sys.platform == "win32":
        return _windows_rss_bytes()
    statm = "/proc/self/statm"
    if os.path.exists(statm):
        with open(statm, encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    import resource

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


@dataclass(frozen=True, slots=True)
class ProcessRSSRecord:
    """Sampled process-memory record for one benchmark call."""

    method: str
    sampling_interval_seconds: float
    samples: int
    baseline_rss_bytes: int
    peak_rss_bytes: int
    final_rss_bytes: int
    peak_delta_rss_bytes: int

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "method": self.method,
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "samples": self.samples,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "final_rss_bytes": self.final_rss_bytes,
            "peak_delta_rss_bytes": self.peak_delta_rss_bytes,
        }


class ProcessRSSMonitor:
    """Poll process RSS in a daemon thread and retain the sampled maximum."""

    def __init__(self, sampling_interval_seconds: float = 0.005) -> None:
        if sampling_interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        self.sampling_interval_seconds = float(sampling_interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0
        self._peak = 0
        self._final = 0
        self._samples = 0

    def _sample(self) -> None:
        resident = current_process_rss_bytes()
        self._samples += 1
        self._peak = max(self._peak, resident)
        self._final = resident

    def _poll(self) -> None:
        while not self._stop_event.wait(self.sampling_interval_seconds):
            self._sample()

    def __enter__(self) -> ProcessRSSMonitor:
        if self._thread is not None:
            raise RuntimeError("memory monitor is already running")
        self._sample()
        self._baseline = self._final
        self._thread = threading.Thread(
            target=self._poll, name="femps-rss-monitor", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()

    def record(self) -> ProcessRSSRecord:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("stop the memory monitor before reading its record")
        return ProcessRSSRecord(
            method="sampled_process_resident_set",
            sampling_interval_seconds=self.sampling_interval_seconds,
            samples=self._samples,
            baseline_rss_bytes=self._baseline,
            peak_rss_bytes=self._peak,
            final_rss_bytes=self._final,
            peak_delta_rss_bytes=max(0, self._peak - self._baseline),
        )
