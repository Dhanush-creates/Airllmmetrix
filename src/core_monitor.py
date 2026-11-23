#!/usr/bin/env python3
"""SovereignLLMProfiler - core telemetry monitor and self-test harness.

A dedicated background thread samples the host every 50 ms (configurable), fully decoupled from the asyncio event
loop that drives inference:

  * system CPU utilisation      - computed from raw psutil.cpu_times() deltas (no shared psutil.cpu_percent state)
  * physical RAM                - available and in-use bytes
  * paging activity (pages/s)   - Windows PDH "\\Memory\\Pages/sec", Linux /proc/vmstat pswpin+pswpout,
                                  elsewhere psutil swap sin/sout deltas
  * swap / page-file in use     - polled at 1 Hz (psutil.swap_memory is comparatively expensive on Windows)
  * inference server footprint  - RSS of the server process tree, discovered by listening port, name or PID
  * profiler footprint          - this process's RSS, so the profiler can subtract itself from system deltas

A pre-warm window (default 3 s) captures background noise and yields a Baseline with explicit noise floors that
every later measurement is compared against.

Running this module directly executes the harness self-test: baseline the host, inject a known CPU + memory load in
a child process tracked as if it were an inference server, and verify the monitor recovers it within tolerance.

    python src/core_monitor.py --inject-mb 512 --duration 4
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import csv
import ctypes
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

try:
    import psutil
except ImportError as exc:  # pragma: no cover - hard dependency
    raise SystemExit("SovereignLLMProfiler requires psutil (pip install psutil)") from exc

__all__ = [
    "Baseline",
    "CoreMonitor",
    "MemoryDelta",
    "Sample",
    "SamplerHealth",
    "WindowStats",
    "isolate_memory",
    "percentile",
]

DEFAULT_INTERVAL_S = 0.05
DEFAULT_PREWARM_S = 3.0
SWAP_POLL_EVERY = 20          # samples between swap_memory() polls (1 s at 50 ms)
CHILD_RESCAN_EVERY = 20       # samples between server process-tree refreshes (1 s at 50 ms; ~4 ms each on Windows)
MAX_SAMPLES = 1_500_000       # ~20 h at 50 ms; oldest 10 % dropped beyond this
DEFAULT_SERVER_NAMES = ("ollama", "ollama_llama_server", "llama-server", "llama_server", "koboldcpp", "lms")
GIB = float(1 << 30)
MIB = float(1 << 20)


class Sample(NamedTuple):
    t: float
    cpu_pct: float
    ram_available: int
    ram_used: int
    paging_rate: float
    swap_used: int
    server_rss: int
    server_procs: int
    self_rss: int


# =============================================================================================
# Statistics helpers
# =============================================================================================


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(v)]


def percentile(values: Iterable[float], q: float) -> float:
    data = sorted(_finite(values))
    if not data:
        return math.nan
    k = (len(data) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def _mean(values: Iterable[float]) -> float:
    data = _finite(values)
    return math.fsum(data) / len(data) if data else math.nan


def _std(values: Iterable[float]) -> float:
    data = _finite(values)
    return statistics.pstdev(data) if len(data) > 1 else 0.0


# =============================================================================================
# CPU and paging probes
# =============================================================================================


class _CpuMeter:
    """System CPU % from raw cpu_times deltas, private state so other psutil users cannot skew it."""

    def __init__(self) -> None:
        self._last = self._read()

    @staticmethod
    def _read() -> tuple[float, float]:
        times = psutil.cpu_times()
        total = math.fsum(times) - getattr(times, "guest", 0.0) - getattr(times, "guest_nice", 0.0)
        idle = times.idle + getattr(times, "iowait", 0.0)
        return total, idle

    def percent(self) -> float:
        total, idle = self._read()
        d_total = total - self._last[0]
        d_idle = idle - self._last[1]
        self._last = (total, idle)
        if d_total <= 0:
            return math.nan
        return max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total)))


class PagingProbe:
    source = "unsupported"

    def rate(self, now: float) -> float:
        return math.nan

    def close(self) -> None:
        return None


class _PdhValue(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double), ("largeValue", ctypes.c_longlong)]


class _PdhFmtCounterValue(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("value", _PdhValue)]


class WindowsPdhPagingProbe(PagingProbe):
    """Hard-fault paging rate from the Windows performance counter \\Memory\\Pages/sec."""

    source = "PDH \\Memory\\Pages/sec"
    _PDH_FMT_DOUBLE = 0x00000200
    _VALID = (0x00000000, 0x00000001)  # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA

    def __init__(self) -> None:
        pdh = ctypes.WinDLL("pdh")  # type: ignore[attr-defined]
        pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_void_p)]
        pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        pdh.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                                                    ctypes.POINTER(_PdhFmtCounterValue)]
        pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        for fn in (pdh.PdhOpenQueryW, pdh.PdhAddEnglishCounterW, pdh.PdhCollectQueryData,
                   pdh.PdhGetFormattedCounterValue, pdh.PdhCloseQuery):
            fn.restype = ctypes.c_ulong
        self._pdh = pdh
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)) != 0:
            raise OSError("PdhOpenQueryW failed")
        if pdh.PdhAddEnglishCounterW(self._query, "\\Memory\\Pages/sec", None, ctypes.byref(self._counter)) != 0:
            pdh.PdhCloseQuery(self._query)
            raise OSError("PdhAddEnglishCounterW(\\Memory\\Pages/sec) failed")
        pdh.PdhCollectQueryData(self._query)

    def rate(self, now: float) -> float:
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return math.nan
        value = _PdhFmtCounterValue()
        status = self._pdh.PdhGetFormattedCounterValue(self._counter, self._PDH_FMT_DOUBLE, None, ctypes.byref(value))
        if status != 0 or value.CStatus not in self._VALID:
            return math.nan
        return float(value.value.doubleValue)

    def close(self) -> None:
        if self._query:
            self._pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()


class _CumulativePagingProbe(PagingProbe):
    def __init__(self) -> None:
        self._last: tuple[float, float] | None = None

    def _pages(self) -> float:
        raise NotImplementedError

    def rate(self, now: float) -> float:
        pages = self._pages()
        last, self._last = self._last, (now, pages)
        if last is None or now <= last[0]:
            return math.nan
        return max(0.0, (pages - last[1]) / (now - last[0]))


class LinuxVmstatPagingProbe(_CumulativePagingProbe):
    source = "/proc/vmstat pswpin+pswpout"

    def __init__(self) -> None:
        super().__init__()
        self._fh = open("/proc/vmstat", "rb")  # noqa: SIM115 - kept open for low-overhead re-reads

    def _pages(self) -> float:
        self._fh.seek(0)
        total = 0
        for line in self._fh.read().splitlines():
            if line.startswith((b"pswpin ", b"pswpout ")):
                total += int(line.split()[1])
        return float(total)

    def close(self) -> None:
        self._fh.close()


class PsutilSwapPagingProbe(_CumulativePagingProbe):
    source = "psutil swap sin+sout"

    def __init__(self) -> None:
        super().__init__()
        try:
            self._page = float(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, ValueError, OSError):
            self._page = 4096.0

    def _pages(self) -> float:
        swap = psutil.swap_memory()
        return (swap.sin + swap.sout) / self._page


def make_paging_probe() -> PagingProbe:
    if sys.platform == "win32":
        try:
            return WindowsPdhPagingProbe()
        except (OSError, AttributeError):
            return PagingProbe()
    if sys.platform.startswith("linux") and Path("/proc/vmstat").exists():
        try:
            return LinuxVmstatPagingProbe()
        except OSError:
            pass
    return PsutilSwapPagingProbe()


def _swap_used() -> int:
    try:
        return int(psutil.swap_memory().used)
    except (OSError, RuntimeError, psutil.Error):
        return 0


# =============================================================================================
# Inference server process tracking
# =============================================================================================


class ServerProcessTracker:
    """Tracks the inference server process tree (e.g. `ollama serve` plus its model runner children)."""

    def __init__(self, pids: Iterable[int] = (), names: Sequence[str] = DEFAULT_SERVER_NAMES,
                 port: int | None = None) -> None:
        self._seeds = set(pids)
        self._names = frozenset(n.lower() for n in names)
        self._port = port
        self._roots: set[int] = set()
        self._procs: dict[int, psutil.Process] = {}
        self._lock = threading.Lock()
        self._self_pid = os.getpid()
        self.discovery = "none"

    def add(self, pid: int) -> None:
        with self._lock:
            self._seeds.add(pid)
        self.refresh(roots=True)

    @staticmethod
    def _listening(port: int) -> set[int]:
        try:
            connections = psutil.net_connections(kind="tcp")
        except (psutil.AccessDenied, PermissionError, OSError):
            return set()
        return {c.pid for c in connections
                if c.pid and c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port}

    def _discover_roots(self) -> set[int]:
        with self._lock:
            roots = set(self._seeds)
        methods = ["pid"] if roots else []
        if self._port:
            by_port = self._listening(self._port)
            if by_port:
                roots |= by_port
                methods.append(f"port:{self._port}")
        if self._names and not (roots - self._seeds):
            for proc in psutil.process_iter(["name"]):
                name = (proc.info.get("name") or "").lower()
                stem = name[:-4] if name.endswith(".exe") else name
                if stem in self._names:
                    roots.add(proc.pid)
            if roots - self._seeds:
                methods.append("name")
        roots.discard(self._self_pid)
        self.discovery = "+".join(methods) or "none"
        return roots

    def refresh(self, roots: bool = False) -> None:
        root_pids = self._discover_roots() if roots else None
        with self._lock:
            if root_pids is not None:
                self._roots = root_pids
            current = dict(self._procs)
            targets = set(self._roots)
        alive: dict[int, psutil.Process] = {}
        for pid in targets:
            try:
                proc = current.get(pid) or psutil.Process(pid)
                if not proc.is_running():
                    continue
                alive[pid] = proc
                for child in proc.children(recursive=True):
                    alive[child.pid] = current.get(child.pid) or child
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        alive.pop(self._self_pid, None)
        with self._lock:
            self._procs = alive

    def rss(self) -> tuple[int, int]:
        with self._lock:
            procs = list(self._procs.items())
        total = 0
        count = 0
        dead = []
        for pid, proc in procs:
            try:
                total += proc.memory_info().rss
                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                dead.append(pid)
        if dead:
            with self._lock:
                for pid in dead:
                    self._procs.pop(pid, None)
        return total, count

    def pids(self) -> set[int]:
        with self._lock:
            return set(self._procs)

    def cpu_seconds(self) -> float:
        with self._lock:
            procs = list(self._procs.values())
        total = 0.0
        for proc in procs:
            try:
                times = proc.cpu_times()
                total += times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return total

    def describe(self) -> list[dict[str, object]]:
        with self._lock:
            procs = sorted(self._procs.items())
        described = []
        for pid, proc in procs:
            try:
                cmdline = " ".join(proc.cmdline()[:3])
                described.append({"pid": pid, "name": proc.name(), "cmdline": cmdline[:120],
                                  "rss": proc.memory_info().rss})
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return described


# =============================================================================================
# Aggregated telemetry
# =============================================================================================


@dataclass(frozen=True)
class WindowStats:
    t_start: float
    t_end: float
    samples: int
    cpu_mean: float
    cpu_p50: float
    cpu_p95: float
    cpu_max: float
    cpu_std: float
    ram_used_mean: float
    ram_used_min: int
    ram_used_peak: int
    ram_used_std: float
    ram_available_mean: float
    ram_available_min: int
    paging_rate_mean: float
    paging_rate_p95: float
    paging_rate_max: float
    paging_pages: float
    swap_used_start: int
    swap_used_end: int
    swap_used_peak: int
    server_rss_mean: float
    server_rss_min: int
    server_rss_peak: int
    server_procs_max: int
    self_rss_mean: float
    self_rss_peak: int

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @classmethod
    def from_samples(cls, samples: Sequence[Sample], t_start: float, t_end: float) -> WindowStats:
        if not samples:
            raise ValueError("telemetry window contains no samples")
        cpu = [s.cpu_pct for s in samples]
        used = [s.ram_used for s in samples]
        paging = [s.paging_rate for s in samples]
        server = [s.server_rss for s in samples]
        pages = 0.0
        for prev, cur in zip(samples, samples[1:]):
            if not math.isnan(cur.paging_rate):
                pages += cur.paging_rate * (cur.t - prev.t)
        return cls(
            t_start=t_start,
            t_end=t_end,
            samples=len(samples),
            cpu_mean=_mean(cpu),
            cpu_p50=percentile(cpu, 50),
            cpu_p95=percentile(cpu, 95),
            cpu_max=max(_finite(cpu), default=math.nan),
            cpu_std=_std(cpu),
            ram_used_mean=_mean(used),
            ram_used_min=min(used),
            ram_used_peak=max(used),
            ram_used_std=_std(used),
            ram_available_mean=_mean(s.ram_available for s in samples),
            ram_available_min=min(s.ram_available for s in samples),
            paging_rate_mean=_mean(paging),
            paging_rate_p95=percentile(paging, 95),
            paging_rate_max=max(_finite(paging), default=math.nan),
            paging_pages=pages,
            swap_used_start=samples[0].swap_used,
            swap_used_end=samples[-1].swap_used,
            swap_used_peak=max(s.swap_used for s in samples),
            server_rss_mean=_mean(server),
            server_rss_min=min(server),
            server_rss_peak=max(server),
            server_procs_max=max(s.server_procs for s in samples),
            self_rss_mean=_mean(s.self_rss for s in samples),
            self_rss_peak=max(s.self_rss for s in samples),
        )


@dataclass(frozen=True)
class Baseline:
    stats: WindowStats
    ram_noise_floor: float
    server_rss_noise_floor: float
    cpu_noise_floor: float
    paging_source: str


@dataclass(frozen=True)
class MemoryDelta:
    model_resident_delta: float
    source: str
    server_rss_delta: float | None
    system_delta: float
    profiler_delta: float
    isolated_system_delta: float
    noise_floor: float
    significant: bool


def isolate_memory(window: WindowStats, baseline: Baseline) -> MemoryDelta:
    """Peak active memory minus the noise baseline, with the profiler's own growth removed."""
    base = baseline.stats
    system_delta = window.ram_used_peak - base.ram_used_mean
    profiler_delta = max(0.0, window.self_rss_peak - base.self_rss_mean)
    isolated_system = system_delta - profiler_delta
    tracked = window.server_procs_max > 0 or base.server_procs_max > 0
    server_delta = (window.server_rss_peak - base.server_rss_mean) if tracked else None
    if server_delta is not None:
        primary, source, floor = server_delta, "server-process-rss", baseline.server_rss_noise_floor
    else:
        primary, source, floor = isolated_system, "system-ram-in-use", baseline.ram_noise_floor
    return MemoryDelta(primary, source, server_delta, system_delta, profiler_delta, isolated_system, floor,
                       abs(primary) > floor)


@dataclass(frozen=True)
class SamplerHealth:
    samples: int
    target_interval_ms: float
    mean_interval_ms: float
    p95_interval_ms: float
    max_interval_ms: float
    overruns: int
    sampler_cpu_s: float
    wall_s: float
    overhead_pct_of_core: float
    paging_source: str
    server_discovery: str


# =============================================================================================
# Monitor
# =============================================================================================


class CoreMonitor:
    """High-frequency background sampler. Use as a (async) context manager or call start()/stop()."""

    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S, *, server_pids: Iterable[int] = (),
                 server_names: Sequence[str] = DEFAULT_SERVER_NAMES, server_port: int | None = None) -> None:
        if not 0.005 <= interval_s <= 5.0:
            raise ValueError("interval_s must be between 5 ms and 5 s")
        self.interval_s = interval_s
        self._tracker = ServerProcessTracker(server_pids, server_names, server_port)
        self._samples: list[Sample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._self = psutil.Process()
        self._overruns = 0
        self._sampler_cpu = 0.0
        self._t_started = 0.0
        self._t_stopped = 0.0
        self.paging_source = "unsupported"
        self.baseline: Baseline | None = None

    # ---- lifecycle ----------------------------------------------------------------------
    def start(self) -> CoreMonitor:
        if self._thread is not None:
            raise RuntimeError("monitor already started")
        self._tracker.refresh(roots=True)
        self._thread = threading.Thread(target=self._run, name="slp-core-monitor", daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            raise RuntimeError("monitor thread failed to produce a first sample")
        if self._error is not None:
            raise RuntimeError("monitor thread failed during start-up") from self._error
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, 10 * self.interval_s))
        if self._error is not None:
            raise RuntimeError("monitor thread failed") from self._error

    def __enter__(self) -> CoreMonitor:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    async def __aenter__(self) -> CoreMonitor:
        return await asyncio.to_thread(self.start)

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self.stop)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- sampling loop ------------------------------------------------------------------
    def _run(self) -> None:
        probe: PagingProbe = PagingProbe()
        busy = 0.0
        try:
            probe = make_paging_probe()
            self.paging_source = probe.source
            cpu = _CpuMeter()
            swap_used = _swap_used()
            interval = self.interval_s
            self._t_started = time.perf_counter()
            next_tick = self._t_started + interval
            time.sleep(interval)
            n = 0
            while not self._stop.is_set():
                now = time.perf_counter()
                if n and n % SWAP_POLL_EVERY == 0:
                    swap_used = _swap_used()
                if n and n % CHILD_RESCAN_EVERY == 0:
                    self._tracker.refresh(roots=False)
                vm = psutil.virtual_memory()
                server_rss, server_procs = self._tracker.rss()
                sample = Sample(now, cpu.percent(), int(vm.available), int(vm.total - vm.available), probe.rate(now),
                                swap_used, server_rss, server_procs, int(self._self.memory_info().rss))
                with self._lock:
                    self._samples.append(sample)
                    if len(self._samples) > MAX_SAMPLES:
                        del self._samples[: MAX_SAMPLES // 10]
                if n == 0:
                    self._ready.set()
                n += 1
                # Busy wall time per tick (upper bound on sampler CPU); thread CPU clocks are tick-quantised on Windows.
                busy += time.perf_counter() - now
                self._sampler_cpu = busy
                next_tick += interval
                delay = next_tick - time.perf_counter()
                if delay <= 0:
                    self._overruns += 1
                    next_tick = time.perf_counter()
                else:
                    time.sleep(delay)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller via stop()
            self._error = exc
            self._ready.set()
        finally:
            self._sampler_cpu = busy
            self._t_stopped = time.perf_counter()
            probe.close()

    # ---- queries ------------------------------------------------------------------------
    def samples(self, t_start: float | None = None, t_end: float | None = None) -> list[Sample]:
        with self._lock:
            data = self._samples
            lo = 0 if t_start is None else bisect.bisect_left(data, t_start, key=lambda s: s.t)
            hi = len(data) if t_end is None else bisect.bisect_right(data, t_end, key=lambda s: s.t)
            return data[lo:hi]

    def latest(self) -> Sample:
        with self._lock:
            if not self._samples:
                raise RuntimeError("no samples collected yet")
            return self._samples[-1]

    def window(self, t_start: float, t_end: float) -> WindowStats | None:
        data = self.samples(t_start, t_end)
        return WindowStats.from_samples(data, t_start, t_end) if data else None

    def track_pid(self, pid: int) -> None:
        self._tracker.add(pid)

    def rescan_server(self) -> None:
        self._tracker.refresh(roots=True)

    def server_cpu_seconds(self) -> float:
        return self._tracker.cpu_seconds()

    def server_processes(self) -> list[dict[str, object]]:
        return self._tracker.describe()

    @property
    def server_discovery(self) -> str:
        return self._tracker.discovery

    # ---- baseline -----------------------------------------------------------------------
    def _baseline_from(self, t_start: float, t_end: float) -> Baseline:
        settle = min(0.25, (t_end - t_start) / 10.0)
        stats = self.window(t_start + settle, t_end)
        if stats is None or stats.samples < 3:
            raise RuntimeError("pre-warm window produced too few samples")
        data = self.samples(t_start + settle, t_end)
        server = [s.server_rss for s in data]
        self.baseline = Baseline(
            stats=stats,
            ram_noise_floor=max(3.0 * stats.ram_used_std, float(stats.ram_used_peak - stats.ram_used_min), 16 * MIB),
            server_rss_noise_floor=max(3.0 * _std(server), float(max(server) - min(server)), 8 * MIB),
            cpu_noise_floor=max(0.0, stats.cpu_p95 - stats.cpu_mean),
            paging_source=self.paging_source,
        )
        return self.baseline

    def prewarm(self, seconds: float = DEFAULT_PREWARM_S) -> Baseline:
        if not self.running:
            raise RuntimeError("start the monitor before pre-warming")
        t_start = time.perf_counter()
        time.sleep(seconds)
        return self._baseline_from(t_start, time.perf_counter())

    async def aprewarm(self, seconds: float = DEFAULT_PREWARM_S) -> Baseline:
        if not self.running:
            raise RuntimeError("start the monitor before pre-warming")
        t_start = time.perf_counter()
        await asyncio.sleep(seconds)
        return self._baseline_from(t_start, time.perf_counter())

    # ---- health & export ----------------------------------------------------------------
    def health(self) -> SamplerHealth:
        data = self.samples()
        gaps = [(b.t - a.t) * 1000.0 for a, b in zip(data, data[1:])]
        end = self._t_stopped if self._t_stopped else time.perf_counter()
        wall = max(1e-9, end - self._t_started) if self._t_started else 0.0
        return SamplerHealth(
            samples=len(data),
            target_interval_ms=self.interval_s * 1000.0,
            mean_interval_ms=_mean(gaps),
            p95_interval_ms=percentile(gaps, 95),
            max_interval_ms=max(gaps, default=math.nan),
            overruns=self._overruns,
            sampler_cpu_s=self._sampler_cpu,
            wall_s=wall,
            overhead_pct_of_core=100.0 * self._sampler_cpu / wall if wall else math.nan,
            paging_source=self.paging_source,
            server_discovery=self._tracker.discovery,
        )

    def export_csv(self, path: Path, phases: Sequence[tuple[str, float, float]] = ()) -> int:
        data = self.samples()
        ordered = sorted(phases, key=lambda p: p[1])
        origin = data[0].t if data else 0.0
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "phase", "cpu_pct", "ram_used_mib", "ram_available_mib", "paging_pages_per_s",
                             "swap_used_mib", "server_rss_mib", "server_procs", "profiler_rss_mib"])
            for s in data:
                phase = ""
                for name, t0, t1 in ordered:
                    if t0 <= s.t <= t1:
                        phase = name
                writer.writerow([f"{s.t - origin:.3f}", phase, f"{s.cpu_pct:.2f}", f"{s.ram_used / MIB:.1f}",
                                 f"{s.ram_available / MIB:.1f}", "" if math.isnan(s.paging_rate) else f"{s.paging_rate:.1f}",
                                 f"{s.swap_used / MIB:.1f}", f"{s.server_rss / MIB:.1f}", s.server_procs,
                                 f"{s.self_rss / MIB:.1f}"])
        os.replace(tmp, path)
        return len(data)


# =============================================================================================
# Self-test harness
# =============================================================================================

_LOAD_CHILD = r"""
import sys, time
mib, hold = int(sys.argv[1]), float(sys.argv[2])
buf = bytearray(mib * 1024 * 1024)
for i in range(0, len(buf), 4096):
    buf[i] = 1
print("ready", flush=True)
end = time.perf_counter() + hold
x = 0
while time.perf_counter() < end:
    for _ in range(20000):
        x += 1
"""


def _fmt(value: float, unit: str = "", digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}{unit}"


def run_self_test(args: argparse.Namespace) -> int:
    logical = psutil.cpu_count(logical=True) or 1
    expected_bytes = args.inject_mb * MIB
    print(f"SovereignLLMProfiler core monitor self-test | interval {args.interval * 1000:.0f} ms | "
          f"pre-warm {args.prewarm:.1f} s | inject {args.inject_mb} MiB + 1 busy core for {args.duration:.1f} s")
    with CoreMonitor(args.interval, server_names=()) as monitor:
        baseline = monitor.prewarm(args.prewarm)
        b = baseline.stats
        print(f"baseline  : cpu {_fmt(b.cpu_mean, '%', 1)} (p95 {_fmt(b.cpu_p95, '%', 1)}) | ram used "
              f"{b.ram_used_mean / GIB:.2f} GiB +/- {b.ram_used_std / MIB:.1f} MiB | noise floor "
              f"{baseline.ram_noise_floor / MIB:.1f} MiB | paging {_fmt(b.paging_rate_mean, ' pg/s', 1)} "
              f"[{baseline.paging_source}] | samples {b.samples}")
        child = subprocess.Popen([sys.executable, "-c", _LOAD_CHILD, str(args.inject_mb), str(args.duration)],
                                 stdout=subprocess.PIPE, text=True)
        try:
            monitor.track_pid(child.pid)
            if (child.stdout.readline() if child.stdout else "").strip() != "ready":
                print("FAIL: load generator did not start")
                return 1
            t_ready = time.perf_counter()
            child.wait(timeout=args.duration + 30)
        finally:
            if child.poll() is None:
                child.kill()
        window = monitor.window(t_ready + 0.15, t_ready + args.duration - 0.15)
        if window is None:
            print("FAIL: no samples captured during the load window")
            return 1
        delta = isolate_memory(window, baseline)
    health = monitor.health()
    cpu_gain = window.cpu_mean - b.cpu_mean
    expected_cpu = 100.0 / logical
    mem_tol = max(0.15 * expected_bytes, 32 * MIB)
    sys_tol = max(0.35 * expected_bytes, baseline.ram_noise_floor * 2)
    checks = [
        ("server RSS delta recovers injected memory", abs((delta.server_rss_delta or 0.0) - expected_bytes) <= mem_tol,
         f"{(delta.server_rss_delta or 0.0) / MIB:.1f} MiB vs {args.inject_mb} MiB (+/- {mem_tol / MIB:.0f})"),
        ("isolated system RAM delta recovers injected memory", abs(delta.isolated_system_delta - expected_bytes) <= sys_tol,
         f"{delta.isolated_system_delta / MIB:.1f} MiB vs {args.inject_mb} MiB (+/- {sys_tol / MIB:.0f})"),
        ("CPU load above baseline detected", cpu_gain >= 0.5 * expected_cpu,
         f"+{cpu_gain:.1f} pp vs >= {0.5 * expected_cpu:.1f} pp (one of {logical} logical cores)"),
        ("sampling cadence on target", abs(health.mean_interval_ms - health.target_interval_ms) <= 0.1 * health.target_interval_ms,
         f"mean {health.mean_interval_ms:.2f} ms, p95 {health.p95_interval_ms:.2f} ms, max {health.max_interval_ms:.1f} ms, "
         f"overruns {health.overruns}"),
        ("sampler overhead below 3% of one core", health.overhead_pct_of_core < 3.0,
         f"{health.overhead_pct_of_core:.2f}% ({health.sampler_cpu_s * 1000:.0f} ms busy over {health.wall_s:.1f} s)"),
    ]
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    passed = all(ok for _, ok, _ in checks)
    print("self-test:", "PASSED" if passed else "FAILED")
    if args.json:
        import json

        print(json.dumps({"baseline": asdict(baseline), "window": asdict(window), "delta": asdict(delta),
                          "health": asdict(health)}, indent=2, default=str))
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SovereignLLMProfiler core monitor self-test harness")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="sampling interval in seconds")
    parser.add_argument("--prewarm", type=float, default=DEFAULT_PREWARM_S, help="noise baseline window in seconds")
    parser.add_argument("--inject-mb", type=int, default=512, help="MiB allocated by the synthetic load process")
    parser.add_argument("--duration", type=float, default=4.0, help="seconds the synthetic load is held")
    parser.add_argument("--json", action="store_true", help="also print raw telemetry as JSON")
    args = parser.parse_args(argv)
    if args.duration < 1.0 or args.inject_mb < 16:
        parser.error("--duration must be >= 1 s and --inject-mb >= 16")
    return run_self_test(args)


if __name__ == "__main__":
    sys.exit(main())
