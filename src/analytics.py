#!/usr/bin/env python3
"""SovereignLLMProfiler - streaming inference analytics engine.

Drives a local streaming LLM endpoint from an asyncio client built on a zero-dependency HTTP/1.1 stream reader
(Ollama /api/generate NDJSON, or OpenAI-compatible /v1/completions SSE such as llama.cpp server) while
core_monitor samples the host every 50 ms on a background thread.

Metrics
  * Baseline system overhead  - CPU / RAM / paging of background noise over a 3 s pre-warm with the model unloaded
  * Model resident RAM delta  - peak RSS of the inference server process tree minus its baseline (primary), and peak
                                system RAM-in-use minus baseline minus profiler growth (cross-check)
  * Time to first token       - perf_counter delta between request write and the first content-bearing stream chunk
  * Output evaluation rate    - server eval_count / eval_duration, plus client-side streaming rate and inter-token
                                latency percentiles from chunk arrival timestamps

Context stress matrix: Short (100), Medium (1000) and Extended (4000) prompt tokens by default (minimum 3 tiers).
Prompt sizes are calibrated against the server's own tokenizer with two probe requests, and every prompt starts with
a unique nonce so KV-cache prefix reuse cannot flatter TTFT.

Outputs
  benchmarks/profiles/system_scorecard.md     human-readable scorecard
  benchmarks/profiles/system_profile.json     aggregated telemetry vectors
  benchmarks/profiles/telemetry_samples.csv   every sample, labelled by phase

Usage
  python src/analytics.py --model qwen2.5-coder:3b
  python src/analytics.py --endpoint http://localhost:11434/api/generate --model qwen3:8b --repeats 3
  python src/analytics.py --endpoint http://127.0.0.1:8080/v1/completions --model local

Exit codes: 0 all runs succeeded | 2 some runs failed | 3 endpoint unavailable | 4 configuration error
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import os
import platform
import random
import socket
import ssl
import statistics
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import psutil

try:
    from .core_monitor import (GIB, MIB, Baseline, CoreMonitor, MemoryDelta, SamplerHealth, WindowStats,
                               isolate_memory, percentile)
except ImportError:
    from core_monitor import (GIB, MIB, Baseline, CoreMonitor, MemoryDelta, SamplerHealth,  # type: ignore[no-redef]
                              WindowStats, isolate_memory, percentile)

__version__ = "0.1.0"
ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "benchmarks" / "profiles"
DEFAULT_SCORECARD = PROFILE_DIR / "system_scorecard.md"

log = logging.getLogger("slp.analytics")


# =============================================================================================
# Errors
# =============================================================================================


class ProfilerError(Exception):
    exit_code = 1


class EndpointError(ProfilerError):
    exit_code = 3


class ConfigError(ProfilerError):
    exit_code = 4


# =============================================================================================
# Endpoint & zero-dependency async HTTP/1.1 streaming client
# =============================================================================================


def is_loopback_host(host: str) -> bool:
    h = host.strip("[]").split("%", 1)[0].lower()
    if h in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Endpoint:
    url: str
    scheme: str
    host: str
    port: int
    root: str
    api_hint: str | None

    @property
    def host_header(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        default = 443 if self.scheme == "https" else 80
        return host if self.port == default else f"{host}:{self.port}"

    @property
    def loopback(self) -> bool:
        return is_loopback_host(self.host)

    @classmethod
    def parse(cls, url: str, allow_remote: bool) -> Endpoint:
        parts = urlsplit(url if "://" in url else f"http://{url}")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigError(f"unsupported endpoint {url!r}; expected http(s)://host:port")
        path = parts.path.rstrip("/")
        hint = None
        for suffix, api in (("/api/generate", "ollama"), ("/api/chat", "ollama"), ("/v1/completions", "openai"),
                            ("/v1/chat/completions", "openai"), ("/v1", "openai"), ("/api", "ollama")):
            if path.endswith(suffix):
                path, hint = path[: -len(suffix)], api
                break
        endpoint = cls(url, parts.scheme, parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                       path, hint)
        if not endpoint.loopback and not allow_remote:
            raise ConfigError(f"{parts.hostname} is not a loopback address; SovereignLLMProfiler profiles local "
                              "endpoints only (pass --allow-remote to override)")
        return endpoint


class HTTPStatusError(EndpointError):
    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(f"HTTP {status}: {body[:300].decode('utf-8', 'replace').strip()}")
        self.status = status


def _parse_json_body(raw: bytes) -> Any:
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            raise
        return json.loads(lines[-1])


class AsyncLocalHTTP:
    """Minimal HTTP/1.1 client on asyncio streams: one connection per request, chunk-accurate timestamps."""

    def __init__(self, endpoint: Endpoint, allow_remote: bool, connect_timeout: float = 5.0) -> None:
        self.endpoint = endpoint
        self._allow_remote = allow_remote
        self._connect_timeout = connect_timeout
        self._addr: tuple[str, int] | None = None
        self._ssl = ssl.create_default_context() if endpoint.scheme == "https" else None
        self.connect_ms: list[float] = []

    async def _resolve(self) -> tuple[str, int]:
        if self._addr is not None:
            return self._addr
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(self.endpoint.host, self.endpoint.port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise EndpointError(f"cannot resolve {self.endpoint.host}: {exc}") from exc
        infos.sort(key=lambda info: 0 if info[0] == socket.AF_INET else 1)  # avoid slow ::1 refusals on Windows
        errors = []
        for _family, _type, _proto, _name, sockaddr in infos:
            ip, port = sockaddr[0], sockaddr[1]
            if not self._allow_remote and not ipaddress.ip_address(ip.split("%", 1)[0]).is_loopback:
                errors.append(f"{ip}: not loopback")
                continue
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, port, ssl=self._ssl,
                                            server_hostname=self.endpoint.host if self._ssl else None),
                    self._connect_timeout)
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
                self._addr = (ip, port)
                return self._addr
            except (OSError, asyncio.TimeoutError) as exc:
                errors.append(f"{ip}:{port} {exc or type(exc).__name__}")
        raise EndpointError(f"cannot connect to {self.endpoint.url}: {'; '.join(errors) or 'no addresses'}")

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        ip, port = await self._resolve()
        started = time.perf_counter()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port, ssl=self._ssl, limit=1 << 24,
                                        server_hostname=self.endpoint.host if self._ssl else None),
                self._connect_timeout)
        except (OSError, asyncio.TimeoutError) as exc:
            raise EndpointError(f"connection to {ip}:{port} failed: {exc or type(exc).__name__}") from exc
        sock = writer.get_extra_info("socket")
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connect_ms.append((time.perf_counter() - started) * 1000.0)
        return reader, writer

    def _head(self, method: str, path: str, body_len: int | None, accept: str) -> bytes:
        lines = [f"{method} {path} HTTP/1.1", f"Host: {self.endpoint.host_header}",
                 f"User-Agent: SovereignLLMProfiler/{__version__}", f"Accept: {accept}", "Connection: close"]
        if body_len is not None:
            lines += ["Content-Type: application/json", f"Content-Length: {body_len}"]
        return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")

    @staticmethod
    async def _read_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
        status_line = await reader.readline()
        if not status_line:
            raise EndpointError("connection closed before an HTTP response was received")
        try:
            status = int(status_line.split(b" ", 2)[1])
        except (IndexError, ValueError) as exc:
            raise EndpointError(f"malformed status line {status_line[:80]!r}") from exc
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode("latin-1").partition(":")
            headers[key.strip().lower()] = value.strip()
        return status, headers

    @staticmethod
    async def _body(reader: asyncio.StreamReader, headers: dict[str, str],
                    timeout: float) -> AsyncIterator[tuple[float, bytes]]:
        if "chunked" in headers.get("transfer-encoding", "").lower():
            while True:
                size_line = await asyncio.wait_for(reader.readline(), timeout)
                if not size_line:
                    raise EndpointError("truncated chunked response")
                size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
                if size == 0:
                    while (await asyncio.wait_for(reader.readline(), timeout)) not in (b"\r\n", b"\n", b""):
                        pass
                    return
                data = await asyncio.wait_for(reader.readexactly(size + 2), timeout)
                yield time.perf_counter(), data[:-2]
        else:
            remaining = int(headers["content-length"]) if "content-length" in headers else None
            while remaining is None or remaining > 0:
                data = await asyncio.wait_for(reader.read(65536 if remaining is None else min(65536, remaining)), timeout)
                if not data:
                    break
                if remaining is not None:
                    remaining -= len(data)
                yield time.perf_counter(), data

    @staticmethod
    async def _close(writer: asyncio.StreamWriter) -> None:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    async def request_json(self, method: str, path: str, payload: Any = None, timeout: float = 30.0) -> Any:
        reader, writer = await self._connect()
        try:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else None
            writer.write(self._head(method, path, len(body) if body is not None else None, "application/json")
                         + (body or b""))
            await writer.drain()
            status, headers = await asyncio.wait_for(self._read_head(reader), timeout)
            raw = bytearray()
            async for _, piece in self._body(reader, headers, timeout):
                raw += piece
            if status >= 400:
                raise HTTPStatusError(status, bytes(raw))
            return _parse_json_body(bytes(raw))
        except asyncio.TimeoutError as exc:
            raise EndpointError(f"{method} {path} timed out after {timeout:.0f}s") from exc
        finally:
            await self._close(writer)

    async def stream_lines(self, method: str, path: str, payload: Any, timeout: float,
                           marks: dict[str, float]) -> AsyncIterator[tuple[float, bytes]]:
        reader, writer = await self._connect()
        try:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            writer.write(self._head(method, path, len(body), "application/x-ndjson, text/event-stream") + body)
            await writer.drain()
            marks["sent"] = time.perf_counter()
            status, headers = await asyncio.wait_for(self._read_head(reader), timeout)
            marks["headers"] = time.perf_counter()
            if status >= 400:
                raw = bytearray()
                async for _, piece in self._body(reader, headers, timeout):
                    raw += piece
                raise HTTPStatusError(status, bytes(raw))
            buffer = bytearray()
            async for t_recv, piece in self._body(reader, headers, timeout):
                buffer += piece
                while (idx := buffer.find(b"\n")) >= 0:
                    line = bytes(buffer[:idx]).rstrip(b"\r")
                    del buffer[: idx + 1]
                    if line:
                        yield t_recv, line
            if buffer.strip():
                yield time.perf_counter(), bytes(buffer).strip()
        except asyncio.TimeoutError as exc:
            raise EndpointError(f"stream stalled for more than {timeout:.0f}s") from exc
        finally:
            await self._close(writer)


# =============================================================================================
# API adapters
# =============================================================================================


@dataclass(frozen=True)
class ServerStats:
    prompt_tokens: int | None = None
    prompt_eval_s: float | None = None
    output_tokens: int | None = None
    eval_s: float | None = None
    load_s: float | None = None
    total_s: float | None = None


@dataclass(frozen=True)
class StreamEvent:
    text: str
    done: bool
    stats: ServerStats | None = None


def _ns(value: Any) -> float | None:
    return float(value) / 1e9 if isinstance(value, (int, float)) and value > 0 else None


class APIAdapter(ABC):
    name = "api"
    supports_num_ctx = False
    supports_unload = False

    def __init__(self, root: str) -> None:
        self.root = root

    @abstractmethod
    async def server_version(self, http: AsyncLocalHTTP) -> str: ...

    @abstractmethod
    async def list_models(self, http: AsyncLocalHTTP) -> list[str]: ...

    @abstractmethod
    def generate_request(self, model: str, prompt: str, *, num_predict: int, num_ctx: int, seed: int,
                         keep_alive: str, cpu_only: bool) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def parse(self, line: bytes) -> StreamEvent | None: ...

    async def unload(self, http: AsyncLocalHTTP, model: str) -> bool:
        return False

    async def resident_report(self, http: AsyncLocalHTTP, model: str) -> dict[str, Any] | None:
        return None

    @staticmethod
    def resolve_model(requested: str | None, available: list[str]) -> str:
        if requested:
            if not available or requested in available:
                return requested
            if f"{requested}:latest" in available:
                return f"{requested}:latest"
            raise EndpointError(f"model {requested!r} is not available on the server; available: {', '.join(available)}")
        if not available:
            raise ConfigError("the server lists no models; pass --model")
        return available[0]


class OllamaAdapter(APIAdapter):
    name = "ollama"
    supports_num_ctx = True
    supports_unload = True

    async def server_version(self, http: AsyncLocalHTTP) -> str:
        data = await http.request_json("GET", f"{self.root}/api/version", timeout=10)
        return f"ollama {data.get('version', '?')}"

    async def list_models(self, http: AsyncLocalHTTP) -> list[str]:
        data = await http.request_json("GET", f"{self.root}/api/tags", timeout=15)
        return [m.get("name") or m.get("model") for m in data.get("models", []) if m.get("name") or m.get("model")]

    def generate_request(self, model: str, prompt: str, *, num_predict: int, num_ctx: int, seed: int,
                         keep_alive: str, cpu_only: bool) -> tuple[str, dict[str, Any]]:
        options: dict[str, Any] = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0.0, "top_k": 1,
                                   "top_p": 1.0, "seed": seed}
        if cpu_only:
            options["num_gpu"] = 0
        return f"{self.root}/api/generate", {"model": model, "prompt": prompt, "stream": True,
                                             "keep_alive": keep_alive, "options": options}

    async def resident_report(self, http: AsyncLocalHTTP, model: str) -> dict[str, Any] | None:
        data = await http.request_json("GET", f"{self.root}/api/ps", timeout=10)
        for entry in data.get("models", []):
            if model in (entry.get("name"), entry.get("model")):
                size = int(entry.get("size") or 0)
                vram = int(entry.get("size_vram") or 0)
                return {"size_bytes": size, "vram_bytes": vram, "host_bytes": max(0, size - vram),
                        "context_length": entry.get("context_length"),
                        "gpu_fraction": (vram / size) if size else 0.0}
        return None

    def parse(self, line: bytes) -> StreamEvent | None:
        obj = json.loads(line)
        if "error" in obj:
            raise EndpointError(f"server error: {obj['error']}")
        text = obj.get("response") or obj.get("thinking") or ""
        if obj.get("done"):
            return StreamEvent(text, True, ServerStats(
                prompt_tokens=obj.get("prompt_eval_count"), prompt_eval_s=_ns(obj.get("prompt_eval_duration")),
                output_tokens=obj.get("eval_count"), eval_s=_ns(obj.get("eval_duration")),
                load_s=_ns(obj.get("load_duration")), total_s=_ns(obj.get("total_duration"))))
        return StreamEvent(text, False)

    async def unload(self, http: AsyncLocalHTTP, model: str) -> bool:
        await http.request_json("POST", f"{self.root}/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
        return True


class OpenAICompatAdapter(APIAdapter):
    name = "openai"

    async def server_version(self, http: AsyncLocalHTTP) -> str:
        with contextlib.suppress(EndpointError):
            props = await http.request_json("GET", f"{self.root}/props", timeout=5)
            build = props.get("build_info") or props.get("model_path") or ""
            return f"openai-compatible ({build})" if build else "openai-compatible (llama.cpp)"
        return "openai-compatible"

    async def list_models(self, http: AsyncLocalHTTP) -> list[str]:
        data = await http.request_json("GET", f"{self.root}/v1/models", timeout=15)
        return [m.get("id") for m in data.get("data", []) if m.get("id")]

    def generate_request(self, model: str, prompt: str, *, num_predict: int, num_ctx: int, seed: int,
                         keep_alive: str, cpu_only: bool) -> tuple[str, dict[str, Any]]:
        return f"{self.root}/v1/completions", {
            "model": model, "prompt": prompt, "max_tokens": num_predict, "temperature": 0.0, "top_p": 1.0,
            "seed": seed, "stream": True, "stream_options": {"include_usage": True}, "cache_prompt": False,
        }

    def parse(self, line: bytes) -> StreamEvent | None:
        if not line.startswith(b"data:"):
            return None
        data = line[5:].strip()
        if data == b"[DONE]":
            return StreamEvent("", True)
        obj = json.loads(data)
        if "error" in obj:
            raise EndpointError(f"server error: {obj['error']}")
        choices = obj.get("choices") or []
        text = ""
        if choices:
            text = choices[0].get("text") or (choices[0].get("delta") or {}).get("content") or ""
        usage = obj.get("usage") or {}
        timings = obj.get("timings") or {}
        stats = None
        if usage or timings:
            stats = ServerStats(
                prompt_tokens=usage.get("prompt_tokens") or timings.get("prompt_n"),
                prompt_eval_s=(timings["prompt_ms"] / 1000.0) if timings.get("prompt_ms") else None,
                output_tokens=usage.get("completion_tokens") or timings.get("predicted_n"),
                eval_s=(timings["predicted_ms"] / 1000.0) if timings.get("predicted_ms") else None)
        return StreamEvent(text, False, stats)


async def select_adapter(api: str, endpoint: Endpoint, http: AsyncLocalHTTP) -> APIAdapter:
    choice = endpoint.api_hint if api == "auto" else api
    if choice == "ollama":
        return OllamaAdapter(endpoint.root)
    if choice == "openai":
        return OpenAICompatAdapter(endpoint.root)
    try:
        await http.request_json("GET", f"{endpoint.root}/api/version", timeout=5)
        return OllamaAdapter(endpoint.root)
    except HTTPStatusError:
        return OpenAICompatAdapter(endpoint.root)


# =============================================================================================
# Streaming inference client
# =============================================================================================


@dataclass
class GenerationTrace:
    t_send: float = math.nan
    t_headers: float = math.nan
    t_done: float = math.nan
    token_times: list[float] = field(default_factory=list)
    output_chars: int = 0
    server: ServerStats = field(default_factory=ServerStats)
    error: str | None = None

    @property
    def ttft_s(self) -> float:
        return self.token_times[0] - self.t_send if self.token_times else math.nan

    @property
    def stream_chunks(self) -> int:
        return len(self.token_times)

    @property
    def client_tps(self) -> float:
        if len(self.token_times) < 2:
            return math.nan
        span = self.token_times[-1] - self.token_times[0]
        return (len(self.token_times) - 1) / span if span > 0 else math.nan

    @property
    def server_tps(self) -> float:
        s = self.server
        return s.output_tokens / s.eval_s if s.output_tokens and s.eval_s else math.nan

    @property
    def prompt_tps(self) -> float:
        s = self.server
        return s.prompt_tokens / s.prompt_eval_s if s.prompt_tokens and s.prompt_eval_s else math.nan

    @property
    def itl_ms(self) -> list[float]:
        return [(b - a) * 1000.0 for a, b in zip(self.token_times, self.token_times[1:])]


class StreamingInferenceClient:
    def __init__(self, http: AsyncLocalHTTP, adapter: APIAdapter, model: str, *, num_ctx: int, seed: int,
                 keep_alive: str, timeout_s: float, cpu_only: bool) -> None:
        self.http = http
        self.adapter = adapter
        self.model = model
        self.num_ctx = num_ctx
        self.seed = seed
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        self.cpu_only = cpu_only

    async def generate(self, prompt: str, num_predict: int) -> GenerationTrace:
        path, payload = self.adapter.generate_request(self.model, prompt, num_predict=num_predict,
                                                      num_ctx=self.num_ctx, seed=self.seed, keep_alive=self.keep_alive,
                                                      cpu_only=self.cpu_only)
        trace = GenerationTrace()
        marks: dict[str, float] = {}
        stream = self.http.stream_lines("POST", path, payload, self.timeout_s, marks)
        try:
            async with contextlib.aclosing(stream) as lines:
                async for t_recv, line in lines:
                    if math.isnan(trace.t_send):
                        trace.t_send = marks.get("sent", math.nan)
                        trace.t_headers = marks.get("headers", math.nan)
                    try:
                        event = self.adapter.parse(line)
                    except json.JSONDecodeError:
                        continue
                    if event is None:
                        continue
                    if event.text:
                        trace.token_times.append(t_recv)
                        trace.output_chars += len(event.text)
                    if event.stats is not None:
                        trace.server = event.stats
                    if event.done:
                        trace.t_done = t_recv
                        break
        except ProfilerError as exc:
            trace.error = str(exc)
        if math.isnan(trace.t_send):
            trace.t_send = marks.get("sent", math.nan)
            trace.t_headers = marks.get("headers", math.nan)
        if math.isnan(trace.t_done) and trace.error is None:
            trace.error = "stream ended without a completion marker"
        return trace


# =============================================================================================
# Deterministic context-sized prompts
# =============================================================================================

_SUBJECTS = ("the scheduler", "the allocator", "the page cache", "the tensor kernel", "the attention block",
             "the decoder", "the NUMA node", "the memory controller", "the thread pool", "the KV cache",
             "the quantised matmul", "the DMA engine", "the interrupt handler", "the prefetcher", "the L2 cache")
_VERBS = ("throttled", "rebalanced", "evicted", "prefetched", "serialised", "pinned", "compacted", "flushed",
          "vectorised", "batched", "migrated", "coalesced", "spilled", "mapped", "stalled on")
_OBJECTS = ("cold pages", "dirty cache lines", "hot weights", "stale activations", "pending interrupts",
            "the token batch", "large pages", "the ring buffer", "fragmented blocks", "the residual stream")
_CONTEXTS = ("under sustained load", "during prompt evaluation", "at the decode boundary", "after a context switch",
             "while the governor scaled frequency", "between two inference passes", "when swap pressure rose")


class PromptFactory:
    """Builds unique prompts of a target token length, calibrated against the server's tokenizer."""

    HEADER = "Profiling session {nonce}. Read the systems log below.\n\n"
    INSTRUCTION = "\n\nTask: summarise the systems log above in one short paragraph."

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.slope = 0.25
        self.intercept = 24.0
        self.calibrated = False
        self.calibration_points: list[tuple[int, int]] = []

    def corpus(self, chars: int, salt: str) -> str:
        rng = random.Random(f"{self.seed}:{salt}")
        parts: list[str] = []
        size = 0
        while size < chars:
            sentence = (f"At step {rng.randint(1, 9999)}, {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
                        f"{rng.choice(_OBJECTS)} {rng.choice(_CONTEXTS)} at {rng.randint(1, 64)} GB/s. ")
            parts.append(sentence)
            size += len(sentence)
        text = "".join(parts)[:chars]
        cut = text.rfind(" ")
        return text[:cut] if cut > chars * 0.9 else text

    def raw(self, body_chars: int, nonce: str) -> str:
        return self.HEADER.format(nonce=nonce) + self.corpus(body_chars, nonce) + self.INSTRUCTION

    def calibrate(self, points: list[tuple[int, int]]) -> None:
        self.calibration_points = points
        if len(points) < 2:
            return
        (c1, t1), (c2, t2) = points[0], points[-1]
        if c2 == c1:
            return
        slope = (t2 - t1) / (c2 - c1)
        if 0.05 <= slope <= 1.0:
            self.slope = slope
            self.intercept = t1 - slope * c1
            self.calibrated = True

    def build(self, target_tokens: int, nonce: str) -> str:
        body_tokens = max(8.0, target_tokens - self.intercept)
        return self.raw(int(body_tokens / self.slope), nonce)


# =============================================================================================
# Profiling orchestration
# =============================================================================================


@dataclass(frozen=True)
class ContextTier:
    name: str
    tokens: int


DEFAULT_TIERS = (ContextTier("Short", 100), ContextTier("Medium", 1000), ContextTier("Extended", 4000))


@dataclass(frozen=True)
class ProfileConfig:
    endpoint: str
    api: str
    model: str | None
    tiers: tuple[ContextTier, ...]
    num_predict: int
    num_ctx: int | None
    repeats: int
    prewarm_s: float
    interval_s: float
    settle_s: float
    timeout_s: float
    seed: int
    cold: bool
    keep_alive: str
    allow_remote: bool
    output: Path
    server_names: tuple[str, ...]
    cpu_only: bool = False


@dataclass
class IterationResult:
    tier: ContextTier
    repeat: int
    prompt_chars: int
    trace: GenerationTrace
    window: WindowStats | None
    memory: MemoryDelta | None
    server_cpu_s: float
    wall_s: float
    cpu_over_baseline: float

    @property
    def ok(self) -> bool:
        return self.trace.error is None

    @property
    def server_cores(self) -> float:
        return self.server_cpu_s / self.wall_s if self.wall_s > 0 else math.nan


@dataclass
class PhaseRecord:
    name: str
    t_start: float
    t_end: float
    window: WindowStats | None
    memory: MemoryDelta | None
    note: str = ""


@dataclass
class ProfileResult:
    config: ProfileConfig
    started_utc: str
    finished_utc: str = ""
    host: dict[str, Any] = field(default_factory=dict)
    server_version: str = ""
    adapter: str = ""
    model: str = ""
    num_ctx: int = 0
    cold_start: bool = False
    baseline: Baseline | None = None
    health: SamplerHealth | None = None
    phases: list[PhaseRecord] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    cold_load_s: float | None = None
    server_resident: dict[str, Any] | None = None
    iterations: list[IterationResult] = field(default_factory=list)
    processes: list[dict[str, Any]] = field(default_factory=list)
    connect_ms: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def cpu_brand() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=2, check=False)
            if out.stdout.strip():
                return out.stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return platform.processor() or platform.machine()


def host_profile() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    freq = None
    with contextlib.suppress(Exception):
        f = psutil.cpu_freq()
        freq = round(f.max or f.current) if f else None
    return {
        "cpu": cpu_brand(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_mhz": freq,
        "ram_total_bytes": vm.total,
        "swap_total_bytes": psutil.swap_memory().total,
        "os": platform.platform(),
        "python": platform.python_version(),
        "psutil": psutil.__version__,
    }


async def wait_for_memory_settle(monitor: CoreMonitor, timeout_s: float = 20.0, window_s: float = 1.0,
                                 tolerance: float = 48 * MIB) -> bool:
    deadline = time.perf_counter() + timeout_s
    await asyncio.sleep(window_s)
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        window = monitor.window(now - window_s, now)
        if window and window.server_rss_peak - window.server_rss_min <= tolerance \
                and window.ram_used_peak - window.ram_used_min <= 2 * tolerance:
            return True
        await asyncio.sleep(0.25)
    return False


def _nonce(tag: str) -> str:
    return f"{tag}-{os.urandom(4).hex()}"


async def run_profile(cfg: ProfileConfig) -> ProfileResult:
    result = ProfileResult(config=cfg, started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    result.host = host_profile()
    endpoint = Endpoint.parse(cfg.endpoint, cfg.allow_remote)
    http = AsyncLocalHTTP(endpoint, cfg.allow_remote)
    adapter = await select_adapter(cfg.api, endpoint, http)
    result.adapter = adapter.name
    result.server_version = await adapter.server_version(http)
    result.model = adapter.resolve_model(cfg.model, await adapter.list_models(http))
    longest = max(t.tokens for t in cfg.tiers)
    needed = longest + cfg.num_predict + 256
    result.num_ctx = cfg.num_ctx or int(math.ceil(needed / 1024.0) * 1024)
    if adapter.supports_num_ctx and result.num_ctx < needed:
        raise ConfigError(f"--num-ctx {result.num_ctx} cannot hold {longest} prompt + {cfg.num_predict} output tokens")
    if not adapter.supports_num_ctx:
        result.warnings.append("context window is fixed by the server (num_ctx not settable via the OpenAI API); "
                               f"it must hold at least {needed} tokens")
    log.info("endpoint %s | %s | model %s | num_ctx %d", endpoint.url, result.server_version, result.model,
             result.num_ctx)

    if cfg.cpu_only and adapter.name != "ollama":
        result.warnings.append("--cpu-only is only enforced through the Ollama API; start the server without GPU "
                               "layers (e.g. llama-server -ngl 0) for a CPU-only profile")
    client = StreamingInferenceClient(http, adapter, result.model, num_ctx=result.num_ctx, seed=cfg.seed,
                                      keep_alive=cfg.keep_alive, timeout_s=cfg.timeout_s, cpu_only=cfg.cpu_only)
    factory = PromptFactory(cfg.seed)
    monitor = CoreMonitor(cfg.interval_s, server_names=cfg.server_names,
                          server_port=endpoint.port if endpoint.loopback else None)
    previous_switch = sys.getswitchinterval()
    sys.setswitchinterval(0.001)
    try:
        async with monitor:
            # Phase 0 - cold state
            if cfg.cold and adapter.supports_unload:
                log.info("unloading %s for a cold baseline", result.model)
                await adapter.unload(http, result.model)
                if not await wait_for_memory_settle(monitor):
                    result.warnings.append("memory did not fully settle after unloading the model")
                monitor.rescan_server()
                result.cold_start = True
            elif cfg.cold:
                result.warnings.append(f"{adapter.name} API cannot unload models; baseline includes resident weights")

            # Phase 1 - pre-warm noise baseline
            log.info("pre-warm: sampling background noise for %.1f s", cfg.prewarm_s)
            t0 = time.perf_counter()
            baseline = await monitor.aprewarm(cfg.prewarm_s)
            result.baseline = baseline
            result.phases.append(PhaseRecord("pre-warm baseline", t0, time.perf_counter(), baseline.stats, None,
                                             "model unloaded" if result.cold_start else "model state unknown"))
            log.info("baseline: cpu %.1f%% | ram used %.2f GiB | server rss %.0f MiB | noise floor %.0f MiB",
                     baseline.stats.cpu_mean, baseline.stats.ram_used_mean / GIB, baseline.stats.server_rss_mean / MIB,
                     baseline.ram_noise_floor / MIB)

            # Phase 2 - model load + tokenizer calibration
            log.info("load + calibration probes")
            t0 = time.perf_counter()
            points = []
            for body_chars in (240, 2400):
                nonce = _nonce("cal")
                prompt = factory.raw(body_chars, nonce)
                probe = await client.generate(prompt, 1)
                if probe.error:
                    raise EndpointError(f"calibration probe failed: {probe.error}")
                if result.cold_load_s is None:
                    result.cold_load_s = probe.server.load_s
                if probe.server.prompt_tokens:
                    body_len = len(prompt) - len(factory.HEADER.format(nonce=nonce)) - len(factory.INSTRUCTION)
                    points.append((body_len, int(probe.server.prompt_tokens)))
            factory.calibrate(points)
            result.calibration = {"calibrated": factory.calibrated, "tokens_per_char": factory.slope,
                                  "fixed_overhead_tokens": factory.intercept, "points": points}
            if not factory.calibrated:
                result.warnings.append("server did not report prompt token counts; prompt sizes use a 4 chars/token "
                                       "estimate")
            monitor.rescan_server()
            t1 = time.perf_counter()
            load_window = monitor.window(t0, t1)
            result.phases.append(PhaseRecord("model load + calibration", t0, t1, load_window,
                                             isolate_memory(load_window, baseline) if load_window else None,
                                             f"cold load {result.cold_load_s:.2f} s" if result.cold_load_s else ""))

            # Phase 3 - decode warm-up (first decode after load pays one-off kernel / graph initialisation)
            t0 = time.perf_counter()
            warm = await client.generate(factory.build(64, _nonce("warmup")), min(32, cfg.num_predict))
            t1 = time.perf_counter()
            warm_window = monitor.window(t0, t1 + cfg.interval_s)
            result.phases.append(PhaseRecord("decode warm-up (excluded)", t0, t1, warm_window,
                                             isolate_memory(warm_window, baseline) if warm_window else None,
                                             f"{_f(warm.server_tps, 1)} tok/s" if not warm.error else warm.error))
            with contextlib.suppress(ProfilerError):
                result.server_resident = await adapter.resident_report(http, result.model)

            # Phase 4 - loaded idle
            await asyncio.sleep(cfg.settle_s)
            t0 = time.perf_counter()
            await asyncio.sleep(1.0)
            idle = monitor.window(t0, time.perf_counter())
            result.phases.append(PhaseRecord("loaded idle (weights resident)", t0, time.perf_counter(), idle,
                                             isolate_memory(idle, baseline) if idle else None))

            # Phase 5 - context stress matrix
            for tier in cfg.tiers:
                for repeat in range(1, cfg.repeats + 1):
                    await asyncio.sleep(cfg.settle_s)
                    prompt = factory.build(tier.tokens, _nonce(f"{tier.name.lower()}{repeat}"))
                    log.info("run %s #%d: ~%d prompt tokens (%d chars), %d output tokens", tier.name, repeat,
                             tier.tokens, len(prompt), cfg.num_predict)
                    cpu0 = monitor.server_cpu_seconds()
                    t0 = time.perf_counter()
                    trace = await client.generate(prompt, cfg.num_predict)
                    t1 = time.perf_counter()
                    cpu1 = monitor.server_cpu_seconds()
                    await asyncio.sleep(2 * cfg.interval_s)
                    window = monitor.window(t0, t1 + cfg.interval_s)
                    memory = isolate_memory(window, baseline) if window else None
                    iteration = IterationResult(tier, repeat, len(prompt), trace, window, memory,
                                                max(0.0, cpu1 - cpu0), t1 - t0,
                                                (window.cpu_mean - baseline.stats.cpu_mean) if window else math.nan)
                    result.iterations.append(iteration)
                    result.phases.append(PhaseRecord(f"{tier.name} #{repeat}", t0, t1, window, memory))
                    if trace.error:
                        log.error("run %s #%d failed: %s", tier.name, repeat, trace.error)
                    else:
                        log.info("  ttft %.0f ms | prompt %s tok @ %.1f tok/s | output %s tok @ %.2f tok/s | "
                                 "model RAM delta %.2f GiB", trace.ttft_s * 1000, trace.server.prompt_tokens,
                                 trace.prompt_tps, trace.server.output_tokens, trace.server_tps,
                                 (memory.model_resident_delta / GIB) if memory else math.nan)
            result.processes = monitor.server_processes()
        result.health = monitor.health()
        result.connect_ms = list(http.connect_ms)
        csv_path = cfg.output.parent / "telemetry_samples.csv"
        monitor.export_csv(csv_path, [(p.name, p.t_start, p.t_end) for p in result.phases])
    finally:
        sys.setswitchinterval(previous_switch)
    result.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


# =============================================================================================
# Reporting
# =============================================================================================


def _ok(value: float | None) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _f(value: float | None, digits: int = 1, unit: str = "") -> str:
    return f"{value:,.{digits}f}{unit}" if _ok(value) else "n/a"


def _gib(value: float | None, signed: bool = False) -> str:
    if not _ok(value):
        return "n/a"
    return f"{value / GIB:+.2f} GiB" if signed else f"{value / GIB:.2f} GiB"


def _mib(value: float | None, signed: bool = False) -> str:
    if not _ok(value):
        return "n/a"
    return f"{value / MIB:+,.0f} MiB" if signed else f"{value / MIB:,.0f} MiB"


def _agg(values: list[float], digits: int = 1, scale: float = 1.0, unit: str = "") -> str:
    data = [v * scale for v in values if _ok(v)]
    if not data:
        return "n/a"
    median = statistics.median(data)
    if len(data) == 1:
        return f"{median:,.{digits}f}{unit}"
    return f"{median:,.{digits}f} ± {statistics.pstdev(data):,.{digits}f}{unit}"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def build_findings(result: ProfileResult) -> list[str]:
    findings: list[str] = []
    by_tier: dict[str, list[IterationResult]] = {}
    for it in result.iterations:
        if it.ok:
            by_tier.setdefault(it.tier.name, []).append(it)
    tiers = [t for t in result.config.tiers if t.name in by_tier]

    def med(tier: ContextTier, fn: Any) -> float:
        values = [fn(it) for it in by_tier[tier.name] if _ok(fn(it))]
        return statistics.median(values) if values else math.nan

    if len(tiers) >= 2:
        first, last = tiers[0], tiers[-1]
        ttft_a, ttft_b = med(first, lambda i: i.trace.ttft_s), med(last, lambda i: i.trace.ttft_s)
        if _ok(ttft_a) and _ok(ttft_b) and ttft_a > 0:
            findings.append(f"TTFT grows {ttft_b / ttft_a:.1f}x from {first.name} ({ttft_a * 1000:,.0f} ms) to "
                            f"{last.name} ({ttft_b * 1000:,.0f} ms): prompt evaluation dominates first-token latency.")
        tps_a, tps_b = med(first, lambda i: i.trace.server_tps), med(last, lambda i: i.trace.server_tps)
        if _ok(tps_a) and _ok(tps_b) and tps_a > 0:
            change = 100.0 * (tps_b - tps_a) / tps_a
            if change <= -5.0:
                reason = ": decode slows as attention spans a longer KV cache."
            elif change >= 5.0:
                reason = ": faster at long context, which usually means run-to-run variance; raise --repeats."
            else:
                reason = ": decode throughput is insensitive to prompt length at this scale."
            findings.append(f"Output evaluation rate {first.name} {tps_a:.2f} tok/s -> {last.name} {tps_b:.2f} tok/s "
                            f"({change:+.1f}%){reason}")
        mem_a = med(first, lambda i: i.memory.model_resident_delta if i.memory else math.nan)
        mem_b = med(last, lambda i: i.memory.model_resident_delta if i.memory else math.nan)
        tok_a = med(first, lambda i: i.trace.server.prompt_tokens or math.nan)
        tok_b = med(last, lambda i: i.trace.server.prompt_tokens or math.nan)
        floor = result.baseline.server_rss_noise_floor if result.baseline else 16 * MIB
        if all(_ok(v) for v in (mem_a, mem_b, tok_a, tok_b)) and tok_b > tok_a:
            if abs(mem_b - mem_a) <= 2 * floor:
                findings.append(f"No host-RAM growth above noise between {first.name} and {last.name} "
                                f"({_gib(mem_a)} vs {_gib(mem_b)}): the KV cache is pre-allocated for num_ctx "
                                f"{result.num_ctx:,} at load time" + (" or held in VRAM." if (result.server_resident or {})
                                                                      .get("vram_bytes") else "."))
            else:
                slope = (mem_b - mem_a) / (tok_b - tok_a) * 1000.0
                findings.append(f"Active host memory grows {slope / MIB:,.1f} MiB per 1,000 additional prompt tokens "
                                f"(KV cache and compute buffers): {_gib(mem_a)} -> {_gib(mem_b)} above baseline.")
    resident = [p for p in result.phases if p.name.startswith("loaded idle") and p.memory]
    if resident:
        m = resident[0].memory
        findings.append(f"Idle resident footprint of the loaded model: {_gib(m.model_resident_delta)} "
                        f"({m.source}; noise floor {_mib(m.noise_floor)}).")
    resident_info = result.server_resident or {}
    if resident_info.get("vram_bytes"):
        findings.append(f"GPU offload detected: {_gib(resident_info['vram_bytes'])} of the model's "
                        f"{_gib(resident_info['size_bytes'])} allocation is in VRAM "
                        f"({100 * resident_info['gpu_fraction']:.0f}%), so host-RAM deltas cover only the CPU-resident "
                        "share. Re-run with --cpu-only for a pure CPU/RAM profile.")
    elif resident_info.get("size_bytes"):
        findings.append(f"Model fully host-resident: the server reports a {_gib(resident_info['size_bytes'])} "
                        "allocation with no VRAM offload.")
    if result.cold_load_s:
        findings.append(f"Cold model load took {result.cold_load_s:.2f} s (server-reported load_duration).")
    base = result.baseline
    if base:
        total = float(result.host.get("ram_total_bytes") or 0)
        pressured = [it for it in result.iterations if it.window and (
            it.window.swap_used_peak - base.stats.swap_used_end > 64 * MIB
            or (total and it.window.ram_available_min < 0.05 * total))]
        faulting = [it for it in result.iterations if it.window and _ok(it.window.paging_rate_mean)
                    and it.window.paging_rate_mean > max(50.0, 5 * (base.stats.paging_rate_mean or 0.0))]
        if pressured:
            names = ", ".join(f"{it.tier.name} #{it.repeat}" for it in pressured)
            findings.append(f"WARNING: memory pressure during {names} (swap growth or < 5% RAM available); "
                            "latency figures are inflated by paging.")
        elif faulting:
            names = ", ".join(f"{it.tier.name} #{it.repeat}" for it in faulting)
            findings.append(f"Hard-fault paging during {names} without swap growth: memory-mapped weights were being "
                            "faulted in from disk, not evicted. Steady-state runs should not show this.")
        else:
            findings.append("No memory pressure or abnormal paging during inference runs; the model fits in RAM.")
        insignificant = [it for it in result.iterations if it.memory and not it.memory.significant]
        if insignificant:
            findings.append(f"{len(insignificant)} run(s) show a memory delta inside the baseline noise floor "
                            f"({_mib(base.ram_noise_floor)}); treat those deltas as unresolved.")
    for it in result.iterations:
        actual = it.trace.server.prompt_tokens
        if it.ok and actual and abs(actual - it.tier.tokens) / it.tier.tokens > 0.15:
            findings.append(f"{it.tier.name} #{it.repeat} evaluated {actual} prompt tokens vs target {it.tier.tokens} "
                            "(server tokenizer or prompt caching differs from calibration).")
    if result.health and result.health.overruns:
        findings.append(f"Sampler missed {result.health.overruns} deadlines; host was saturated during profiling.")
    failed = [it for it in result.iterations if not it.ok]
    if failed:
        findings.append(f"{len(failed)} run(s) failed: " + "; ".join(f"{it.tier.name} #{it.repeat}: {it.trace.error}"
                                                                      for it in failed))
    return findings


def render_scorecard(result: ProfileResult) -> str:
    cfg = result.config
    host = result.host
    base = result.baseline
    health = result.health
    out: list[str] = []
    w = out.append

    w("# SovereignLLMProfiler - System Scorecard")
    w("")
    w(f"Generated {result.finished_utc} (started {result.started_utc}) by SovereignLLMProfiler v{__version__}.")
    w("")
    w("## Run configuration")
    w("")
    w("| Parameter | Value |")
    w("|---|---|")
    w(f"| Endpoint | `{cfg.endpoint}` ({result.adapter} API, {_md_escape(result.server_version)}) |")
    w(f"| Model | `{result.model}` |")
    w(f"| Context tiers | {', '.join(f'{t.name} {t.tokens:,}' for t in cfg.tiers)} prompt tokens |")
    w(f"| Output tokens per run | {cfg.num_predict} (temperature 0, top_k 1, seed {cfg.seed}) |")
    w(f"| Context window (num_ctx) | {result.num_ctx:,} |")
    w(f"| Repeats per tier | {cfg.repeats} |")
    w(f"| Start state | {'cold (model unloaded before baseline)' if result.cold_start else 'warm / server-managed'} |")
    w(f"| Device policy | {'CPU only (num_gpu=0)' if cfg.cpu_only else 'server default (GPU offload allowed)'} |")
    if result.server_resident:
        r = result.server_resident
        w(f"| Server-reported model allocation | {_gib(r['size_bytes'])} total: {_gib(r['host_bytes'])} host RAM, "
          f"{_gib(r['vram_bytes'])} VRAM ({100 * r['gpu_fraction']:.0f}% GPU) |")
    w(f"| Telemetry interval | {cfg.interval_s * 1000:.0f} ms target" + (
        f"; achieved {health.mean_interval_ms:.2f} ms mean, {health.p95_interval_ms:.2f} ms p95, "
        f"{health.max_interval_ms:.1f} ms max, {health.overruns} overruns |" if health else " |"))
    w(f"| Pre-warm window | {cfg.prewarm_s:.1f} s |")
    if health:
        w(f"| Paging source | {_md_escape(health.paging_source)} |")
        w(f"| Server discovery | {health.server_discovery} |")
    if result.processes:
        procs = "; ".join(f"{_md_escape(str(p['name']))} pid {p['pid']}" for p in result.processes)
        w(f"| Server process tree | {procs} |")
    cal = result.calibration
    if cal:
        w(f"| Tokenizer calibration | {'server-calibrated' if cal.get('calibrated') else 'estimated'}: "
          f"{cal.get('tokens_per_char', 0):.3f} tokens/char + {cal.get('fixed_overhead_tokens', 0):.0f} template tokens |")
    if result.connect_ms:
        w(f"| Loopback connect time | {_f(statistics.median(result.connect_ms), 2)} ms median |")
    w("")
    w("## Host")
    w("")
    w("| Component | Value |")
    w("|---|---|")
    w(f"| CPU | {_md_escape(str(host.get('cpu')))} ({host.get('physical_cores')} physical / "
      f"{host.get('logical_cores')} logical cores{', ' + str(host['cpu_max_mhz']) + ' MHz' if host.get('cpu_max_mhz') else ''}) |")
    w(f"| Physical RAM | {_gib(host.get('ram_total_bytes'))} |")
    w(f"| Swap / page file | {_gib(host.get('swap_total_bytes'))} |")
    w(f"| OS | {_md_escape(str(host.get('os')))} |")
    w(f"| Runtime | Python {host.get('python')}, psutil {host.get('psutil')} |")
    w("")

    if base:
        s = base.stats
        w("## 1. Baseline system overhead (background noise)")
        w("")
        w(f"Pre-warm window of {s.duration_s:.2f} s, {s.samples} samples"
          + (", inference model unloaded." if result.cold_start else "."))
        w("")
        w("| Metric | Mean | p95 | Peak | Std dev |")
        w("|---|---|---|---|---|")
        w(f"| CPU utilisation (all cores) | {_f(s.cpu_mean, 1, '%')} | {_f(s.cpu_p95, 1, '%')} | {_f(s.cpu_max, 1, '%')} | "
          f"{_f(s.cpu_std, 2, ' pp')} |")
        w(f"| RAM in use | {_gib(s.ram_used_mean)} | - | {_gib(s.ram_used_peak)} | {_mib(s.ram_used_std)} |")
        w(f"| RAM available | {_gib(s.ram_available_mean)} | - | min {_gib(s.ram_available_min)} | - |")
        w(f"| Paging (pages/s) | {_f(s.paging_rate_mean, 1)} | {_f(s.paging_rate_p95, 1)} | {_f(s.paging_rate_max, 1)} | - |")
        w(f"| Swap / page file in use | {_gib(s.swap_used_end)} | - | {_gib(s.swap_used_peak)} | - |")
        w(f"| Inference server RSS (idle) | {_mib(s.server_rss_mean)} | - | {_mib(s.server_rss_peak)} | - |")
        w(f"| Profiler RSS | {_mib(s.self_rss_mean)} | - | {_mib(s.self_rss_peak)} | - |")
        w("")
        w("| Noise floor | Value |")
        w("|---|---|")
        w(f"| System RAM noise floor (max of 3σ, peak-to-peak, 16 MiB) | {_mib(base.ram_noise_floor)} |")
        w(f"| Server RSS noise floor | {_mib(base.server_rss_noise_floor)} |")
        w(f"| CPU noise band (p95 - mean) | {_f(base.cpu_noise_floor, 2, ' pp')} |")
        if health:
            w(f"| Profiler sampler overhead | {health.overhead_pct_of_core:.2f}% of one core "
              f"({health.sampler_cpu_s * 1000:,.0f} ms busy over {health.wall_s:,.1f} s) |")
        w("")

    w("## 2. Model memory isolation")
    w("")
    w("Model resident delta = peak inference-server process-tree RSS minus its pre-warm mean. The isolated system delta "
      "(peak RAM in use minus baseline minus profiler growth) is reported as an independent cross-check.")
    w("")
    w("| Phase | Duration | Server RSS peak | Model resident delta | Isolated system delta | Profiler growth | "
      "Above noise | Notes |")
    w("|---|---|---|---|---|---|---|---|")
    for phase in result.phases:
        if phase.window is None:
            continue
        m = phase.memory
        w(f"| {_md_escape(phase.name)} | {phase.window.duration_s:,.2f} s | {_gib(phase.window.server_rss_peak)} | "
          f"{_gib(m.model_resident_delta, True) if m else '-'} | {_gib(m.isolated_system_delta, True) if m else '-'} | "
          f"{_mib(m.profiler_delta, True) if m else '-'} | {('yes' if m.significant else 'no') if m else '-'} | "
          f"{_md_escape(phase.note)} |")
    w("")

    w("## 3. Context stress matrix")
    w("")
    w("Medians across repeats (± population std dev when repeats > 1). TTFT is measured client-side from request write "
      "to the first content-bearing chunk; evaluation rates are server-reported, the client rate is derived from chunk "
      "arrival timestamps.")
    w("")
    w("| Tier | Target ctx | Prompt tokens | TTFT (ms) | Prompt eval (tok/s) | Output tokens | Output eval rate "
      "(tok/s) | Client stream rate (tok/s) | ITL p50 / p95 (ms) | Model RAM delta (peak) | Isolated system delta | "
      "CPU mean / peak | CPU over baseline | Server cores busy | Paging mean / max (pg/s) | Runs ok |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tier in cfg.tiers:
        runs = [it for it in result.iterations if it.tier == tier]
        good = [it for it in runs if it.ok]
        if not good:
            w(f"| {tier.name} | {tier.tokens:,} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
              f"n/a | n/a | 0/{len(runs)} |")
            continue
        itl50 = [percentile(it.trace.itl_ms, 50) for it in good]
        itl95 = [percentile(it.trace.itl_ms, 95) for it in good]
        win = [it.window for it in good if it.window]
        w(f"| {tier.name} | {tier.tokens:,} | {_agg([float(it.trace.server.prompt_tokens or math.nan) for it in good], 0)} | "
          f"{_agg([it.trace.ttft_s for it in good], 0, 1000.0)} | {_agg([it.trace.prompt_tps for it in good], 1)} | "
          f"{_agg([float(it.trace.server.output_tokens or it.trace.stream_chunks) for it in good], 0)} | "
          f"{_agg([it.trace.server_tps for it in good], 2)} | {_agg([it.trace.client_tps for it in good], 2)} | "
          f"{_agg(itl50, 1)} / {_agg(itl95, 1)} | "
          f"{_agg([it.memory.model_resident_delta for it in good if it.memory], 2, 1 / GIB, ' GiB')} | "
          f"{_agg([it.memory.isolated_system_delta for it in good if it.memory], 2, 1 / GIB, ' GiB')} | "
          f"{_agg([w_.cpu_mean for w_ in win], 1, unit='%')} / {_agg([w_.cpu_max for w_ in win], 1, unit='%')} | "
          f"{_agg([it.cpu_over_baseline for it in good], 1, unit=' pp')} | {_agg([it.server_cores for it in good], 2)} | "
          f"{_agg([w_.paging_rate_mean for w_ in win], 1)} / {_agg([w_.paging_rate_max for w_ in win], 1)} | "
          f"{len(good)}/{len(runs)} |")
    w("")

    w("## 4. Per-run telemetry vectors")
    w("")
    w("| Run | Wall (s) | Prompt chars | Prompt tokens | Prompt eval (s) | TTFT (ms) | Output tokens | Eval (s) | "
      "Output rate (tok/s) | Client rate (tok/s) | ITL p50 / p95 / max (ms) | Server RSS peak | Model RAM delta | "
      "RAM available min | CPU mean | Samples | Status |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for it in result.iterations:
        t = it.trace
        win = it.window
        itl = t.itl_ms
        status = "ok" if it.ok else f"FAILED: {_md_escape(t.error or '')}"
        w(f"| {it.tier.name} #{it.repeat} | {it.wall_s:,.2f} | {it.prompt_chars:,} | {t.server.prompt_tokens or 'n/a'} | "
          f"{_f(t.server.prompt_eval_s, 2)} | {_f(t.ttft_s * 1000 if _ok(t.ttft_s) else None, 0)} | "
          f"{t.server.output_tokens or t.stream_chunks} | {_f(t.server.eval_s, 2)} | {_f(t.server_tps, 2)} | "
          f"{_f(t.client_tps, 2)} | {_f(percentile(itl, 50), 1)} / {_f(percentile(itl, 95), 1)} / "
          f"{_f(max(itl) if itl else None, 1)} | {_gib(win.server_rss_peak) if win else 'n/a'} | "
          f"{_gib(it.memory.model_resident_delta, True) if it.memory else 'n/a'} | "
          f"{_gib(win.ram_available_min) if win else 'n/a'} | {_f(win.cpu_mean, 1, '%') if win else 'n/a'} | "
          f"{win.samples if win else 0} | {status} |")
    w("")

    w("## 5. Findings")
    w("")
    for finding in build_findings(result) + [f"NOTE: {warning}" for warning in result.warnings]:
        w(f"- {finding}")
    w("")
    w("## Methodology")
    w("")
    w(f"- Telemetry: dedicated sampler thread at {cfg.interval_s * 1000:.0f} ms (psutil cpu_times deltas, "
      "virtual_memory, platform paging counter, server process-tree RSS and CPU time); the asyncio client runs on the "
      "main thread with a 1 ms GIL switch interval so chunk timestamps are not delayed by the sampler.")
    w("- Baseline: pre-warm window after unloading the model; the first 250 ms are discarded to let counters settle. "
      "Noise floors gate whether a delta is reported as significant.")
    w("- Isolation: the model delta comes from the inference server's own process tree (listening port, then name); "
      "the profiler subtracts its own RSS growth from the system-level cross-check.")
    w("- Prompts: deterministic synthetic systems logs sized with a two-point calibration against the server tokenizer; "
      "each prompt starts with a random nonce so no KV-cache prefix can be reused between runs.")
    w("- Decoding: greedy (temperature 0, top_k 1, fixed seed) so runs are comparable; output rate uses server-side "
      "eval_count / eval_duration, cross-checked by client-side chunk timing.")
    w("")
    return "\n".join(out)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else round(obj, 6)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def profile_json(result: ProfileResult) -> dict[str, Any]:
    iterations = []
    for it in result.iterations:
        t = it.trace
        iterations.append({
            "tier": it.tier.name, "target_tokens": it.tier.tokens, "repeat": it.repeat, "ok": it.ok, "error": t.error,
            "prompt_chars": it.prompt_chars, "wall_s": it.wall_s, "server_cpu_s": it.server_cpu_s,
            "server_cores": it.server_cores, "cpu_over_baseline_pp": it.cpu_over_baseline,
            "ttft_ms": t.ttft_s * 1000 if _ok(t.ttft_s) else None, "server_tps": t.server_tps,
            "client_tps": t.client_tps, "prompt_tps": t.prompt_tps, "server": asdict(t.server),
            "stream_chunks": t.stream_chunks, "output_chars": t.output_chars,
            "token_offsets_ms": [(x - t.t_send) * 1000 for x in t.token_times] if _ok(t.t_send) else [],
            "itl_ms": {"p50": percentile(t.itl_ms, 50), "p95": percentile(t.itl_ms, 95), "p99": percentile(t.itl_ms, 99),
                       "max": max(t.itl_ms) if t.itl_ms else None},
            "window": asdict(it.window) if it.window else None,
            "memory": asdict(it.memory) if it.memory else None,
        })
    return _jsonable({
        "tool": f"SovereignLLMProfiler {__version__}",
        "started_utc": result.started_utc, "finished_utc": result.finished_utc,
        "config": {**asdict(result.config), "tiers": [asdict(t) for t in result.config.tiers]},
        "host": result.host, "server_version": result.server_version, "adapter": result.adapter,
        "model": result.model, "num_ctx": result.num_ctx, "cold_start": result.cold_start,
        "cold_load_s": result.cold_load_s, "calibration": result.calibration,
        "server_resident": result.server_resident,
        "baseline": asdict(result.baseline) if result.baseline else None,
        "sampler_health": asdict(result.health) if result.health else None,
        "phases": [{"name": p.name, "note": p.note, "window": asdict(p.window) if p.window else None,
                    "memory": asdict(p.memory) if p.memory else None} for p in result.phases],
        "server_processes": result.processes, "connect_ms": result.connect_ms,
        "iterations": iterations, "findings": build_findings(result), "warnings": result.warnings,
    })


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# =============================================================================================
# CLI
# =============================================================================================


def parse_tiers(spec: str) -> tuple[ContextTier, ...]:
    default_names = ("Short", "Medium", "Extended")
    tiers = []
    for index, item in enumerate(part.strip() for part in spec.split(",") if part.strip()):
        name, _, value = item.rpartition("=")
        try:
            tokens = int(value)
        except ValueError as exc:
            raise ConfigError(f"invalid context tier {item!r}") from exc
        if tokens < 16:
            raise ConfigError("context tiers must be at least 16 tokens")
        tiers.append(ContextTier(name or (default_names[index] if index < 3 else f"Tier {index + 1}"), tokens))
    if len(tiers) < 3:
        raise ConfigError("the context stress matrix needs at least 3 tiers")
    if len({t.name for t in tiers}) != len(tiers):
        raise ConfigError("context tier names must be unique")
    return tuple(tiers)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="analytics", description="SovereignLLMProfiler streaming inference analytics")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434", help="local streaming endpoint (base URL or "
                   "full /api/generate or /v1/completions URL)")
    p.add_argument("--api", choices=["auto", "ollama", "openai"], default="auto")
    p.add_argument("--model", help="model to profile (default: first model the server lists)")
    p.add_argument("--contexts", default="Short=100,Medium=1000,Extended=4000",
                   help="comma-separated prompt token tiers, optionally Name=tokens (minimum 3)")
    p.add_argument("--num-predict", type=int, default=128, help="output tokens generated per run")
    p.add_argument("--num-ctx", type=int, help="context window (default: largest tier + output + 256, rounded to 1024)")
    p.add_argument("--repeats", type=int, default=1, help="runs per tier (medians reported)")
    p.add_argument("--prewarm", type=float, default=3.0, help="noise-baseline pre-warm window in seconds")
    p.add_argument("--interval", type=float, default=0.05, help="telemetry sampling interval in seconds")
    p.add_argument("--settle", type=float, default=1.5, help="quiet period before each run in seconds")
    p.add_argument("--timeout", type=float, default=900.0, help="max seconds without stream progress")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warm-start", action="store_true", help="do not unload the model before the baseline")
    p.add_argument("--cpu-only", action="store_true", help="force CPU inference (Ollama num_gpu=0) so the whole "
                   "model footprint lands in host RAM")
    p.add_argument("--keep-alive", default="10m", help="Ollama keep_alive during the matrix")
    p.add_argument("--server-process", default="ollama,ollama_llama_server,llama-server,llama_server",
                   help="process names used when the listening port cannot be resolved")
    p.add_argument("--allow-remote", action="store_true", help="permit non-loopback endpoints")
    p.add_argument("--output", type=Path, default=DEFAULT_SCORECARD, help="scorecard markdown path")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def config_from_args(args: argparse.Namespace) -> ProfileConfig:
    if args.num_predict < 1 or args.repeats < 1:
        raise ConfigError("--num-predict and --repeats must be >= 1")
    if args.prewarm < 1.0:
        raise ConfigError("--prewarm must be at least 1 s to characterise background noise")
    if not 0.01 <= args.interval <= 1.0:
        raise ConfigError("--interval must be between 0.01 and 1.0 s")
    return ProfileConfig(
        endpoint=args.endpoint, api=args.api, model=args.model, tiers=parse_tiers(args.contexts),
        num_predict=args.num_predict, num_ctx=args.num_ctx, repeats=args.repeats, prewarm_s=args.prewarm,
        interval_s=args.interval, settle_s=args.settle, timeout_s=args.timeout, seed=args.seed,
        cold=not args.warm_start, keep_alive=args.keep_alive, allow_remote=args.allow_remote,
        output=args.output.resolve(),
        server_names=tuple(n.strip() for n in args.server_process.split(",") if n.strip()),
        cpu_only=args.cpu_only,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        cfg = config_from_args(args)
        result = asyncio.run(run_profile(cfg))
    except ProfilerError as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return exc.exit_code
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    write_atomic(cfg.output, render_scorecard(result))
    json_path = cfg.output.parent / "system_profile.json"
    write_atomic(json_path, json.dumps(profile_json(result), indent=2))
    log.info("scorecard  -> %s", cfg.output)
    log.info("profile    -> %s", json_path)
    log.info("telemetry  -> %s", cfg.output.parent / "telemetry_samples.csv")
    return 2 if any(not it.ok for it in result.iterations) else 0


if __name__ == "__main__":
    sys.exit(main())
