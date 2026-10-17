"""In-process metrics: labelled counters and histograms with p50/p95/p99. Stdlib only.

Histograms keep the most recent `max_samples` observations, so percentiles describe recent
behaviour and memory stays bounded. `count` and `mean` cover every observation.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

LabelKey = tuple[tuple[str, str], ...]


def _key(labels: Mapping[str, object]) -> LabelKey:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of samples at or below it."""
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


@dataclass(frozen=True, slots=True)
class HistogramSummary:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


class _Histogram:
    def __init__(self, max_samples: int) -> None:
        self.window: deque[float] = deque(maxlen=max_samples)
        self.count = 0
        self.total = 0.0

    def add(self, value: float) -> None:
        self.window.append(value)
        self.count += 1
        self.total += value

    def summary(self) -> HistogramSummary:
        values = sorted(self.window)
        return HistogramSummary(
            count=self.count,
            mean=self.total / self.count,
            p50=_percentile(values, 50),
            p95=_percentile(values, 95),
            p99=_percentile(values, 99),
            max=values[-1],
        )


class Metrics:
    def __init__(self, max_samples: int = 10_000) -> None:
        self._max_samples = max_samples
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, LabelKey], float] = {}
        self._histograms: dict[tuple[str, LabelKey], _Histogram] = {}

    def inc(self, name: str, value: float = 1.0, /, **labels: object) -> None:
        key = (name, _key(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, /, **labels: object) -> None:
        key = (name, _key(labels))
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = self._histograms[key] = _Histogram(self._max_samples)
            histogram.add(value)

    def counter(self, name: str, /, **labels: object) -> float:
        with self._lock:
            return self._counters.get((name, _key(labels)), 0.0)

    def summary(self, name: str, /, **labels: object) -> HistogramSummary | None:
        with self._lock:
            histogram = self._histograms.get((name, _key(labels)))
            return histogram.summary() if histogram else None

    @contextmanager
    def timer(self, name: str, /, **labels: object) -> Iterator[None]:
        """Observe the elapsed milliseconds of the `with` block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000, **labels)

    def snapshot(self) -> dict[str, list[dict[str, object]]]:
        with self._lock:
            counters = [
                {"name": n, "labels": dict(k), "value": v}
                for (n, k), v in sorted(self._counters.items())
            ]
            histograms = [
                {"name": n, "labels": dict(k), **vars_of(h.summary())}
                for (n, k), h in sorted(self._histograms.items())
            ]
        return {"counters": counters, "histograms": histograms}


def vars_of(summary: HistogramSummary) -> dict[str, object]:
    return {
        "count": summary.count,
        "mean": summary.mean,
        "p50": summary.p50,
        "p95": summary.p95,
        "p99": summary.p99,
        "max": summary.max,
    }


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics


def set_metrics(metrics: Metrics) -> None:
    global _metrics
    _metrics = metrics
