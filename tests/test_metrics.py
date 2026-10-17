import pytest

from resolve_agent.obs import Metrics


def test_counters_are_labelled() -> None:
    m = Metrics()
    m.inc("tool.calls", tool="a", outcome="ok")
    m.inc("tool.calls", tool="a", outcome="ok")
    m.inc("tool.calls", 3, tool="a", outcome="not_eligible")
    assert m.counter("tool.calls", tool="a", outcome="ok") == 2
    assert (
        m.counter("tool.calls", outcome="not_eligible", tool="a") == 3
    )  # label order is irrelevant
    assert m.counter("tool.calls", tool="b", outcome="ok") == 0


def test_percentiles_use_nearest_rank() -> None:
    m = Metrics()
    for value in range(1, 101):
        m.observe("latency", float(value))
    summary = m.summary("latency")
    assert summary is not None
    assert (summary.count, summary.p50, summary.p95, summary.p99, summary.max) == (
        100,
        50,
        95,
        99,
        100,
    )
    assert summary.mean == pytest.approx(50.5)


def test_single_observation() -> None:
    m = Metrics()
    m.observe("latency", 7.0)
    summary = m.summary("latency")
    assert summary is not None and (summary.p50, summary.p95, summary.p99) == (7.0, 7.0, 7.0)


def test_unknown_histogram_has_no_summary() -> None:
    assert Metrics().summary("nothing") is None


def test_window_is_bounded_but_count_is_not() -> None:
    m = Metrics(max_samples=10)
    for value in range(100):
        m.observe("latency", float(value))
    summary = m.summary("latency")
    assert summary is not None
    assert summary.count == 100
    assert summary.p50 == 94  # median of the last ten values (90..99)


def test_a_label_may_be_called_name() -> None:
    m = Metrics()
    m.observe("span.duration_ms", 1.0, kind="tool", name="tool.x")
    assert m.summary("span.duration_ms", kind="tool", name="tool.x") is not None


def test_timer_observes_elapsed_milliseconds() -> None:
    m = Metrics()
    with m.timer("block"):
        pass
    summary = m.summary("block")
    assert summary is not None and summary.count == 1 and summary.max >= 0


def test_snapshot_lists_everything() -> None:
    m = Metrics()
    m.inc("c", route="x")
    m.observe("h", 2.0)
    snapshot = m.snapshot()
    assert snapshot["counters"] == [{"name": "c", "labels": {"route": "x"}, "value": 1.0}]
    assert snapshot["histograms"][0]["name"] == "h"
    assert snapshot["histograms"][0]["p95"] == 2.0
