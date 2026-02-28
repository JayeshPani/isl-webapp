from __future__ import annotations

from dataclasses import replace

from backend.config import Settings
from backend.inference import TemporalStabilizer


def _settings_for_test() -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        smoothing_window=5,
        min_votes=3,
        min_confidence=0.2,
        min_margin=0.05,
        commit_cooldown_ms=0,
        duplicate_block_until_changed=True,
        duplicate_reset_unknown_frames=3,
        unknown_label="unknown",
    )


def test_temporal_stabilizer_commits_after_min_votes() -> None:
    stabilizer = TemporalStabilizer(_settings_for_test())

    result = stabilizer.push("A", 0.95, [{"label": "A", "conf": 0.95}], ts_ms=1)
    assert result["committed_text"] == ""

    result = stabilizer.push("A", 0.94, [{"label": "A", "conf": 0.94}], ts_ms=2)
    assert result["committed_text"] == ""

    result = stabilizer.push("A", 0.93, [{"label": "A", "conf": 0.93}], ts_ms=3)
    assert result["committed_text"] == "A"
    assert result["token_committed"] is True


def test_temporal_stabilizer_suppresses_duplicates_until_change() -> None:
    stabilizer = TemporalStabilizer(_settings_for_test())

    for ts in range(1, 7):
        stabilizer.push("A", 0.9, [{"label": "A", "conf": 0.9}], ts_ms=ts)

    assert stabilizer.committed_text == "A"

    for ts in range(7, 13):
        stabilizer.push("unknown", 0.0, [], ts_ms=ts)

    for ts in range(13, 19):
        stabilizer.push("A", 0.9, [{"label": "A", "conf": 0.9}], ts_ms=ts)

    assert stabilizer.committed_text == "AA"


def test_temporal_stabilizer_allows_repeat_after_unknown_transition() -> None:
    stabilizer = TemporalStabilizer(_settings_for_test())

    for ts in range(1, 4):
        stabilizer.push("A", 0.92, [{"label": "A", "conf": 0.92}], ts_ms=ts)
    assert stabilizer.committed_text == "A"

    for ts in range(4, 8):
        stabilizer.push("A", 0.92, [{"label": "A", "conf": 0.92}], ts_ms=ts)
    assert stabilizer.committed_text == "A"

    for ts in range(8, 11):
        stabilizer.push("unknown", 0.0, [], ts_ms=ts)

    for ts in range(11, 14):
        stabilizer.push("A", 0.92, [{"label": "A", "conf": 0.92}], ts_ms=ts)

    assert stabilizer.committed_text == "AA"


def test_temporal_stabilizer_brief_unknown_noise_does_not_reopen_duplicate_commit() -> None:
    stabilizer = TemporalStabilizer(_settings_for_test())

    for ts in range(1, 4):
        stabilizer.push("A", 0.92, [{"label": "A", "conf": 0.92}], ts_ms=ts)
    assert stabilizer.committed_text == "A"

    stabilizer.push("unknown", 0.0, [], ts_ms=4)

    for ts in range(5, 8):
        stabilizer.push("A", 0.92, [{"label": "A", "conf": 0.92}], ts_ms=ts)

    assert stabilizer.committed_text == "A"
