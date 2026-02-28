from __future__ import annotations

from dataclasses import replace

import numpy as np

from backend.config import Settings
from backend.utils.hand_roi import HandROIExtractor, HandROIState


def _settings() -> Settings:
    base = Settings.from_env()
    return replace(base, enable_hand_roi=False, hand_roi_only_for_direct_cls=True)


def test_hand_roi_disabled_returns_original_frame() -> None:
    settings = _settings()
    extractor = HandROIExtractor(settings)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)

    result = extractor.extract(frame, HandROIState())
    assert result.used_hand_roi is False
    assert result.note == "disabled"
    assert result.frame.shape == frame.shape


def test_hand_roi_mode_gate_applies_only_to_direct_cls_by_default() -> None:
    settings = replace(_settings(), enable_hand_roi=True)
    extractor = HandROIExtractor(settings)

    assert extractor.should_apply("direct_cls") is True
    assert extractor.should_apply("detect") is False


def test_hand_roi_unavailable_falls_back_without_crash() -> None:
    settings = replace(_settings(), enable_hand_roi=True)
    extractor = HandROIExtractor(settings)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)

    if extractor.available:
        return

    result = extractor.extract(frame, HandROIState())
    assert result.used_hand_roi is False
    assert result.available is False
    assert result.note is not None
