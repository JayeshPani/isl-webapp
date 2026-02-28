from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv


def _to_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _to_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _to_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


@dataclass(frozen=True)
class Settings:
    model_path: Path
    model_mode: str
    frame_size: int
    send_fps: int
    min_confidence: float
    min_margin: float
    smoothing_window: int
    min_votes: int
    commit_cooldown_ms: int
    duplicate_block_until_changed: bool
    duplicate_block_ms: int
    duplicate_reset_unknown_frames: int
    unknown_label: str
    space_label: str
    delete_label: str
    enable_hand_roi: bool
    hand_roi_only_for_direct_cls: bool
    hand_roi_mask_background: bool
    hand_roi_expand_ratio: float
    hand_roi_smoothing: float
    hand_roi_min_area_ratio: float
    hand_roi_reuse_last_bbox_frames: int
    hand_roi_max_num_hands: int
    hand_roi_min_detection_confidence: float
    hand_roi_min_tracking_confidence: float
    direct_cls_enable_tta: bool
    direct_cls_include_flip: bool
    direct_cls_include_zoom: bool
    direct_cls_zoom_ratio: float
    direct_cls_min_view_votes: int
    enable_topk: bool
    topk: int
    debug: bool
    enable_sequence_hook: bool
    host: str
    port: int
    mock_class_names: List[str]

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env")
        default_model = project_root / "backend" / "models" / "best.pt"
        model_path = Path(os.getenv("MODEL_PATH", str(default_model))).expanduser()
        if not model_path.is_absolute():
            model_path = project_root / model_path

        return cls(
            model_path=model_path,
            model_mode=os.getenv("MODEL_MODE", "detect").strip(),
            frame_size=_to_int("FRAME_SIZE", 416),
            send_fps=_to_int("SEND_FPS", 15),
            min_confidence=_to_float("MIN_CONFIDENCE", 0.45),
            min_margin=_to_float("MIN_MARGIN", 0.10),
            smoothing_window=_to_int("SMOOTHING_WINDOW", 8),
            min_votes=_to_int("MIN_VOTES", 4),
            commit_cooldown_ms=_to_int("COMMIT_COOLDOWN_MS", 700),
            duplicate_block_until_changed=_to_bool("DUPLICATE_BLOCK_UNTIL_CHANGED", True),
            duplicate_block_ms=_to_int("DUPLICATE_BLOCK_MS", 1400),
            duplicate_reset_unknown_frames=_to_int("DUPLICATE_RESET_UNKNOWN_FRAMES", 3),
            unknown_label=os.getenv("UNKNOWN_LABEL", "unknown").strip(),
            space_label=os.getenv("SPACE_LABEL", "space").strip(),
            delete_label=os.getenv("DELETE_LABEL", "delete").strip(),
            enable_hand_roi=_to_bool("ENABLE_HAND_ROI", False),
            hand_roi_only_for_direct_cls=_to_bool("HAND_ROI_ONLY_FOR_DIRECT_CLS", True),
            hand_roi_mask_background=_to_bool("HAND_ROI_MASK_BACKGROUND", True),
            hand_roi_expand_ratio=_to_float("HAND_ROI_EXPAND_RATIO", 0.35),
            hand_roi_smoothing=_to_float("HAND_ROI_SMOOTHING", 0.55),
            hand_roi_min_area_ratio=_to_float("HAND_ROI_MIN_AREA_RATIO", 0.03),
            hand_roi_reuse_last_bbox_frames=_to_int("HAND_ROI_REUSE_LAST_BBOX_FRAMES", 4),
            hand_roi_max_num_hands=_to_int("HAND_ROI_MAX_NUM_HANDS", 1),
            hand_roi_min_detection_confidence=_to_float("HAND_ROI_MIN_DETECTION_CONFIDENCE", 0.50),
            hand_roi_min_tracking_confidence=_to_float("HAND_ROI_MIN_TRACKING_CONFIDENCE", 0.50),
            direct_cls_enable_tta=_to_bool("DIRECT_CLS_ENABLE_TTA", True),
            direct_cls_include_flip=_to_bool("DIRECT_CLS_INCLUDE_FLIP", True),
            direct_cls_include_zoom=_to_bool("DIRECT_CLS_INCLUDE_ZOOM", True),
            direct_cls_zoom_ratio=_to_float("DIRECT_CLS_ZOOM_RATIO", 0.78),
            direct_cls_min_view_votes=_to_int("DIRECT_CLS_MIN_VIEW_VOTES", 2),
            enable_topk=_to_bool("ENABLE_TOPK", True),
            topk=_to_int("TOPK", 3),
            debug=_to_bool("DEBUG", True),
            enable_sequence_hook=_to_bool("ENABLE_SEQUENCE_HOOK", False),
            host=os.getenv("HOST", "127.0.0.1").strip(),
            port=_to_int("PORT", 8000),
            mock_class_names=_to_list(
                "MOCK_CLASS_NAMES",
                [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "HELLO",
                    "THANK_YOU",
                    "space",
                    "delete",
                    "unknown",
                ],
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached app settings loaded from the environment."""
    return Settings.from_env()
