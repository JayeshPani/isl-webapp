from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import Settings


BBox = Tuple[int, int, int, int]


@dataclass
class HandROIState:
    """Per-session state to smooth ROI and reuse recent bbox on brief detector misses."""

    last_bbox: Optional[BBox] = None
    miss_streak: int = 0


@dataclass
class HandROIResult:
    """Output of hand-ROI extraction."""

    frame: np.ndarray
    used_hand_roi: bool
    reused_last_bbox: bool
    bbox: Optional[BBox]
    provider: str
    available: bool
    note: Optional[str] = None


class HandROIExtractor:
    """
    Optional hand ROI extractor using MediaPipe Hands.

    If MediaPipe is unavailable, extraction falls back to the original frame.
    """

    AREA_WEIGHT = 0.65
    CENTER_WEIGHT = 0.35

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.enable_hand_roi
        self.available = False
        self.provider = "disabled"
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._hands = None

        if not self.enabled:
            return

        try:
            import mediapipe as mp

            mp_hands = mp.solutions.hands
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max(1, settings.hand_roi_max_num_hands),
                min_detection_confidence=float(np.clip(settings.hand_roi_min_detection_confidence, 0.0, 1.0)),
                min_tracking_confidence=float(np.clip(settings.hand_roi_min_tracking_confidence, 0.0, 1.0)),
            )
            self.available = True
            self.provider = "mediapipe"
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.available = False
            self.provider = "unavailable"

    def should_apply(self, model_mode: str) -> bool:
        if not self.enabled:
            return False
        if self.settings.hand_roi_only_for_direct_cls and model_mode.lower() != "direct_cls":
            return False
        return True

    def _score_bbox(self, bbox: BBox, image_hw: Tuple[int, int]) -> float:
        image_h, image_w = image_hw
        x1, y1, x2, y2 = bbox
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        frame_area = max(1.0, float(image_w * image_h))
        area_ratio = float((box_w * box_h) / frame_area)

        box_cx = (x1 + x2) / 2.0
        box_cy = (y1 + y2) / 2.0
        frame_cx = image_w / 2.0
        frame_cy = image_h / 2.0
        dist = ((box_cx - frame_cx) ** 2 + (box_cy - frame_cy) ** 2) ** 0.5
        max_dist = max(1e-6, (frame_cx**2 + frame_cy**2) ** 0.5)
        center_closeness = float(np.clip(1.0 - (dist / max_dist), 0.0, 1.0))

        return (area_ratio * self.AREA_WEIGHT) + (center_closeness * self.CENTER_WEIGHT)

    def _smooth_bbox(self, previous: BBox, current: BBox, image_hw: Tuple[int, int]) -> BBox:
        alpha = float(np.clip(self.settings.hand_roi_smoothing, 0.0, 0.95))
        x1 = int((alpha * previous[0]) + ((1.0 - alpha) * current[0]))
        y1 = int((alpha * previous[1]) + ((1.0 - alpha) * current[1]))
        x2 = int((alpha * previous[2]) + ((1.0 - alpha) * current[2]))
        y2 = int((alpha * previous[3]) + ((1.0 - alpha) * current[3]))

        image_h, image_w = image_hw
        x1 = int(np.clip(x1, 0, max(0, image_w - 2)))
        y1 = int(np.clip(y1, 0, max(0, image_h - 2)))
        x2 = int(np.clip(x2, x1 + 1, max(1, image_w - 1)))
        y2 = int(np.clip(y2, y1 + 1, max(1, image_h - 1)))
        return (x1, y1, x2, y2)

    def _landmarks_to_bbox_and_points(
        self,
        hand_landmarks: object,
        image_hw: Tuple[int, int],
    ) -> Tuple[Optional[BBox], Optional[np.ndarray]]:
        image_h, image_w = image_hw
        points: list[Tuple[float, float]] = []
        for landmark in hand_landmarks.landmark:
            px = float(landmark.x) * image_w
            py = float(landmark.y) * image_h
            points.append((px, py))

        if not points:
            return None, None

        pts = np.array(points, dtype=np.float32)
        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        expand = max(0.0, float(self.settings.hand_roi_expand_ratio))
        x_pad = box_w * expand
        y_pad = box_h * expand

        xi1 = int(np.clip(x1 - x_pad, 0, max(0, image_w - 2)))
        yi1 = int(np.clip(y1 - y_pad, 0, max(0, image_h - 2)))
        xi2 = int(np.clip(x2 + x_pad, xi1 + 1, max(1, image_w - 1)))
        yi2 = int(np.clip(y2 + y_pad, yi1 + 1, max(1, image_h - 1)))

        int_points = np.stack(
            [
                np.clip(pts[:, 0], 0, max(1, image_w - 1)).astype(np.int32),
                np.clip(pts[:, 1], 0, max(1, image_h - 1)).astype(np.int32),
            ],
            axis=1,
        )
        return (xi1, yi1, xi2, yi2), int_points

    def _pick_best_hand(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[BBox], Optional[np.ndarray]]:
        if self._hands is None:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            result = self._hands.process(rgb)

        landmarks = getattr(result, "multi_hand_landmarks", None)
        if not landmarks:
            return None, None

        image_h, image_w = frame.shape[:2]
        best_bbox: Optional[BBox] = None
        best_points: Optional[np.ndarray] = None
        best_score = -1.0

        for hand_landmarks in landmarks:
            bbox, points = self._landmarks_to_bbox_and_points(hand_landmarks, (image_h, image_w))
            if bbox is None or points is None:
                continue

            score = self._score_bbox(bbox, (image_h, image_w))
            if score > best_score:
                best_bbox = bbox
                best_points = points
                best_score = score

        return best_bbox, best_points

    def _mask_frame(self, frame: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
        if hand_points.size == 0:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(hand_points)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def _crop(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]

    def extract(self, frame: np.ndarray, state: HandROIState) -> HandROIResult:
        if not self.enabled:
            return HandROIResult(
                frame=frame,
                used_hand_roi=False,
                reused_last_bbox=False,
                bbox=None,
                provider=self.provider,
                available=self.available,
                note="disabled",
            )
        if not self.available:
            note = f"hand ROI unavailable: {self.error}" if self.error else "hand ROI unavailable"
            return HandROIResult(
                frame=frame,
                used_hand_roi=False,
                reused_last_bbox=False,
                bbox=None,
                provider=self.provider,
                available=self.available,
                note=note,
            )

        image_h, image_w = frame.shape[:2]
        bbox, hand_points = self._pick_best_hand(frame)
        reused_last_bbox = False

        if bbox is None:
            state.miss_streak += 1
            can_reuse = (
                state.last_bbox is not None
                and state.miss_streak <= max(0, self.settings.hand_roi_reuse_last_bbox_frames)
            )
            if not can_reuse:
                return HandROIResult(
                    frame=frame,
                    used_hand_roi=False,
                    reused_last_bbox=False,
                    bbox=None,
                    provider=self.provider,
                    available=self.available,
                    note="no hand detected",
                )
            bbox = state.last_bbox
            reused_last_bbox = True
        else:
            state.miss_streak = 0
            if state.last_bbox is not None:
                bbox = self._smooth_bbox(state.last_bbox, bbox, (image_h, image_w))
            state.last_bbox = bbox

        if bbox is None:
            return HandROIResult(
                frame=frame,
                used_hand_roi=False,
                reused_last_bbox=False,
                bbox=None,
                provider=self.provider,
                available=self.available,
                note="invalid bbox",
            )

        x1, y1, x2, y2 = bbox
        area_ratio = float((max(0, x2 - x1) * max(0, y2 - y1)) / max(1.0, float(image_w * image_h)))
        if area_ratio < max(0.0, self.settings.hand_roi_min_area_ratio):
            return HandROIResult(
                frame=frame,
                used_hand_roi=False,
                reused_last_bbox=False,
                bbox=None,
                provider=self.provider,
                available=self.available,
                note="hand bbox too small",
            )

        source = frame
        if self.settings.hand_roi_mask_background and hand_points is not None and not reused_last_bbox:
            source = self._mask_frame(frame, hand_points)

        cropped = self._crop(source, bbox)
        if cropped.size == 0:
            return HandROIResult(
                frame=frame,
                used_hand_roi=False,
                reused_last_bbox=False,
                bbox=None,
                provider=self.provider,
                available=self.available,
                note="empty hand crop",
            )

        return HandROIResult(
            frame=cropped,
            used_hand_roi=True,
            reused_last_bbox=reused_last_bbox,
            bbox=bbox,
            provider=self.provider,
            available=self.available,
            note=None,
        )
