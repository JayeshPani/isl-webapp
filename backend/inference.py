from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Settings
from .utils.fps import FPSMeter
from .utils.hand_roi import HandROIExtractor, HandROIState
from .utils.postprocessing import normalize_topk, now_ms
from .utils.preprocessing import center_crop_square, center_zoom_square, resize_with_letterbox

try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None


@dataclass
class PredictionEvent:
    label: str
    confidence: float
    ts_ms: int


@dataclass
class DetectionCandidate:
    label: str
    confidence: float
    score: float
    area_ratio: float
    center_closeness: float


@dataclass
class ClassificationViewResult:
    view_name: str
    probs: np.ndarray
    top_label: str
    top_conf: float
    margin: float
    score: float


class TemporalStabilizer:
    """Smooth noisy per-frame predictions and commit reliable tokens."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.buffer: Deque[PredictionEvent] = deque(maxlen=settings.smoothing_window)
        self.committed_text: str = ""
        self.last_commit_ts: int = 0
        self.last_committed_label: str = ""
        self.repeat_gate_open: bool = True
        self.unknown_streak: int = 0

    def _sanitize_raw(self, raw_label: str, raw_conf: float, topk: List[Dict[str, float]]) -> Tuple[str, float]:
        label = raw_label or self.settings.unknown_label
        conf = float(raw_conf)

        if conf < self.settings.min_confidence:
            return self.settings.unknown_label, 0.0

        if len(topk) >= 2:
            margin = float(topk[0]["conf"]) - float(topk[1]["conf"])
            if margin < self.settings.min_margin:
                return self.settings.unknown_label, conf

        return label, conf

    def _compute_stable(self) -> Tuple[str, float]:
        valid = [
            event
            for event in self.buffer
            if event.label != self.settings.unknown_label and event.confidence >= self.settings.min_confidence
        ]

        if len(valid) < self.settings.min_votes:
            return self.settings.unknown_label, 0.0

        counts = Counter(event.label for event in valid)
        weights: Dict[str, float] = defaultdict(float)
        for event in valid:
            weights[event.label] += event.confidence

        ranked = sorted(weights.items(), key=lambda item: (item[1], counts[item[0]]), reverse=True)
        if not ranked:
            return self.settings.unknown_label, 0.0

        top_label, top_weight = ranked[0]
        top_votes = counts[top_label]
        second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_weight - second_weight

        if top_votes < self.settings.min_votes or margin < self.settings.min_margin:
            return self.settings.unknown_label, 0.0

        avg_conf = top_weight / max(1, top_votes)
        return top_label, float(avg_conf)

    def _commit_token(self, stable_label: str, ts_ms: int) -> Tuple[bool, Optional[str]]:
        if stable_label == self.settings.unknown_label:
            self.unknown_streak += 1
            if self.unknown_streak >= self.settings.duplicate_reset_unknown_frames:
                self.repeat_gate_open = True
            return False, None

        self.unknown_streak = 0

        if stable_label != self.last_committed_label:
            self.repeat_gate_open = True

        if ts_ms - self.last_commit_ts < self.settings.commit_cooldown_ms:
            return False, None

        if stable_label == self.settings.space_label:
            if self.committed_text and not self.committed_text.endswith(" "):
                self.committed_text += " "
                self.last_commit_ts = ts_ms
                self.last_committed_label = stable_label
                return True, " "
            return False, None

        if stable_label == self.settings.delete_label:
            if self.committed_text:
                self.committed_text = self.committed_text[:-1]
                self.last_commit_ts = ts_ms
                self.last_committed_label = stable_label
                return True, "<DELETE>"
            return False, None

        if self.settings.duplicate_block_until_changed:
            if stable_label == self.last_committed_label and not self.repeat_gate_open:
                return False, None
        elif stable_label == self.last_committed_label:
            if ts_ms - self.last_commit_ts < self.settings.duplicate_block_ms:
                return False, None

        self.committed_text += stable_label
        self.last_commit_ts = ts_ms
        self.last_committed_label = stable_label
        self.repeat_gate_open = False
        return True, stable_label

    def push(self, raw_label: str, raw_conf: float, topk: List[Dict[str, float]], ts_ms: int) -> Dict[str, object]:
        clean_label, clean_conf = self._sanitize_raw(raw_label, raw_conf, topk)
        self.buffer.append(PredictionEvent(label=clean_label, confidence=clean_conf, ts_ms=ts_ms))

        stable_label, stable_conf = self._compute_stable()
        token_committed, committed_token = self._commit_token(stable_label, ts_ms)

        return {
            "stable_pred": stable_label,
            "stable_conf": round(stable_conf, 4),
            "committed_text": self.committed_text,
            "token_committed": token_committed,
            "committed_token": committed_token,
            "buffer_state": [
                {"label": event.label, "conf": round(event.confidence, 3)} for event in list(self.buffer)
            ],
        }

    def clear_text(self) -> str:
        self.committed_text = ""
        self.last_committed_label = ""
        self.last_commit_ts = 0
        self.repeat_gate_open = True
        self.unknown_streak = 0
        return self.committed_text

    def backspace(self) -> str:
        if self.committed_text:
            self.committed_text = self.committed_text[:-1]
        return self.committed_text


class DynamicSequenceRecognizerHook:
    """Stub hook for future dynamic sign recognition (LSTM/TCN/Transformer)."""

    def __init__(self, enabled: bool = False, window: int = 24) -> None:
        self.enabled = enabled
        self.window = window
        self.sequence: Deque[PredictionEvent] = deque(maxlen=window)

    def push(self, label: str, confidence: float, ts_ms: int) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None

        self.sequence.append(PredictionEvent(label=label, confidence=confidence, ts_ms=ts_ms))
        return {
            "enabled": True,
            "frames_buffered": len(self.sequence),
            "dynamic_prediction": None,
            "note": "Attach sequence model here for dynamic sign support.",
        }


class YOLOInferenceEngine:
    """Model wrapper with support for direct classification and detection outputs."""

    DETECTION_CONF_WEIGHT = 0.60
    DETECTION_AREA_WEIGHT = 0.25
    DETECTION_CENTER_WEIGHT = 0.15

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None
        self.model_loaded = False
        self.model_error: Optional[str] = None
        self.class_names: Dict[int, str] = {idx: name for idx, name in enumerate(settings.mock_class_names)}
        self._lock = threading.Lock()

    def _normalize_names(self, names: object) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        return self.class_names

    def load_model(self) -> None:
        if YOLO is None:
            self.model_error = "ultralytics package is not available. Install requirements first."
            self.model_loaded = False
            return

        model_path = Path(self.settings.model_path)
        if not model_path.exists():
            self.model_error = f"Model file not found at '{model_path}'. Place your trained best.pt there."
            self.model_loaded = False
            return

        try:
            self.model = YOLO(str(model_path))
            names = getattr(self.model, "names", None)
            if names is None and hasattr(self.model, "model"):
                names = getattr(self.model.model, "names", None)
            self.class_names = self._normalize_names(names)
            self.model_loaded = True
            self.model_error = None
        except Exception as exc:  # noqa: BLE001
            self.model_error = f"Failed to load model: {exc}"
            self.model_loaded = False

    def _score_detection_candidate(
        self,
        confidence: float,
        xyxy: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        """
        Score a detection using confidence + bbox size + center proximity.

        Returns:
        - score: weighted score used to pick the best sign candidate
        - area_ratio: bbox area normalized by frame area
        - center_closeness: [0, 1], where 1 is closest to image center
        """
        image_h, image_w = image_hw
        x1, y1, x2, y2 = [float(v) for v in xyxy]

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        frame_area = max(1.0, float(image_w * image_h))
        area_ratio = min(1.0, (box_w * box_h) / frame_area)

        box_cx = (x1 + x2) / 2.0
        box_cy = (y1 + y2) / 2.0
        frame_cx = image_w / 2.0
        frame_cy = image_h / 2.0
        dist = ((box_cx - frame_cx) ** 2 + (box_cy - frame_cy) ** 2) ** 0.5
        max_dist = max(1e-6, (frame_cx**2 + frame_cy**2) ** 0.5)
        center_closeness = float(np.clip(1.0 - (dist / max_dist), 0.0, 1.0))

        score = (
            (confidence * self.DETECTION_CONF_WEIGHT)
            + (area_ratio * self.DETECTION_AREA_WEIGHT)
            + (center_closeness * self.DETECTION_CENTER_WEIGHT)
        )
        return score, area_ratio, center_closeness

    def _parse_detection(self, result: object) -> Tuple[str, float, List[Dict[str, float]]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self.settings.unknown_label, 0.0, []

        confs = getattr(boxes, "conf", None)
        clss = getattr(boxes, "cls", None)
        xyxy = getattr(boxes, "xyxy", None)
        if confs is None or clss is None or xyxy is None:
            return self.settings.unknown_label, 0.0, []

        confs_np = confs.cpu().numpy()
        clss_np = clss.cpu().numpy().astype(int)
        xyxy_np = xyxy.cpu().numpy()
        if confs_np.size == 0 or xyxy_np.size == 0:
            return self.settings.unknown_label, 0.0, []

        image_h, image_w = result.orig_shape[:2]
        total = min(len(confs_np), len(clss_np), len(xyxy_np))
        if total == 0:
            return self.settings.unknown_label, 0.0, []

        candidates: List[DetectionCandidate] = []
        for idx in range(total):
            cls_id = int(clss_np[idx])
            label = self.class_names.get(cls_id, str(cls_id))
            confidence = float(confs_np[idx])
            score, area_ratio, center_closeness = self._score_detection_candidate(
                confidence=confidence,
                xyxy=xyxy_np[idx],
                image_hw=(image_h, image_w),
            )
            candidates.append(
                DetectionCandidate(
                    label=label,
                    confidence=confidence,
                    score=score,
                    area_ratio=area_ratio,
                    center_closeness=center_closeness,
                )
            )

        best_candidate = max(candidates, key=lambda item: item.score)
        topk_candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        topk = [
            {"label": item.label, "conf": float(item.confidence)}
            for item in topk_candidates[: self.settings.topk]
        ]
        return best_candidate.label, float(best_candidate.confidence), topk

    def _extract_class_probs(self, result: object) -> np.ndarray:
        probs = getattr(result, "probs", None)
        if probs is None:
            return np.array([], dtype=float)

        data = getattr(probs, "data", None)
        if data is None:
            return np.array([], dtype=float)

        confs = np.array(data.detach().cpu().numpy(), dtype=float).flatten()
        if confs.size == 0:
            return np.array([], dtype=float)
        return confs

    def _topk_from_probs(self, confs: np.ndarray) -> List[Dict[str, float]]:
        if confs.size == 0:
            return []

        order = np.argsort(confs)[::-1]
        topk: List[Dict[str, float]] = []
        for idx in order[: self.settings.topk]:
            label = self.class_names.get(int(idx), str(int(idx)))
            topk.append({"label": label, "conf": float(confs[int(idx)])})
        return topk

    def _parse_classification(self, result: object) -> Tuple[str, float, List[Dict[str, float]]]:
        confs = self._extract_class_probs(result)
        topk = self._topk_from_probs(confs)
        if not topk:
            return self.settings.unknown_label, 0.0, []
        best = topk[0]
        return best["label"], float(best["conf"]), topk

    def _classification_view_score(self, top_conf: float, margin: float) -> float:
        """
        Score a classification view using confidence + top1/top2 margin.

        This score is used as the aggregation weight in multi-view direct_cls inference.
        """
        return float(max(1e-6, (0.75 * top_conf) + (0.25 * max(0.0, margin))))

    def _build_classification_views(self, frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        views: List[Tuple[str, np.ndarray]] = [("orig", frame)]

        zoom_frame: Optional[np.ndarray] = None
        if self.settings.direct_cls_include_zoom:
            ratio = float(np.clip(self.settings.direct_cls_zoom_ratio, 0.50, 1.0))
            zoom_frame = center_zoom_square(frame, ratio)
            views.append(("zoom", zoom_frame))

        if self.settings.direct_cls_include_flip:
            views.append(("flip", cv2.flip(frame, 1)))
            if zoom_frame is not None:
                views.append(("flip_zoom", cv2.flip(zoom_frame, 1)))

        return views

    def _aggregate_classification_views(
        self,
        view_outputs: List[ClassificationViewResult],
    ) -> Tuple[str, float, List[Dict[str, float]]]:
        if not view_outputs:
            return self.settings.unknown_label, 0.0, []

        if len(view_outputs) == 1:
            only = view_outputs[0]
            topk = self._topk_from_probs(only.probs)
            if not topk:
                return self.settings.unknown_label, 0.0, []
            return only.top_label, float(topk[0]["conf"]), topk

        weighted_sum = np.zeros_like(view_outputs[0].probs, dtype=float)
        weight_total = 0.0
        vote_counter: Counter[str] = Counter()

        for view in view_outputs:
            weighted_sum += view.probs * view.score
            weight_total += view.score
            vote_counter[view.top_label] += 1

        if weight_total <= 0.0:
            return self.settings.unknown_label, 0.0, []

        aggregated_probs = weighted_sum / weight_total
        topk = self._topk_from_probs(aggregated_probs)
        if not topk:
            return self.settings.unknown_label, 0.0, []

        best_label = topk[0]["label"]
        best_conf = float(topk[0]["conf"])

        min_votes = max(1, self.settings.direct_cls_min_view_votes)
        if vote_counter.get(best_label, 0) < min_votes:
            best_conf *= 0.75
            topk[0]["conf"] = best_conf

        return best_label, best_conf, topk

    def _predict_direct_cls(self, frame: np.ndarray) -> Tuple[str, float, List[Dict[str, float]]]:
        sources: List[np.ndarray]
        view_names: List[str]
        if self.settings.direct_cls_enable_tta:
            views = self._build_classification_views(frame)
            view_names = [name for name, _ in views]
            sources = [img for _, img in views]
        else:
            view_names = ["orig"]
            sources = [frame]

        with self._lock:
            results = self.model.predict(
                source=sources,
                imgsz=self.settings.frame_size,
                verbose=False,
                conf=0.001,
            )

        if not results:
            return self.settings.unknown_label, 0.0, []

        view_outputs: List[ClassificationViewResult] = []
        for idx, result in enumerate(results[: len(view_names)]):
            probs = self._extract_class_probs(result)
            if probs.size == 0:
                continue

            topk = self._topk_from_probs(probs)
            if not topk:
                continue

            top_conf = float(topk[0]["conf"])
            second_conf = float(topk[1]["conf"]) if len(topk) > 1 else 0.0
            margin = top_conf - second_conf

            view_outputs.append(
                ClassificationViewResult(
                    view_name=view_names[idx],
                    probs=probs,
                    top_label=str(topk[0]["label"]),
                    top_conf=top_conf,
                    margin=margin,
                    score=self._classification_view_score(top_conf, margin),
                )
            )

        return self._aggregate_classification_views(view_outputs)

    def predict(self, frame: np.ndarray) -> Tuple[str, float, List[Dict[str, float]]]:
        if not self.model_loaded or self.model is None:
            return self.settings.unknown_label, 0.0, []

        mode = self.settings.model_mode.lower()
        if mode == "direct_cls":
            return self._predict_direct_cls(frame)

        with self._lock:
            results = self.model.predict(
                source=frame,
                imgsz=self.settings.frame_size,
                verbose=False,
                conf=0.001,
            )

        if not results:
            return self.settings.unknown_label, 0.0, []

        result = results[0]
        if mode == "detect":
            return self._parse_detection(result)

        if getattr(result, "probs", None) is not None:
            return self._parse_classification(result)
        return self._parse_detection(result)


@dataclass
class SessionState:
    stabilizer: TemporalStabilizer
    fps_meter: FPSMeter
    sequence_hook: DynamicSequenceRecognizerHook
    hand_roi_state: HandROIState


class ISLInferenceService:
    """Coordinator for model inference, temporal smoothing, and per-client text state."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = YOLOInferenceEngine(settings)
        self.engine.load_model()
        self.hand_roi = HandROIExtractor(settings)
        self._sessions: Dict[str, SessionState] = {}
        self._session_lock = threading.Lock()

    def _get_session(self, session_id: str) -> SessionState:
        with self._session_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(
                    stabilizer=TemporalStabilizer(self.settings),
                    fps_meter=FPSMeter(window_size=30),
                    sequence_hook=DynamicSequenceRecognizerHook(enabled=self.settings.enable_sequence_hook),
                    hand_roi_state=HandROIState(),
                )
            return self._sessions[session_id]

    def remove_session(self, session_id: str) -> None:
        with self._session_lock:
            self._sessions.pop(session_id, None)

    def handle_control(self, session_id: str, action: str) -> Dict[str, object]:
        session = self._get_session(session_id)
        if action == "clear_text":
            text = session.stabilizer.clear_text()
            return {"ok": True, "committed_text": text}
        if action == "backspace":
            text = session.stabilizer.backspace()
            return {"ok": True, "committed_text": text}
        return {"ok": False, "error": f"Unknown control action '{action}'"}

    def process_frame(
        self,
        frame: np.ndarray,
        session_id: str,
        client_ts_ms: Optional[int] = None,
    ) -> Dict[str, object]:
        started = time.perf_counter()
        session = self._get_session(session_id)

        square = center_crop_square(frame)
        hand_roi_result = None
        infer_input = square
        if self.hand_roi.should_apply(self.settings.model_mode):
            hand_roi_result = self.hand_roi.extract(square, session.hand_roi_state)
            infer_input = hand_roi_result.frame

        prepared = resize_with_letterbox(infer_input, self.settings.frame_size)
        raw_pred, raw_conf, topk = self.engine.predict(prepared)
        normalized_topk = normalize_topk(topk, self.settings.topk) if self.settings.enable_topk else []

        ts_ms = now_ms()
        smoothed = session.stabilizer.push(raw_pred, raw_conf, normalized_topk, ts_ms)
        sequence_output = session.sequence_hook.push(raw_pred, raw_conf, ts_ms)

        latency_ms = (time.perf_counter() - started) * 1000.0
        model_fps_estimate = session.fps_meter.tick()

        payload: Dict[str, object] = {
            "raw_pred": raw_pred,
            "raw_conf": round(float(raw_conf), 4),
            "stable_pred": smoothed["stable_pred"],
            "stable_conf": smoothed["stable_conf"],
            "committed_text": smoothed["committed_text"],
            "token_committed": smoothed["token_committed"],
            "committed_token": smoothed["committed_token"],
            "topk": normalized_topk,
            "latency_ms": round(latency_ms, 2),
            "model_fps_estimate": round(model_fps_estimate, 2),
            "server_ts": ts_ms,
            "client_ts": client_ts_ms,
            "model_loaded": self.engine.model_loaded,
            "model_error": self.engine.model_error,
        }

        if self.settings.debug:
            payload["buffer_state"] = smoothed["buffer_state"]
            payload["sequence_hook"] = sequence_output
            payload["hand_roi"] = (
                {
                    "enabled": self.hand_roi.enabled,
                    "available": self.hand_roi.available,
                    "provider": self.hand_roi.provider,
                    "used_hand_roi": bool(hand_roi_result.used_hand_roi) if hand_roi_result is not None else False,
                    "reused_last_bbox": bool(hand_roi_result.reused_last_bbox) if hand_roi_result is not None else False,
                    "bbox": list(hand_roi_result.bbox) if hand_roi_result and hand_roi_result.bbox else None,
                    "note": hand_roi_result.note if hand_roi_result is not None else None,
                }
            )

        return payload

    def model_status(self) -> Dict[str, object]:
        return {
            "model_loaded": self.engine.model_loaded,
            "model_error": self.engine.model_error,
            "model_path": str(self.settings.model_path),
            "mode": self.settings.model_mode,
            "frame_size": self.settings.frame_size,
            "send_fps": self.settings.send_fps,
            "min_confidence": self.settings.min_confidence,
            "min_margin": self.settings.min_margin,
            "unknown_label": self.settings.unknown_label,
            "hand_roi_enabled": self.hand_roi.enabled,
            "hand_roi_available": self.hand_roi.available,
            "hand_roi_provider": self.hand_roi.provider,
            "hand_roi_error": self.hand_roi.error,
            "direct_cls_enable_tta": self.settings.direct_cls_enable_tta,
            "direct_cls_include_flip": self.settings.direct_cls_include_flip,
            "direct_cls_include_zoom": self.settings.direct_cls_include_zoom,
            "direct_cls_zoom_ratio": self.settings.direct_cls_zoom_ratio,
            "classes": list(self.engine.class_names.values()),
        }
