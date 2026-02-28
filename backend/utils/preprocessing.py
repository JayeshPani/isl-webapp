from __future__ import annotations

import base64
from typing import Tuple

import cv2
import numpy as np


def decode_image_payload(image_payload: str) -> np.ndarray:
    """Decode a base64 image payload (data URL or raw base64) into BGR frame."""
    if not image_payload:
        raise ValueError("Empty image payload")

    payload = image_payload
    if image_payload.startswith("data:image") and "," in image_payload:
        payload = image_payload.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 image payload") from exc

    return decode_image_bytes(image_bytes)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (jpeg/png/etc) into BGR frame."""
    if not image_bytes:
        raise ValueError("Empty image bytes")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes")
    return frame


def resize_with_letterbox(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image to target square while preserving aspect ratio with padding."""
    if target_size <= 0:
        return frame

    height, width = frame.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("Invalid frame dimensions")

    scale = min(target_size / width, target_size / height)
    new_w, new_h = int(width * scale), int(height * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    """Center-crop the frame into a square ROI for model inference."""
    height, width = frame.shape[:2]
    size = min(width, height)
    x0 = (width - size) // 2
    y0 = (height - size) // 2
    return frame[y0 : y0 + size, x0 : x0 + size]


def center_zoom_square(frame: np.ndarray, zoom_ratio: float) -> np.ndarray:
    """
    Zoom into the center of a square frame and resize back to original shape.

    `zoom_ratio` must be in (0, 1]. Lower values produce a tighter center crop.
    """
    if zoom_ratio >= 1.0:
        return frame
    if zoom_ratio <= 0.0:
        raise ValueError("zoom_ratio must be > 0")

    height, width = frame.shape[:2]
    side = min(height, width)
    crop_side = max(2, int(side * zoom_ratio))

    x0 = (width - crop_side) // 2
    y0 = (height - crop_side) // 2
    cropped = frame[y0 : y0 + crop_side, x0 : x0 + crop_side]
    if cropped.size == 0:
        return frame

    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def decode_and_prepare(image_payload: str, frame_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Decode payload and return original + preprocessed frame."""
    frame = decode_image_payload(image_payload)
    square = center_crop_square(frame)
    prepared = resize_with_letterbox(square, frame_size)
    return frame, prepared
