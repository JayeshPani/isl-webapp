from __future__ import annotations

import base64

import cv2
import numpy as np

from backend.utils.preprocessing import center_zoom_square, decode_image_payload


def test_decode_image_payload_roundtrip() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 255

    ok, encoded = cv2.imencode(".jpg", image)
    assert ok is True

    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    payload = f"data:image/jpeg;base64,{b64}"

    decoded = decode_image_payload(payload)
    assert decoded.shape[0] == 64
    assert decoded.shape[1] == 64
    assert decoded.shape[2] == 3


def test_center_zoom_square_preserves_shape() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[30:50, 30:50] = 255
    zoomed = center_zoom_square(image, 0.75)

    assert zoomed.shape == image.shape
    assert np.mean(zoomed) > np.mean(image)
