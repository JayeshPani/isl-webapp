from __future__ import annotations

import time
from typing import Dict, Iterable, List


def now_ms() -> int:
    """Current UNIX timestamp in milliseconds."""
    return int(time.time() * 1000)


def normalize_topk(topk: Iterable[Dict[str, float]], limit: int = 3) -> List[Dict[str, float]]:
    """Normalize top-k predictions into a compact JSON-friendly format."""
    cleaned: List[Dict[str, float]] = []
    for item in topk:
        label = str(item.get("label", ""))
        conf = float(item.get("conf", 0.0))
        cleaned.append({"label": label, "conf": round(conf, 4)})

    cleaned.sort(key=lambda x: x["conf"], reverse=True)
    return cleaned[:limit]


def ui_label(label: str, unknown_label: str) -> str:
    """Map internal unknown label to a user-facing status string."""
    if not label or label == unknown_label:
        return "No sign detected"
    return label
