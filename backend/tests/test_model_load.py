from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from backend.config import Settings
from backend.inference import ISLInferenceService


def test_model_load_fails_gracefully_when_weights_missing() -> None:
    base = Settings.from_env()
    missing_path = Path("/tmp/this_model_file_should_not_exist_isl.pt")
    settings = replace(base, model_path=missing_path)

    service = ISLInferenceService(settings)
    status = service.model_status()

    assert status["model_loaded"] is False
    assert status["model_error"] is not None
