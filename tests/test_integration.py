"""Opt-in integration test against the real Gemini API.

Skipped unless credentials are present, so CI (which has none) stays green while
still letting you catch real response-shape breakage locally:

    GEMINI_API_KEY=AIza...     pytest tests/test_integration.py
    GOOGLE_CLOUD_PROJECT=proj  pytest tests/test_integration.py   # Vertex AI
"""

import os
from pathlib import Path

import pytest

from prompt2box import DetectionResult, detect

_HAS_DEV_KEY = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
_HAS_VERTEX = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))

pytestmark = pytest.mark.skipif(
    not (_HAS_DEV_KEY or _HAS_VERTEX),
    reason="no API credentials (set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT to run)",
)

_IMAGE = Path(__file__).resolve().parent.parent / "docs" / "example2.jpg"


def test_real_detection_smoke():
    kwargs = {} if _HAS_DEV_KEY else {"vertexai": True}
    result = detect(str(_IMAGE), **kwargs)

    assert isinstance(result, DetectionResult)
    w, h = result.image_size
    assert w > 0 and h > 0
    # Every returned box must be in-bounds and well-formed.
    for d in result:
        assert d.label
        assert 0 <= d.x_min <= d.x_max <= w
        assert 0 <= d.y_min <= d.y_max <= h
