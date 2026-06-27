"""Shared test fixtures — a fake Gemini client so no network calls happen."""

from __future__ import annotations

import pytest
from PIL import Image


class FakeResponse:
    """Mimics the GenAI response object.

    The detector prefers ``.parsed`` (populated by the SDK when a
    ``response_schema`` is set) and falls back to ``.text``. This fixture can
    model either production shape: pass ``parsed`` to exercise the primary path,
    or ``text`` to exercise the fallback parser.
    """

    def __init__(self, text: str | None = None, parsed=None):
        self.text = text
        self.parsed = parsed
        self.candidates = []


class FakeModels:
    def __init__(self, response_text: str | None, recorder: dict, parsed=None):
        self._response_text = response_text
        self._parsed = parsed
        self._recorder = recorder

    def generate_content(self, *, model, contents, config):
        # Record the call so tests can assert on what was sent.
        self._recorder["model"] = model
        self._recorder["contents"] = contents
        self._recorder["config"] = config
        self._recorder["calls"] = self._recorder.get("calls", 0) + 1
        return FakeResponse(self._response_text, self._parsed)


class FakeClient:
    """Drop-in stand-in for ``genai.Client``; injected via ``Detector(client=...)``."""

    def __init__(self, response_text: str | None = None, parsed=None):
        self.calls = {}
        self.models = FakeModels(response_text, self.calls, parsed)


@pytest.fixture
def make_client():
    """Factory: build a FakeClient.

    ``make_client(text)`` exercises the text-fallback parser;
    ``make_client(parsed=[...])`` exercises the primary ``response.parsed`` path.
    """

    def _make(response_text: str | None = None, parsed=None) -> FakeClient:
        return FakeClient(response_text, parsed)

    return _make


@pytest.fixture
def image_path(tmp_path):
    """A real 400x200 PNG on disk so Pillow can read its size and crop it."""
    path = tmp_path / "scene.png"
    Image.new("RGB", (400, 200), (210, 210, 210)).save(path)
    return path
