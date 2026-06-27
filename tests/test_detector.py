"""Tests for Detector/detect, result helpers, and drawing — fully mocked."""

import json

import pytest

from prompt2box import DetectionResult, Detector, detect
from prompt2box.detector import _generate_with_retry

RESPONSE = json.dumps(
    [
        {"label": "cat", "box_2d": [100, 50, 500, 250]},
        {"label": "sofa", "box_2d": [400, 300, 900, 950]},
    ]
)


def test_detect_returns_result(make_client, image_path):
    det = Detector(client=make_client(RESPONSE))
    result = det.detect(image_path)
    assert isinstance(result, DetectionResult)
    assert result.labels == ["cat", "sofa"]
    assert result.image_size == (400, 200)
    assert len(result) == 2


def test_detect_convenience_with_injected_client(make_client, image_path):
    result = detect(image_path, client=make_client(RESPONSE))
    assert result.labels == ["cat", "sofa"]


def test_classes_param_shapes_prompt(make_client, image_path):
    client = make_client(RESPONSE)
    Detector(client=client).detect(image_path, classes=["cat", "dog"])
    sent_text = client.calls["contents"][-1].text
    assert "cat, dog" in sent_text


def test_prompt_overrides_classes(make_client, image_path):
    client = make_client(RESPONSE)
    Detector(client=client).detect(image_path, prompt="only the rug", classes=["cat"])
    assert client.calls["contents"][-1].text == "only the rug"


def test_missing_file_raises(make_client, tmp_path):
    det = Detector(client=make_client(RESPONSE))
    with pytest.raises(FileNotFoundError):
        det.detect(tmp_path / "nope.png")


def test_empty_response_raises(make_client, image_path):
    det = Detector(client=make_client("   "))
    with pytest.raises(ValueError):
        det.detect(image_path)


def test_result_helpers(make_client, image_path):
    result = Detector(client=make_client(RESPONSE)).detect(image_path)

    assert result.filter("cat").labels == ["cat"]
    assert result[0].label == "cat"

    as_dict = result.to_dict()
    assert as_dict["width"] == 400 and as_dict["height"] == 200
    assert len(as_dict["objects"]) == 2

    parsed = json.loads(result.to_json())
    assert parsed[0]["label"] == "cat"
    assert {"x_min", "y_min", "x_max", "y_max", "box_normalized"} <= parsed[0].keys()


def test_detection_geometry_and_crop(make_client, image_path, tmp_path):
    cat = Detector(client=make_client(RESPONSE)).detect(image_path)[0]
    assert cat.box == (cat.x_min, cat.y_min, cat.x_max, cat.y_max)
    assert cat.width == cat.x_max - cat.x_min
    assert cat.area == cat.width * cat.height

    out = cat.crop(image_path, tmp_path / "cat.png")
    assert out.is_file()
    from PIL import Image

    with Image.open(out) as im:
        assert im.size == (cat.width, cat.height)


def test_save_annotated_writes_image(make_client, image_path, tmp_path):
    result = Detector(client=make_client(RESPONSE)).detect(image_path)
    out = result.save_annotated(tmp_path / "boxed.png")
    assert out.is_file()
    from PIL import Image

    with Image.open(out) as im:
        assert im.size == (400, 200)  # same dimensions as source


def test_detect_batch_runs_all_images(make_client, tmp_path):
    from PIL import Image

    paths = []
    for name in ("a.png", "b.png", "c.png"):
        p = tmp_path / name
        Image.new("RGB", (400, 200), (200, 200, 200)).save(p)
        paths.append(p)

    results = Detector(client=make_client(RESPONSE)).detect_batch(paths, max_workers=2)
    assert len(results) == 3
    assert all(r is not None and r.labels == ["cat", "sofa"] for r in results)


def test_detect_batch_isolates_failures(make_client, tmp_path):
    from PIL import Image

    good = tmp_path / "good.png"
    Image.new("RGB", (400, 200), (200, 200, 200)).save(good)
    missing = tmp_path / "missing.png"  # never created -> FileNotFoundError

    results = Detector(client=make_client(RESPONSE)).detect_batch([good, missing])
    assert results[0] is not None and results[0].labels == ["cat", "sofa"]
    assert results[1] is None  # failure isolated, batch not aborted


def test_detect_batch_empty_returns_empty(make_client):
    assert Detector(client=make_client(RESPONSE)).detect_batch([]) == []


def test_aq_key_warns(monkeypatch, caplog):
    """Constructing with an AQ. key should emit a helpful warning."""
    import logging

    import prompt2box.detector as det_mod

    monkeypatch.setattr(det_mod.genai, "Client", lambda **kw: object())
    with caplog.at_level(logging.WARNING):
        Detector(api_key="AQ.Ab8fake")
    assert any("AQ." in rec.message and "Vertex" in rec.message for rec in caplog.records)


def test_auth_error_is_made_friendly(make_client, image_path):
    """A 401 ACCESS_TOKEN_TYPE_UNSUPPORTED should be re-raised with guidance."""

    class FailingClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise RuntimeError("401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED")

    with pytest.raises(RuntimeError, match="Vertex AI"):
        Detector(client=FailingClient()).detect(image_path)


def test_non_auth_error_passes_through_unchanged(image_path):
    """A non-auth API error must propagate as-is, not get wrapped."""

    class FailingClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise KeyError("some unrelated bug")

    with pytest.raises(KeyError, match="unrelated"):
        Detector(client=FailingClient()).detect(image_path)


def _response_with_finish(finish_reason):
    """Build a real (empty-text) GenerateContentResponse with the given enum."""
    from google.genai import types

    cand = types.Candidate(finish_reason=finish_reason, content=types.Content(parts=[]))
    return types.GenerateContentResponse(candidates=[cand])


def test_safety_block_gives_informative_error(image_path):
    """An empty response with a real SAFETY finish_reason explains why."""
    from google.genai import types

    class BlockingClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return _response_with_finish(types.FinishReason.SAFETY)

    with pytest.raises(ValueError, match="finish_reason=SAFETY"):
        Detector(client=BlockingClient()).detect(image_path)


def test_empty_stop_is_not_misreported_as_safety(image_path):
    """A normal (STOP) but empty response must NOT claim a safety block.

    Uses the real FinishReason enum — a string would not exercise the enum
    stringification that the reason-builder must handle.
    """
    from google.genai import types

    from prompt2box.detector import _empty_response_reason

    msg = _empty_response_reason(_response_with_finish(types.FinishReason.STOP))
    assert "SAFETY" not in msg
    assert msg == "Gemini returned an empty response."


def test_detect_refine_accepts_config(make_client, image_path):
    """detect(refine=RefineConfig(...)) applies the custom config."""
    from prompt2box import RefineConfig

    raw = json.dumps(
        [
            {"label": "speck", "box_2d": [0, 0, 2, 2]},
            {"label": "cat", "box_2d": [100, 100, 500, 500]},
        ]
    )
    # min_area_frac high enough to drop the tiny speck.
    result = Detector(client=make_client(raw)).detect(
        image_path, refine=RefineConfig(min_area_frac=0.01)
    )
    assert result.labels == ["cat"]
    assert any("too small" in reason for _, reason in result.dropped)


def test_slice_returns_detection_result(make_client, image_path):
    result = Detector(client=make_client(RESPONSE)).detect(image_path)
    sliced = result[:1]
    assert isinstance(sliced, DetectionResult)
    assert sliced.labels == ["cat"]


def test_detect_sends_response_schema(make_client, image_path):
    """Detection must request structured output (response_schema)."""
    client = make_client(RESPONSE)
    Detector(client=client).detect(image_path)
    cfg = client.calls["config"]
    assert cfg.response_schema is not None


@pytest.mark.parametrize("shape", ["text", "parsed"])
def test_detect_same_result_for_both_response_shapes(make_client, image_path, shape):
    """Detection must yield identical results whether the SDK returns structured
    `.parsed` (production-primary path) or only `.text` (fallback)."""
    if shape == "parsed":
        from prompt2box.detector import _DetectionItem

        client = make_client(
            parsed=[
                _DetectionItem(label="cat", box_2d=[100, 50, 500, 250]),
                _DetectionItem(label="sofa", box_2d=[400, 300, 900, 950]),
            ]
        )
    else:
        client = make_client(RESPONSE)

    result = Detector(client=client).detect(image_path)
    assert result.labels == ["cat", "sofa"]
    assert result[0].box == (20, 20, 100, 100)  # cat on a 400x200 image


def test_close_closes_owned_client(monkeypatch):
    closed = {"n": 0}

    class FakeC:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed["n"] += 1

    import prompt2box.detector as det_mod

    monkeypatch.setattr(det_mod.genai, "Client", FakeC)
    det = Detector(api_key="AIzaFake")
    det.close()
    det.close()  # idempotent
    assert closed["n"] == 2  # called each time, no error


def test_context_manager_closes(monkeypatch):
    closed = {"n": 0}

    class FakeC:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed["n"] += 1

    import prompt2box.detector as det_mod

    monkeypatch.setattr(det_mod.genai, "Client", FakeC)
    with Detector(api_key="AIzaFake"):
        pass
    assert closed["n"] == 1


def test_close_does_not_touch_injected_client(make_client):
    injected = make_client("[]")
    calls = []
    injected.close = lambda: calls.append(1)
    Detector(client=injected).close()
    assert calls == []  # caller owns the injected client's lifecycle


def test_new_client_builds_distinct_clients(monkeypatch):
    """Non-injected detectors must mint a fresh client each call (per-thread safety)."""
    import prompt2box.detector as det_mod

    monkeypatch.setattr(det_mod.genai, "Client", lambda **kw: object())
    det = Detector(api_key="AIzaFake")
    a, b = det._new_client(), det._new_client()
    assert a is not b  # distinct objects -> never shared across threads


def test_new_client_reuses_injected_client(make_client):
    """An injected client is reused as-is (caller owns its thread-safety)."""
    injected = make_client("[]")
    det = Detector(client=injected)
    assert det._new_client() is injected
    assert det._new_client() is injected


def test_vertex_mode_requires_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="project"):
        Detector(vertexai=True)


def test_vertex_mode_builds_client(monkeypatch):
    """vertexai=True should construct a Vertex-configured genai.Client."""
    captured = {}

    import prompt2box.detector as det_mod

    class FakeGenaiClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(det_mod.genai, "Client", FakeGenaiClient)
    Detector(vertexai=True, project="my-proj", location="us-central1")
    assert captured["vertexai"] is True
    assert captured["project"] == "my-proj"
    assert captured["location"] == "us-central1"
    assert "http_options" in captured  # request timeout is configured


class _ApiError(Exception):
    """Mimics google.genai.errors.APIError: carries an int .code."""

    def __init__(self, code, message=""):
        super().__init__(message or f"HTTP {code}")
        self.code = code


def test_retry_recovers_from_transient_error():
    """_generate_with_retry should retry a 429 (by status code) then succeed."""
    attempts = {"n": 0}

    class FlakyClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise _ApiError(429, "Too Many Requests")
                return "ok"

    out = _generate_with_retry(
        FlakyClient,
        model="m",
        contents=[],
        config=None,
        max_retries=2,
        initial_delay=0.0,
    )
    assert out == "ok"
    assert attempts["n"] == 2


def test_retry_uses_status_code_not_string():
    """A 400 (client error) must NOT be retried even if '503' is in its text."""
    from prompt2box.detector import _is_retryable

    assert _is_retryable(_ApiError(429)) is True
    assert _is_retryable(_ApiError(503)) is True
    assert _is_retryable(_ApiError(400, "contains 503 in the message")) is False
    assert _is_retryable(ValueError("random 429 in text")) is False  # no .code, not transport
    assert _is_retryable(OSError("connection reset by peer")) is True  # transport fallback


def test_retry_reraises_non_transient_immediately():
    calls = {"n": 0}

    class BadClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                calls["n"] += 1
                raise ValueError("bad request 400")

    with pytest.raises(ValueError):
        _generate_with_retry(
            BadClient, model="m", contents=[], config=None, max_retries=3, initial_delay=0.0
        )
    assert calls["n"] == 1  # no retries on a non-transient error
