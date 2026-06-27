"""Core object-detection logic for Prompt2Box.

Asks a Gemini model to return bounding boxes for the items in an image.
Gemini emits boxes in normalized ``[ymin, xmin, ymax, xmax]`` coordinates on a
0-1000 scale; this module converts them to absolute pixel coordinates and wraps
them in ergonomic result objects (``Detection`` / ``DetectionResult``).

Public surface:
    detect(image, ...)        -> DetectionResult   # one-shot convenience
    Detector(api_key, model)  -> reusable client wrapper
    Detection                 -> a single box + label, with crop()/box helpers
    DetectionResult           -> list-like container with save_annotated()/to_json()
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from google import genai
from google.genai import types
from pydantic import BaseModel

if TYPE_CHECKING:
    from prompt2box.refine import RefineConfig

logger = logging.getLogger(__name__)

# HTTP status codes worth retrying. For the Gemini API a 500 is a transient
# internal error (worth retrying), not a malformed-request signal — those are
# 4xx and surface immediately.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Cap how long a single request may hang (MILLISECONDS — confirmed via the SDK's
# HttpOptions.timeout field). Detection is a few seconds in practice; this just
# prevents an indefinite stall on a dead socket.
_REQUEST_TIMEOUT_MS = 120_000

DEFAULT_MODEL = "gemini-2.5-flash"


def _http_options() -> types.HttpOptions:
    """A fresh HttpOptions per client (never a shared, possibly-mutated singleton)."""
    return types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS)


class _DetectionItem(BaseModel):
    """Schema handed to Gemini as ``response_schema`` so the model is *forced* to
    emit conformant JSON (label + 0-1000 box). Output is read from
    ``response.parsed`` when present; ``_parse_json`` remains a text backstop."""

    label: str
    box_2d: list[int]


# Structure is enforced by ``response_schema`` + ``response_mime_type`` below, so
# this prompt only carries the *semantics* the schema can't express: the box
# coordinate order, one-entry-per-item, and label disambiguation.
_SYSTEM_PROMPT = """You are an object-detection engine. Detect every distinct, prominent item in the image.

For each item provide:
- label: a short lowercase name (e.g. "coffee mug").
- box_2d: [ymin, xmin, ymax, xmax] as integers normalized to 0-1000, where
  [0,0] is the top-left and [1000,1000] is the bottom-right.

Rules:
- One entry per distinct physical item; do not merge separate items.
- If the same kind of item appears multiple times, return one entry each and
  disambiguate the label (e.g. "person (left)", "person (right)").
- Never return masks or segmentation polygons - bounding boxes only.
- Return at most 25 objects; if there are more, keep the most prominent.
"""

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    """A single detected item with a pixel-space bounding box.

    Immutable value object. Box coordinates are absolute pixels in the source
    image, origin at the top-left. ``box_normalized`` keeps Gemini's 0-1000
    ``(ymin, xmin, ymax, xmax)`` values (ordered) for reference.
    """

    label: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    box_normalized: tuple[int, int, int, int] | tuple[()] = ()

    @property
    def box(self) -> tuple[int, int, int, int]:
        """``(x_min, y_min, x_max, y_max)`` — the order Pillow's crop() wants."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    def crop(self, image_path: str | Path, output_path: str | Path) -> Path:
        """Crop this box out of ``image_path`` and save it to ``output_path``."""
        from PIL import Image

        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(Path(image_path).expanduser()) as img:
            img.crop(self.box).save(out)
        return out

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "box_normalized": list(self.box_normalized),  # JSON-friendly
        }


@dataclass
class DetectionResult(Sequence):
    """List-like container of detections for one image.

    Supports iteration, ``len()``, and indexing, plus helpers to serialize
    (``to_json``) and to render an annotated copy of the source image
    (``save_annotated`` / ``annotated_image``).
    """

    image_path: Path
    image_size: tuple[int, int]  # (width, height)
    objects: list[Detection] = field(default_factory=list)
    # Detections removed by refine(), each paired with the reason. Empty unless
    # this result is the output of refine() / detect(refine=True).
    dropped: list[tuple[Detection, str]] = field(default_factory=list)

    # --- Sequence protocol so it behaves like a list of Detection ---
    def __iter__(self) -> Iterator[Detection]:
        return iter(self.objects)

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, index):
        # A slice returns a DetectionResult (list-like), an int returns a Detection.
        if isinstance(index, slice):
            return DetectionResult(self.image_path, self.image_size, self.objects[index])
        return self.objects[index]

    @property
    def labels(self) -> list[str]:
        return [d.label for d in self.objects]

    def filter(self, label: str) -> DetectionResult:
        """Return a new result keeping only detections whose label contains ``label``."""
        needle = label.lower()
        kept = [d for d in self.objects if needle in d.label.lower()]
        return DetectionResult(self.image_path, self.image_size, kept)

    def refine(self, config=None, **overrides) -> DetectionResult:
        """Return a new result with low-quality detections removed.

        Drops whole-image catch-all boxes, near-duplicate boxes, and (optionally)
        tiny or blocklisted ones. The removed detections and their reasons are
        available on the returned result's ``.dropped``. The original result is
        left unchanged.

        Pass a :class:`~prompt2box.refine.RefineConfig`, or keyword overrides
        (``max_area_frac=``, ``dedup_iou=``, ``min_area_frac=``, ``drop_labels=``).
        """
        from prompt2box.refine import RefineConfig, refine_detections

        cfg = config or RefineConfig(**overrides)
        kept, dropped = refine_detections(self.objects, self.image_size, cfg)
        return DetectionResult(self.image_path, self.image_size, kept, dropped)

    def to_dict(self) -> dict:
        return {
            "image": str(self.image_path),
            "width": self.image_size[0],
            "height": self.image_size[1],
            "objects": [d.to_dict() for d in self.objects],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps([d.to_dict() for d in self.objects], indent=indent)

    def annotated_image(self):
        """Return a PIL.Image copy of the source with boxes + labels drawn."""
        from prompt2box.draw import render_detections

        return render_detections(self.image_path, self.objects)

    def save_annotated(self, output_path: str | Path) -> Path:
        """Draw boxes + labels and save the annotated image to ``output_path``."""
        from prompt2box.draw import draw_detections

        return draw_detections(self.image_path, self.objects, output_path)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class Detector:
    """Reusable Gemini-backed object detector.

    Construct once (it holds a configured ``genai.Client``) and call
    :meth:`detect` many times.

    Two auth backends are supported:

    * **Developer API** (default) — a Gemini API key. Falls back to
      ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` from the environment.
    * **Vertex AI** (``vertexai=True``) — no API key; authenticates via
      Application Default Credentials (run ``gcloud auth application-default
      login`` once). Use this if your account only issues ``AQ.`` keys, which
      are currently rejected by the Developer API.

    Args:
        api_key: Gemini Developer API key (Developer-API mode only).
        model: Gemini model id (default ``gemini-2.5-flash``).
        client: Optional pre-built ``genai.Client`` (overrides everything else).
        vertexai: Use the Vertex AI backend instead of the Developer API.
        project: GCP project id for Vertex AI. Falls back to
            ``GOOGLE_CLOUD_PROJECT``.
        location: Vertex AI location (default ``global``). Falls back to
            ``GOOGLE_CLOUD_LOCATION``.
        max_image_size: Downscale images whose longest edge exceeds this (px)
            before upload, to cut cost/latency (0 disables). Boxes are
            normalized, so this does not affect accuracy.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: genai.Client | None = None,
        *,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        max_image_size: int = 1536,
    ) -> None:
        if max_image_size < 0:
            raise ValueError(f"max_image_size must be >= 0 (got {max_image_size})")
        self.model = model
        if "gemini" not in model.lower():
            # box_2d spatial detection is a Gemini-trained capability. Other Vertex
            # Model Garden models (Claude, Llama, …) won't return usable boxes, and
            # they use different SDKs entirely — so warn rather than silently fail.
            logger.warning(
                "Model %r is not a Gemini model. Bounding-box detection (box_2d) is "
                "Gemini-specific; non-Gemini models won't return usable boxes.",
                model,
            )
        # Images larger than this (longest edge, px) are downscaled before upload
        # to cut cost/latency. Boxes are normalized 0-1000, so accuracy is
        # unaffected — we still map back to the original pixel dimensions.
        self.max_image_size = max_image_size

        # We keep a factory so detect_batch() can build a fresh client per worker
        # thread (never sharing the SDK client across threads). When the caller
        # injects their own client we reuse it and assume they own thread-safety
        # (and its lifecycle — close() never touches an injected client).
        self._client_factory: Callable[[], genai.Client]
        self._owns_client = client is None
        if client is not None:
            self._client_factory = lambda: client
        elif vertexai:
            project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
            if not project:
                raise ValueError(
                    "Vertex AI mode needs a project. Pass project=... or set "
                    "GOOGLE_CLOUD_PROJECT, and run `gcloud auth application-default login`."
                )
            # Uses Application Default Credentials — no API key involved.
            self._client_factory = lambda: genai.Client(
                vertexai=True, project=project, location=location, http_options=_http_options()
            )
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key and key.startswith("AQ."):
                # AI Studio's newer "AQ." auth keys are currently rejected by the
                # Developer API (401 ACCESS_TOKEN_TYPE_UNSUPPORTED). Warn early so
                # the user isn't left guessing — point them at the working paths.
                logger.warning(
                    "This looks like an 'AQ.' auth key, which the Gemini Developer API "
                    "currently rejects. Use an 'AIza...' key, or switch to Vertex AI "
                    "(vertexai=True / --vertex)."
                )
            # genai.Client also reads the env itself, but we pass the key
            # explicitly when present so callers can override the environment.
            self._client_factory = lambda: genai.Client(api_key=key, http_options=_http_options())
        self._client = self._client_factory()

    def _new_client(self) -> genai.Client:
        """Build a fresh client (used per worker thread); reuses an injected one."""
        return self._client_factory()

    def close(self) -> None:
        """Close the primary client's connection pool. Idempotent.

        No-op for a caller-injected client (the caller owns its lifecycle).
        """
        if self._owns_client:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    def __enter__(self) -> Detector:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def detect(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        classes: Sequence[str] | None = None,
        refine: bool | RefineConfig = False,
    ) -> DetectionResult:
        """Detect items in ``image_path``.

        Args:
            image_path: Path to a local image file.
            prompt: Free-form instruction overriding the default
                ("detect everything"), e.g. ``"only the animals"``.
            classes: Restrict detection to these labels, e.g. ``["cat", "dog"]``.
                Ignored when ``prompt`` is given.
            refine: ``True`` applies :meth:`DetectionResult.refine` with defaults;
                pass a :class:`~prompt2box.refine.RefineConfig` to customize.
                Removed items land in the result's ``.dropped``.

        Returns:
            A :class:`DetectionResult`.
        """
        return self._detect(self._client, image_path, prompt=prompt, classes=classes, refine=refine)

    def _detect(
        self,
        client: genai.Client,
        image_path: str | Path,
        *,
        prompt: str | None,
        classes: Sequence[str] | None,
        refine: bool | RefineConfig,
    ) -> DetectionResult:
        """Core detection against a specific client (so threads can use their own)."""
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        # Loads the image (registering HEIC support if needed), downscales large
        # images, and returns the bytes to upload plus the ORIGINAL dimensions.
        image_bytes, mime_type, width, height = _prepare_image(path, self.max_image_size)

        user_text = "Detect all prominent items and return their bounding boxes."
        if prompt:
            user_text = prompt.strip()
        elif classes:
            wanted = ", ".join(classes)
            user_text = f"Detect only these items and return their bounding boxes: {wanted}."

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            types.Part.from_text(text=user_text),
        ]

        logger.info("Detecting in %s (%dx%d) with %s", path.name, width, height, self.model)
        start = time.monotonic()
        try:
            response = _generate_with_retry(
                client,
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=list[_DetectionItem],
                ),
            )
        except Exception as exc:
            friendly = _friendly_auth_error(exc)
            if friendly is not None:
                raise friendly from exc
            raise
        logger.info("Detection completed in %.1fs", time.monotonic() - start)

        items = _extract_items(response)
        objects = [
            d for d in (_to_detection(i, width, height) for i in items if isinstance(i, dict)) if d
        ]
        logger.info("Parsed %d detection(s)", len(objects))
        result = DetectionResult(image_path=path, image_size=(width, height), objects=objects)
        if refine:
            cfg = refine if not isinstance(refine, bool) else None
            result = result.refine(cfg)
            logger.info("Refined to %d (dropped %d)", len(result), len(result.dropped))
        return result

    def detect_batch(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str | None = None,
        classes: Sequence[str] | None = None,
        refine: bool | RefineConfig = False,
        max_workers: int = 4,
    ) -> list[DetectionResult | None]:
        """Detect over many images concurrently (Gemini calls are I/O-bound).

        Results are returned in the same order as ``image_paths``. An image that
        fails (bad file, API error) yields ``None`` in its slot rather than
        aborting the whole batch — check for it before using a result.

        Each worker thread uses its own ``genai.Client`` (built from this
        detector's config) so the SDK client is never shared across threads.
        """
        from concurrent.futures import ThreadPoolExecutor

        paths = list(image_paths)
        if not paths:
            return []

        # One client per worker thread, lazily created and reused within it.
        thread_local = threading.local()
        created: list[genai.Client] = []
        created_lock = threading.Lock()

        def _client_for_thread() -> genai.Client:
            client = getattr(thread_local, "client", None)
            if client is None:
                client = self._new_client()
                thread_local.client = client
                with created_lock:
                    created.append(client)
            return client

        def _one(p: str | Path) -> DetectionResult | None:
            try:
                return self._detect(
                    _client_for_thread(), p, prompt=prompt, classes=classes, refine=refine
                )
            except Exception as exc:  # noqa: BLE001 — isolate per-image failures
                logger.warning("Detection failed for %s: %s", p, exc)
                return None

        try:
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(paths)))) as pool:
                return list(pool.map(_one, paths))
        finally:
            # Release per-thread clients' connections. Never close a caller-injected
            # client (an injected client's factory returns the shared self._client).
            for c in created:
                if c is not self._client:
                    try:
                        c.close()
                    except Exception:  # noqa: BLE001 — best-effort cleanup
                        pass


def detect(
    image_path: str | Path,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    classes: Sequence[str] | None = None,
    client: genai.Client | None = None,
    vertexai: bool = False,
    project: str | None = None,
    location: str | None = None,
    max_image_size: int = 1536,
    refine: bool | RefineConfig = False,
) -> DetectionResult:
    """One-shot convenience wrapper around :class:`Detector`.

    >>> from prompt2box import detect
    >>> result = detect("photo.jpg")              # Developer API (GEMINI_API_KEY)
    >>> result = detect("photo.jpg", vertexai=True, project="my-gcp-project")
    >>> result = detect("photo.jpg", refine=True) # drop whole-image/duplicate boxes
    >>> result.labels
    ['cat', 'sofa', 'lamp']
    >>> result.save_annotated("out.jpg")
    """
    with Detector(
        api_key=api_key,
        model=model,
        client=client,
        vertexai=vertexai,
        project=project,
        location=location,
        max_image_size=max_image_size,
    ) as detector:
        return detector.detect(image_path, prompt=prompt, classes=classes, refine=refine)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _friendly_auth_error(exc: Exception) -> Exception | None:
    """Return a friendlier auth error, or ``None`` to re-raise the original as-is.

    Detection here is intentionally substring-based: there is no typed/status-code
    signal that distinguishes "AI Studio AQ. key rejected" from other 401s, so we
    match the server's message text. (Retries, by contrast, key off status codes.)
    """
    text = str(exc)
    if "ACCESS_TOKEN_TYPE_UNSUPPORTED" in text or "API keys are not supported" in text:
        return RuntimeError(
            "Authentication failed. If you used an AI Studio 'AQ.' key, the Gemini "
            "Developer API currently rejects it — use an 'AIza...' key, or switch to "
            "Vertex AI (vertexai=True / --vertex after `gcloud auth application-default "
            f"login`).\nOriginal error: {text}"
        )
    return None


def _empty_response_reason(response) -> str:
    """Build an informative message when the model returns no usable text.

    Surfaces safety blocks and truncation (``finish_reason`` /
    ``prompt_feedback``) instead of a bare "empty response".
    """
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        return f"Gemini blocked the request (block_reason={block_reason})."

    for cand in getattr(response, "candidates", None) or []:
        finish = getattr(cand, "finish_reason", None)
        if finish is None:
            continue
        # finish_reason is an enum (str(...) is "FinishReason.SAFETY"); compare on
        # .name so a normal STOP is correctly recognized and not flagged.
        name = getattr(finish, "name", str(finish)).upper()
        if name not in ("STOP", "FINISH_REASON_STOP"):
            return (
                f"Gemini returned no text (finish_reason={name}). "
                "Common causes: SAFETY block, RECITATION, or MAX_TOKENS."
            )
    return "Gemini returned an empty response."


def _guess_mime_type(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "image/jpeg")


_heif_lock = threading.Lock()
_heif_registered = False


def _register_heif(path: Path) -> None:
    """Enable Pillow to open HEIC/HEIF, with a clear error if support is missing.

    Registration mutates global Pillow state, so it's done once behind a lock —
    safe to call concurrently from detect_batch() worker threads.
    """
    global _heif_registered
    if path.suffix.lower() not in (".heic", ".heif") or _heif_registered:
        return
    with _heif_lock:
        if _heif_registered:
            return
        try:
            import pillow_heif
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {path.suffix} images needs the optional 'pillow-heif' package. "
                "Install it with: pip install 'prompt2box[heic]'"
            ) from exc
        pillow_heif.register_heif_opener()
        _heif_registered = True


def _prepare_image(path: Path, max_size: int) -> tuple[bytes, str, int, int]:
    """Return (upload_bytes, mime_type, original_width, original_height).

    Downscales images whose longest edge exceeds ``max_size`` (re-encoding as
    JPEG) to cut upload cost/latency. The returned width/height are always the
    *original* dimensions, since Gemini's boxes are normalized 0-1000 and we map
    them back to the full-resolution image.
    """
    from PIL import Image

    _register_heif(path)
    with Image.open(path) as img:
        width, height = img.size
        if max_size and max(width, height) > max_size:
            import io

            # Keep PNG lossless (screenshots/diagrams with text/alpha); use JPEG
            # for photographic formats where it's much smaller at no visible cost.
            keep_png = path.suffix.lower() == ".png"
            # Only carry an alpha channel if the source actually has one.
            has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            mode = "RGBA" if (keep_png and has_alpha) else "RGB"
            scaled = img.convert(mode)
            scaled.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            if keep_png:
                scaled.save(buf, format="PNG", optimize=True)
                out_mime = "image/png"
            else:
                scaled.convert("RGB").save(buf, format="JPEG", quality=90)
                out_mime = "image/jpeg"
            logger.info(
                "Downscaled %s from %dx%d to %dx%d for upload",
                path.name,
                width,
                height,
                *scaled.size,
            )
            return buf.getvalue(), out_mime, width, height

    # Small enough — upload the original bytes untouched.
    return path.read_bytes(), _guess_mime_type(path), width, height


def _response_text(response) -> str:
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    # Fall back to concatenating candidate parts when .text is empty.
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                text += part.text
    return text.strip()


def _extract_items(response) -> list:
    """Get the list of raw detection items from a response.

    Prefers the SDK's schema-validated ``response.parsed`` (populated because we
    set ``response_schema``); falls back to parsing ``response.text``. Raises an
    informative ValueError when the model returned nothing usable.
    """
    parsed = getattr(response, "parsed", None)
    if parsed:
        items = [
            p.model_dump() if isinstance(p, BaseModel) else p
            for p in parsed
            if isinstance(p, (BaseModel, dict))
        ]
        if items:
            return items

    text = _response_text(response)
    if not text:
        raise ValueError(_empty_response_reason(response))
    return _parse_json(text)


def _parse_json(text: str) -> list:
    """Parse the model response into a list, tolerating markdown fences/prose."""
    text = text.strip()
    candidates = [text]

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    array = re.search(r"\[[\s\S]*\]", text)
    if array:
        candidates.append(array.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            # Some responses wrap the list, e.g. {"objects": [...]}.
            for value in data.values():
                if isinstance(value, list):
                    return value
            continue
        if isinstance(data, list):
            return data

    raise ValueError(f"Could not parse Gemini response as JSON: {text[:500]}")


def _to_detection(item: dict, width: int, height: int) -> Detection | None:
    """Convert one raw Gemini item into a pixel-space Detection."""
    box = item.get("box_2d") or item.get("box") or item.get("bbox")
    label = str(item.get("label") or item.get("name") or "object").strip()
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        logger.warning("Skipping item with malformed box: %s", item)
        return None

    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box)
    except (TypeError, ValueError):
        logger.warning("Skipping item with non-numeric box: %s", item)
        return None

    # Normalized 0-1000 -> absolute pixels.
    x0 = round(xmin / 1000.0 * width)
    y0 = round(ymin / 1000.0 * height)
    x1 = round(xmax / 1000.0 * width)
    y1 = round(ymax / 1000.0 * height)

    # Clamp and order so x0<=x1, y0<=y1 and stay inside the image.
    x0, x1 = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
    y0, y1 = sorted((max(0, min(y0, height)), max(0, min(y1, height))))

    # Store box_normalized ordered the same way (ymin<=ymax, xmin<=xmax) so it
    # stays consistent with the pixel box if a consumer rescales it.
    nymin, nymax = sorted((ymin, ymax))
    nxmin, nxmax = sorted((xmin, xmax))

    return Detection(
        label=label,
        x_min=x0,
        y_min=y0,
        x_max=x1,
        y_max=y1,
        box_normalized=(
            int(round(nymin)),
            int(round(nxmin)),
            int(round(nymax)),
            int(round(nxmax)),
        ),
    )


def _is_retryable(exc: Exception) -> bool:
    """True if ``exc`` is a transient error worth retrying.

    Primary signal is the HTTP status code: ``genai.errors.APIError`` exposes
    ``.code`` (and any exception carrying an int ``.code`` is treated the same).
    Only when there's no status code do we fall back to a narrow substring check
    for transport-layer failures (timeouts / dropped connections) — deliberately
    not matching bare numbers like "503" that could appear coincidentally.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in _RETRYABLE_STATUS
    text = str(exc).lower()
    return any(
        tok in text
        for tok in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "connection error",
        )
    )


def _generate_with_retry(client, *, model, contents, config, max_retries=3, initial_delay=2.0):
    """Call Gemini with exponential backoff on rate-limit/transient errors."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:  # noqa: BLE001 — SDK raises various provider errors
            if _is_retryable(exc) and attempt < max_retries:
                logger.warning(
                    "Gemini call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    str(exc)[:150],
                )
                time.sleep(delay)
                delay *= 2.0
                continue
            raise
