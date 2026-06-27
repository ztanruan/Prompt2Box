# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `detect()` one-shot API and reusable `Detector` class.
- `Detection` / `DetectionResult` result objects (crop, filter, to_json,
  save_annotated, list-like access).
- Two auth backends: Gemini Developer API key and Vertex AI (`--vertex`).
- `Detector.detect_batch()` for concurrent multi-image detection.
- CLI: `prompt2box` with `--classes`, `--prompt`, `--vertex`, `--json-out`.
- Friendly error guidance when an AI Studio `AQ.` auth key is used (currently
  rejected by the Developer API).
- Mocked test suite + golden-response test; ruff lint; CI matrix.
- Dev tooling: pre-commit (ruff + ruff-format), `examples/detect.py`,
  and `CITATION.cff`.
- Automatic downscaling of large images before upload (`max_image_size` /
  `--max-size`) — cuts cost/latency without affecting accuracy.
- HEIC/HEIF support via the optional `[heic]` extra (`pillow-heif`), with a
  clear error when it's missing.
- IoU evaluation harness in `eval/` (precision / recall / mean-IoU) plus
  unit tests, and an opt-in real-API integration test.
- `refine()` / `detect(refine=...)` / `RefineConfig`: deterministically drop
  whole-image catch-all boxes, near-duplicates, and (optionally) tiny/blocklisted
  ones, recording why each was dropped in `.dropped`.
- CLI `.env` loading (current directory; real env vars take precedence).
- Any Gemini vision model works via `model=` / `-m`; a non-Gemini model id warns
  (box_2d detection is Gemini-specific).

### Hardening (production-readiness pass)
- Structured output: detection sends a `response_schema` and reads the SDK's
  schema-validated `response.parsed` (the text parser stays as a backstop).
- `Detector` is a context manager with `close()`; the one-shot `detect()` closes
  its client automatically, so no connection pool leaks.
- Retries now key off the SDK's typed HTTP status code instead of string-matching
  the error message (no false positives on a coincidental "503").
- Safety blocks / truncation surface a real reason (`finish_reason`,
  `block_reason`) instead of a bare "empty response".
- `detect_batch()` builds a `genai.Client` per worker thread and closes them
  afterward (no shared client, no leaked connections).
- A per-request timeout is configured (no indefinite stalls).
- `Detection` is now an immutable (frozen) value object; `box_normalized` is an
  ordered 4-tuple.
- Large PNGs stay lossless when downscaled (JPEG only for photographic formats,
  alpha kept only when present).
- Annotated-output and HEIC handling no longer crash on `.heic`/`.gif` inputs;
  HEIF registration is thread-safe.
- CLI `.env` parsing tolerates `export ` and inline comments.
- Typed (`py.typed`) and type-checked (mypy in CI); CI also enforces
  `ruff format` and runs on Python 3.10–3.13.

[Unreleased]: https://github.com/ztanruan/Prompt2Box/commits/main
