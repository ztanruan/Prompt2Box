"""Command-line entry point for Prompt2Box."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from prompt2box import __version__
from prompt2box.detector import DEFAULT_MODEL, Detector

# Raster formats Pillow can reliably *write* an annotated copy to. For inputs
# outside this set (HEIC, GIF, …) we fall back to PNG so saving never fails.
_SAVE_SAFE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env in the working directory, if present.

    Zero-dependency and non-overriding: real environment variables win, so this
    only fills in what's missing. Keeps the shipped .env.example honest.
    """
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):  # tolerate `export KEY=value`
                line = line[len("export ") :]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value and value[0] in "'\"":  # quoted: take the quoted span verbatim
                quote = value[0]
                end = value.find(quote, 1)
                value = value[1:end] if end != -1 else value[1:]
            else:  # unquoted: strip a trailing inline comment
                value = value.split(" #", 1)[0].strip()
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass  # unreadable .env is not fatal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt2box",
        description="Detect bounding boxes of items in a local image using Gemini.",
    )
    parser.add_argument("image", help="Path to a local image file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save the annotated image (default: <image>_boxed.<ext>). "
        "Use --no-image to skip drawing.",
    )
    parser.add_argument("--no-image", action="store_true", help="Do not write an annotated image")
    parser.add_argument("-j", "--json-out", help="Also write the detections JSON to this path")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, help=f"Gemini model id (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-p", "--prompt", help='Free-form instruction, e.g. "only detect the animals"'
    )
    parser.add_argument(
        "-c",
        "--classes",
        nargs="+",
        metavar="LABEL",
        help="Restrict detection to these labels, e.g. -c cat dog",
    )
    parser.add_argument(
        "--api-key", help="Gemini API key (overrides GEMINI_API_KEY / GOOGLE_API_KEY)"
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        help="Use Vertex AI (no API key; auth via `gcloud auth application-default login`)",
    )
    parser.add_argument("--project", help="GCP project id for --vertex (or GOOGLE_CLOUD_PROJECT)")
    parser.add_argument("--location", help="Vertex AI location for --vertex (default: global)")
    parser.add_argument(
        "--max-size",
        type=int,
        default=1536,
        metavar="PX",
        help="Downscale images larger than this (longest edge) before upload (default: 1536)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Drop whole-image catch-all boxes and duplicate detections",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging to stderr")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _default_output_path(image: Path) -> Path:
    # Keep the input's extension only if Pillow can write it; else use PNG so the
    # annotated copy always saves (e.g. HEIC/GIF inputs).
    suffix = image.suffix if image.suffix.lower() in _SAVE_SAFE_SUFFIXES else ".png"
    return image.with_name(f"{image.stem}_boxed{suffix}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    _load_dotenv()
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not args.vertex and not api_key:
        print(
            "error: no API key found. Set GEMINI_API_KEY (or pass --api-key),\n"
            "or use --vertex for Vertex AI auth.\n"
            "Get a key at https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return 2

    image = Path(args.image).expanduser()
    try:
        if args.vertex:
            detector = Detector(
                model=args.model,
                vertexai=True,
                project=args.project,
                location=args.location,
                max_image_size=args.max_size,
            )
        else:
            detector = Detector(api_key=api_key, model=args.model, max_image_size=args.max_size)
        result = detector.detect(
            image, prompt=args.prompt, classes=args.classes, refine=args.refine
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — surface any provider/parse error to the user
        print(f"error: detection failed: {exc}", file=sys.stderr)
        return 1

    json_text = result.to_json()
    print(json_text)

    if args.json_out:
        Path(args.json_out).expanduser().write_text(json_text, encoding="utf-8")
        print(f"saved JSON -> {args.json_out}", file=sys.stderr)

    if not args.no_image:
        out_path = Path(args.output).expanduser() if args.output else _default_output_path(image)
        written = result.save_annotated(out_path)
        print(f"saved annotated image -> {written}", file=sys.stderr)

    print(f"detected {len(result)} item(s)", file=sys.stderr)
    for det, reason in result.dropped:
        print(f"  dropped {det.label!r}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
