"""Minimal Prompt2Box example.

Developer API:
    export GEMINI_API_KEY=AIza...
    python examples/detect.py path/to/image.jpg

Vertex AI (no key; run `gcloud auth application-default login` first):
    export GOOGLE_CLOUD_PROJECT=your-project
    python examples/detect.py path/to/image.jpg --vertex
"""

from __future__ import annotations

import sys

from prompt2box import detect


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python examples/detect.py <image> [--vertex] [-- 'only the animals']")
        return 2

    image = argv[0]
    kwargs: dict = {}
    if "--vertex" in argv:
        kwargs["vertexai"] = True  # picks up GOOGLE_CLOUD_PROJECT
    # Anything after a literal `--` becomes a free-form prompt.
    if "--" in argv:
        kwargs["prompt"] = " ".join(argv[argv.index("--") + 1 :]) or None

    result = detect(image, **kwargs)

    print(f"Detected {len(result)} item(s): {', '.join(result.labels) or '(none)'}")
    for d in result:
        print(f"  {d.label:<20} {d.box}")

    out = result.save_annotated("example_out.jpg")
    print(f"Annotated image -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
