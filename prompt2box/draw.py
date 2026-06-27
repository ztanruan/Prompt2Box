"""Draw detected bounding boxes and labels onto an image."""

from __future__ import annotations

from pathlib import Path

from prompt2box.detector import Detection, _register_heif

# A small, visually distinct palette cycled across detections.
_PALETTE = [
    (255, 56, 56),
    (56, 255, 56),
    (56, 56, 255),
    (255, 178, 29),
    (255, 56, 255),
    (29, 178, 255),
    (255, 112, 31),
    (133, 56, 255),
]


def render_detections(image_path: str | Path, detections: list[Detection]):
    """Return a PIL.Image copy of ``image_path`` with boxes + labels drawn."""
    from PIL import Image, ImageDraw

    path = Path(image_path).expanduser()
    _register_heif(path)  # so HEIC sources open even outside the detect() flow
    with Image.open(path) as opened:
        img = opened.convert("RGB")  # detaches from the file handle
    draw = ImageDraw.Draw(img)

    # Scale line/font with image size so boxes stay readable on large images.
    line_width = max(2, round(min(img.size) / 300))
    font = _load_font(max(14, round(min(img.size) / 45)))

    for i, det in enumerate(detections):
        color = _PALETTE[i % len(_PALETTE)]
        draw.rectangle(
            [det.x_min, det.y_min, det.x_max, det.y_max], outline=color, width=line_width
        )
        _draw_label(draw, det.label, det.x_min, det.y_min, color, font)

    return img


def draw_detections(
    image_path: str | Path,
    detections: list[Detection],
    output_path: str | Path,
) -> Path:
    """Render boxes + labels and save to ``output_path``. Returns the path written."""
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    render_detections(image_path, detections).save(out)
    return out


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    # No system TTF found — keep labels legible by sizing the default font
    # (Pillow >= 10.1 accepts a size; older Pillow ignores it).
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_label(draw, text: str, x: int, y: int, color, font) -> None:
    """Draw a filled label chip with the text just above the box top edge."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
    except AttributeError:  # very old Pillow
        tw, th = draw.textsize(text, font=font)

    pad = 2
    chip_top = max(0, y - th - 2 * pad)
    draw.rectangle([x, chip_top, x + tw + 2 * pad, chip_top + th + 2 * pad], fill=color)
    draw.text((x + pad, chip_top + pad), text, fill=(255, 255, 255), font=font)
