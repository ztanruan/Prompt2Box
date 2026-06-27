"""Detection-quality refinement.

Raw LLM detection output has predictable failure modes that the model won't fix
for you: whole-image "catch-all" boxes (a box that's basically the entire
frame), the same object returned two or three times, and tiny specks. This module
applies cheap, deterministic post-processing to drop those — and tells you *why*
each detection was removed, so nothing disappears silently.

    from prompt2box import detect
    clean = detect("photo.jpg").refine()
    clean.dropped        # [(Detection(label='background', ...), 'covers 98% of image'), ...]
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt2box.detector import Detection


@dataclass
class RefineConfig:
    """Thresholds for :func:`refine_detections`.

    Attributes:
        min_area_frac: Drop boxes whose area is below this fraction of the image
            (0 disables — the default, so legitimate small objects are kept).
        max_area_frac: Drop boxes whose area exceeds this fraction of the image —
            these are almost always whole-scene "catch-all" detections.
        dedup_iou: Merge near-duplicate boxes with the same/related label whose
            IoU is at or above this value (keeps the larger one).
        drop_labels: Labels to always remove (case-insensitive substring match),
            e.g. ``("background", "image", "scene")``.
    """

    min_area_frac: float = 0.0
    max_area_frac: float = 0.92
    dedup_iou: float = 0.9
    drop_labels: tuple[str, ...] = ()


def box_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union of two ``(x_min, y_min, x_max, y_max)`` boxes.

    The single IoU implementation for the project — the eval harness imports it
    too, so detection-time dedup and offline scoring can never diverge.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _labels_related(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return a in b or b in a


def refine_detections(
    objects: list[Detection],
    image_size: tuple[int, int],
    config: RefineConfig | None = None,
) -> tuple[list[Detection], list[tuple[Detection, str]]]:
    """Return ``(kept, dropped)`` where ``dropped`` pairs each removed detection
    with a human-readable reason."""
    cfg = config or RefineConfig()
    width, height = image_size
    image_area = max(1, width * height)

    kept: list[Detection] = []
    dropped: list[tuple[Detection, str]] = []

    # Pass 1: per-box label and area filters.
    survivors: list[Detection] = []
    for det in objects:
        label_lc = det.label.lower()
        if any(bad.lower() in label_lc for bad in cfg.drop_labels):
            dropped.append((det, f"label matches drop list ({det.label!r})"))
            continue

        frac = det.area / image_area
        if cfg.min_area_frac and frac < cfg.min_area_frac:
            dropped.append((det, f"too small ({frac:.2%} of image)"))
            continue
        if cfg.max_area_frac and frac > cfg.max_area_frac:
            dropped.append((det, f"covers {frac:.0%} of image (likely whole-scene)"))
            continue
        survivors.append(det)

    # Pass 2: dedup near-identical boxes. Keep the larger of an overlapping pair.
    for det in sorted(survivors, key=lambda d: d.area, reverse=True):
        duplicate_of = next(
            (
                k
                for k in kept
                if _labels_related(det.label, k.label) and box_iou(det.box, k.box) >= cfg.dedup_iou
            ),
            None,
        )
        if duplicate_of is not None:
            dropped.append((det, f"duplicate of {duplicate_of.label!r} (IoU≥{cfg.dedup_iou})"))
        else:
            kept.append(det)

    # Preserve the model's original ordering for the kept set.
    order = {id(d): i for i, d in enumerate(objects)}
    kept.sort(key=lambda d: order.get(id(d), 0))
    return kept, dropped
