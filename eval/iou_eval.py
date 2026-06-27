"""Rough IoU evaluation harness for Prompt2Box.

Answers the only question that matters for a detector: *how good are the boxes?*

Give it a manifest of images + ground-truth boxes (pixel xyxy), and it runs
detection, greedily matches predictions to ground truth by label + IoU, and
reports precision / recall / mean-IoU at a threshold.

    python eval/iou_eval.py eval/sample/manifest.example.json --vertex

Manifest format (JSON):
    {
      "images": [
        {
          "path": "docs/example2.jpg",
          "objects": [
            {"label": "laptop", "box": [480, 67, 900, 600]},
            {"label": "smartphone", "box": [580, 434, 780, 500]}
          ]
        }
      ]
    }

NOTE: results are only as meaningful as your ground truth. Build a real manifest
from a labeled set (e.g. convert a COCO slice — see eval/README.md). Gemini boxes
are approximate, so expect modest IoU; this harness is for *measuring* that, not
for pretending it's tight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Single source of truth for IoU — shared with detection-time dedup.
from prompt2box.refine import box_iou


def _labels_match(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    return a in b or b in a


def match_and_score(
    predictions: list[dict], ground_truth: list[dict], iou_threshold: float = 0.5
) -> dict:
    """Greedily match predictions to ground truth for one image.

    Each item is a dict with ``label`` and ``box`` (x_min, y_min, x_max, y_max).
    Returns tp/fp/fn counts and the IoU of each matched pair.
    """
    used_preds: set[int] = set()
    matched_ious: list[float] = []
    tp = 0

    for gt in ground_truth:
        best_i, best_iou = -1, 0.0
        for i, pred in enumerate(predictions):
            if i in used_preds or not _labels_match(pred["label"], gt["label"]):
                continue
            iou = box_iou(tuple(pred["box"]), tuple(gt["box"]))
            if iou > best_iou:
                best_i, best_iou = i, iou
        if best_i >= 0 and best_iou >= iou_threshold:
            used_preds.add(best_i)
            matched_ious.append(best_iou)
            tp += 1

    fp = len(predictions) - len(used_preds)
    fn = len(ground_truth) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "ious": matched_ious}


def evaluate_manifest(
    manifest_path: str | Path, iou_threshold: float = 0.5, **detect_kwargs
) -> dict:
    """Run detection over a manifest and aggregate scores. Needs API credentials."""
    from prompt2box import detect

    manifest = json.loads(Path(manifest_path).read_text())
    base = Path(manifest_path).parent
    total = {"tp": 0, "fp": 0, "fn": 0, "ious": []}

    for entry in manifest["images"]:
        img_path = (
            (base / entry["path"]) if not Path(entry["path"]).is_absolute() else Path(entry["path"])
        )
        # Manifest paths may also be relative to the repo root.
        if not img_path.is_file() and Path(entry["path"]).is_file():
            img_path = Path(entry["path"])

        result = detect(str(img_path), **detect_kwargs)
        preds = [{"label": d.label, "box": list(d.box)} for d in result]
        score = match_and_score(preds, entry["objects"], iou_threshold)

        total["tp"] += score["tp"]
        total["fp"] += score["fp"]
        total["fn"] += score["fn"]
        total["ious"] += score["ious"]
        print(
            f"  {entry['path']}: tp={score['tp']} fp={score['fp']} fn={score['fn']} "
            f"meanIoU={_mean(score['ious']):.2f}"
        )

    precision = total["tp"] / (total["tp"] + total["fp"]) if (total["tp"] + total["fp"]) else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if (total["tp"] + total["fn"]) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": _mean(total["ious"]),
        "tp": total["tp"],
        "fp": total["fp"],
        "fn": total["fn"],
        "iou_threshold": iou_threshold,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Rough IoU eval for Prompt2Box.")
    p.add_argument("manifest", help="Path to a JSON manifest (see module docstring)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU match threshold (default 0.5)")
    p.add_argument("--vertex", action="store_true", help="Use Vertex AI auth")
    p.add_argument("-m", "--model", default=None, help="Gemini model id")
    args = p.parse_args(argv)

    kwargs: dict = {}
    if args.vertex:
        kwargs["vertexai"] = True
    if args.model:
        kwargs["model"] = args.model

    print(f"Evaluating {args.manifest} @ IoU>={args.iou} ...")
    summary = evaluate_manifest(args.manifest, iou_threshold=args.iou, **kwargs)
    print("\n=== SUMMARY ===")
    print(f"  precision @ {summary['iou_threshold']}: {summary['precision']:.2f}")
    print(f"  recall    @ {summary['iou_threshold']}: {summary['recall']:.2f}")
    print(f"  mean IoU (matched):    {summary['mean_iou']:.2f}")
    print(f"  tp={summary['tp']} fp={summary['fp']} fn={summary['fn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
