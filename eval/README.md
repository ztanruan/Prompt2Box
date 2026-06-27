# Evaluation

A small harness to measure how good Prompt2Box's boxes actually are — because for
a detector, "how accurate?" is the question that matters, and prose limitations
aren't evidence.

## Run it

```bash
# Needs API credentials (Developer API key or --vertex)
python eval/iou_eval.py eval/sample/manifest.example.json --vertex
```

Output:

```
  ../../docs/example2.jpg: tp=2 fp=4 fn=1 meanIoU=0.78
=== SUMMARY ===
  precision @ 0.5: 0.33
  recall    @ 0.5: 0.67
  mean IoU (matched):    0.78
  tp=2 fp=4 fn=1
```

## Manifest format

JSON with pixel-space ground-truth boxes (`x_min, y_min, x_max, y_max`):

```json
{
  "images": [
    {"path": "img.jpg", "objects": [{"label": "laptop", "box": [470, 80, 905, 610]}]}
  ]
}
```

The bundled `sample/manifest.example.json` uses **approximate placeholder boxes**,
so its numbers are illustrative only. For a real measurement, supply your own
ground truth.

## Building a real manifest from COCO

1. Grab a slice of [COCO val2017](https://cocodataset.org/) (images + `instances_val2017.json`).
2. For each image, convert each annotation's `bbox` (COCO is `[x, y, w, h]`) to
   `[x, y, x + w, y + h]` and map `category_id` → category `name`.
3. Emit the manifest shape above.

Then run the harness and put the resulting precision/recall/mean-IoU in the
README so users have a real number, not a promise.

## How matching works

Per image, each ground-truth box is greedily matched to the highest-IoU
prediction whose label matches (case-insensitive substring, either direction),
provided IoU ≥ threshold (default 0.5). Matched → true positive; unmatched
predictions → false positives; unmatched ground truth → false negatives.

The matching math (`box_iou`, `match_and_score`) is unit-tested in
`tests/test_eval.py` and needs no API access.
