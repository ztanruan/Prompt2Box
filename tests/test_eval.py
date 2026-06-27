"""Unit tests for the eval harness math (no API access needed)."""

import importlib.util
from pathlib import Path

# eval/ is a dev tool, not part of the installed package — load it by path.
_spec = importlib.util.spec_from_file_location(
    "iou_eval", Path(__file__).resolve().parent.parent / "eval" / "iou_eval.py"
)
iou_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iou_eval)


def test_iou_identical_boxes_is_one():
    assert iou_eval.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou_eval.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150 -> 1/3.
    assert abs(iou_eval.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9


def test_match_counts_tp_fp_fn():
    preds = [
        {"label": "cat", "box": [0, 0, 10, 10]},  # matches gt cat
        {"label": "dog", "box": [100, 100, 110, 110]},  # no gt -> fp
    ]
    gt = [
        {"label": "cat", "box": [0, 0, 10, 10]},  # matched -> tp
        {"label": "bird", "box": [50, 50, 60, 60]},  # unmatched -> fn
    ]
    score = iou_eval.match_and_score(preds, gt, iou_threshold=0.5)
    assert score["tp"] == 1
    assert score["fp"] == 1
    assert score["fn"] == 1
    assert score["ious"] == [1.0]


def test_match_respects_threshold():
    preds = [{"label": "cat", "box": [0, 0, 10, 10]}]
    gt = [{"label": "cat", "box": [5, 0, 15, 10]}]  # IoU = 1/3 < 0.5
    score = iou_eval.match_and_score(preds, gt, iou_threshold=0.5)
    assert score["tp"] == 0 and score["fp"] == 1 and score["fn"] == 1


def test_label_substring_match():
    preds = [{"label": "person (left)", "box": [0, 0, 10, 10]}]
    gt = [{"label": "person", "box": [0, 0, 10, 10]}]
    score = iou_eval.match_and_score(preds, gt, iou_threshold=0.5)
    assert score["tp"] == 1
