"""Tests for detection refinement (no API access needed)."""

import json

from prompt2box import Detector, RefineConfig
from prompt2box.detector import Detection
from prompt2box.refine import refine_detections

IMG = (1000, 1000)  # 1,000,000 px


def _det(label, x0, y0, x1, y1):
    return Detection(label=label, x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def test_drops_whole_image_box():
    objs = [_det("cat", 100, 100, 300, 300), _det("background", 0, 0, 1000, 1000)]
    kept, dropped = refine_detections(objs, IMG)
    assert [d.label for d in kept] == ["cat"]
    assert len(dropped) == 1
    assert "image" in dropped[0][1]


def test_dedups_near_identical_boxes():
    objs = [
        _det("car", 100, 100, 300, 300),
        _det("car", 102, 101, 298, 301),  # ~same box -> duplicate
        _det("tree", 500, 500, 600, 600),
    ]
    kept, dropped = refine_detections(objs, IMG, RefineConfig(dedup_iou=0.9))
    assert sorted(d.label for d in kept) == ["car", "tree"]
    assert len(dropped) == 1
    assert "duplicate" in dropped[0][1]


def test_min_area_drops_tiny_when_enabled():
    objs = [_det("speck", 0, 0, 5, 5), _det("box", 100, 100, 400, 400)]
    kept, dropped = refine_detections(objs, IMG, RefineConfig(min_area_frac=0.001))
    assert [d.label for d in kept] == ["box"]
    assert "too small" in dropped[0][1]


def test_min_area_off_by_default_keeps_small():
    objs = [_det("speck", 0, 0, 5, 5)]
    kept, dropped = refine_detections(objs, IMG)  # min_area_frac default 0
    assert len(kept) == 1 and not dropped


def test_drop_labels():
    objs = [_det("watermark", 0, 0, 50, 50), _det("dog", 100, 100, 400, 400)]
    kept, dropped = refine_detections(objs, IMG, RefineConfig(drop_labels=("watermark",)))
    assert [d.label for d in kept] == ["dog"]


def test_kept_preserves_original_order():
    objs = [_det("a", 0, 0, 100, 100), _det("b", 200, 200, 300, 300), _det("c", 400, 400, 500, 500)]
    kept, _ = refine_detections(objs, IMG)
    assert [d.label for d in kept] == ["a", "b", "c"]


def test_result_refine_method_does_not_mutate_original(make_client, image_path):
    # 400x200 image; the "frame" box covers the whole image -> dropped by refine.
    raw = json.dumps(
        [
            {"label": "frame", "box_2d": [0, 0, 1000, 1000]},
            {"label": "cat", "box_2d": [100, 100, 500, 500]},
        ]
    )
    result = Detector(client=make_client(raw)).detect(image_path)
    refined = result.refine()

    assert len(result) == 2  # original untouched
    assert refined.labels == ["cat"]
    assert len(refined.dropped) == 1
    assert refined.dropped[0][0].label == "frame"


def test_detect_refine_flag(make_client, image_path):
    raw = json.dumps(
        [
            {"label": "whole scene", "box_2d": [0, 0, 1000, 1000]},
            {"label": "lamp", "box_2d": [100, 100, 300, 300]},
        ]
    )
    result = Detector(client=make_client(raw)).detect(image_path, refine=True)
    assert result.labels == ["lamp"]
    assert len(result.dropped) == 1


def test_refine_result_still_serializes(make_client, image_path):
    raw = json.dumps([{"label": "background", "box_2d": [0, 0, 1000, 1000]}])
    result = Detector(client=make_client(raw)).detect(image_path, refine=True)
    # All dropped, none kept — JSON is an empty array, no crash.
    assert json.loads(result.to_json()) == []
