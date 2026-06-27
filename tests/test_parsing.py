"""Unit tests for JSON parsing and 0-1000 -> pixel conversion (no client)."""

import pytest

from prompt2box.detector import _parse_json, _to_detection


def test_parse_plain_array():
    assert _parse_json('[{"label":"a","box_2d":[0,0,10,10]}]') == [
        {"label": "a", "box_2d": [0, 0, 10, 10]}
    ]


def test_parse_strips_markdown_fence():
    text = '```json\n[{"label":"a","box_2d":[1,2,3,4]}]\n```'
    assert _parse_json(text) == [{"label": "a", "box_2d": [1, 2, 3, 4]}]


def test_parse_unwraps_object_with_list():
    assert _parse_json('{"objects": [{"label":"a","box_2d":[0,0,1,1]}]}') == [
        {"label": "a", "box_2d": [0, 0, 1, 1]}
    ]


def test_parse_extracts_array_amid_prose():
    text = 'Here you go:\n[{"label":"a","box_2d":[0,0,1,1]}]\nHope that helps!'
    assert _parse_json(text) == [{"label": "a", "box_2d": [0, 0, 1, 1]}]


def test_parse_raises_on_garbage():
    with pytest.raises(ValueError):
        _parse_json("not json at all")


def test_conversion_normalized_to_pixels():
    # ymin=100,xmin=50,ymax=500,xmax=250 on a 400x200 image.
    d = _to_detection({"label": "x", "box_2d": [100, 50, 500, 250]}, 400, 200)
    assert (d.x_min, d.y_min, d.x_max, d.y_max) == (20, 20, 100, 100)
    assert d.box_normalized == (100, 50, 500, 250)


def test_conversion_clamps_and_orders():
    # Out-of-range and reversed coords should clamp to the image and reorder.
    d = _to_detection({"label": "x", "box_2d": [1200, -50, 800, 900]}, 400, 200)
    assert 0 <= d.x_min <= d.x_max <= 400
    assert 0 <= d.y_min <= d.y_max <= 200


def test_conversion_accepts_alternate_keys():
    d = _to_detection({"name": "cat", "bbox": [0, 0, 1000, 1000]}, 400, 200)
    assert d.label == "cat"
    assert d.box == (0, 0, 400, 200)


def test_conversion_rejects_malformed_box():
    assert _to_detection({"label": "x", "box_2d": [1, 2, 3]}, 400, 200) is None
    assert _to_detection({"label": "x", "box_2d": "nope"}, 400, 200) is None
