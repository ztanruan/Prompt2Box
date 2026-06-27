"""Golden test: a real Gemini response (captured from a live run) must parse
into the exact pixel boxes we expect.

This guards against regressions in parsing / coordinate conversion if the
response shape we rely on ever changes. The fixture is the verbatim JSON Gemini
returned for the sample pug photo (900x1200).
"""

import json
from pathlib import Path

from prompt2box import Detector

FIXTURE = Path(__file__).parent / "fixtures" / "gemini_response.json"


def test_golden_response_parses_to_expected_boxes(make_client, tmp_path):
    from PIL import Image

    # Recreate the exact source dimensions from the live capture (900x1200).
    img = tmp_path / "pug.png"
    Image.new("RGB", (900, 1200), (200, 200, 200)).save(img)

    raw = FIXTURE.read_text()
    result = Detector(client=make_client(raw)).detect(img)

    assert result.labels == ["pug", "scarf"]
    # pug: ymin=250,xmin=0,ymax=1000,xmax=1000 on 900x1200 -> (0,300,900,1200)
    pug = result[0]
    assert pug.box == (0, 300, 900, 1200)
    scarf = result[1]
    assert scarf.box == (0, 720, 900, 1200)

    # Round-trip the serialized form too.
    payload = json.loads(result.to_json())
    assert {o["label"] for o in payload} == {"pug", "scarf"}
    assert "confidence" not in payload[0]  # dead field was removed
