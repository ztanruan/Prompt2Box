# Examples

- **`detect.py`** — minimal end-to-end script: detect, print labels + boxes, save
  an annotated image.

```bash
# Developer API
export GEMINI_API_KEY=AIza...
python examples/detect.py some_photo.jpg

# Vertex AI
export GOOGLE_CLOUD_PROJECT=your-project
python examples/detect.py some_photo.jpg --vertex

# Constrain what is detected
python examples/detect.py street.jpg -- only the vehicles
```
