# Contributing to Prompt2Box

Thanks for your interest! This is a small, focused library — a thin, ergonomic
wrapper around Gemini's spatial-understanding mode. Contributions that keep it
small and sharp are very welcome.

## Dev setup

```bash
git clone https://github.com/ztanruan/Prompt2Box
cd Prompt2Box
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a PR

```bash
pre-commit install  # optional, but it runs the two below automatically
ruff check .        # lint (must pass)
pytest -q           # tests (must pass — no API key needed, Gemini is mocked)
```

CI runs both on Python 3.10–3.12. Please add a test for any behavior change;
the suite mocks Gemini via `Detector(client=...)`, so tests stay fast and
offline.

## Scope

Good fits: better parsing robustness, output formats, drawing options, CLI
ergonomics, docs. Out of scope: turning this into a general agent framework, or
adding heavyweight non-optional dependencies. If in doubt, open an issue first.

## Reporting bugs

Include the command/code you ran, the full error, your `google-genai` version
(`pip show google-genai`), and whether you're on the Developer API or Vertex AI.
Never paste an API key into an issue.
