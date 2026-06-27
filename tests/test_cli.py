"""Tests for CLI helpers (no API access)."""

from pathlib import Path

from prompt2box.cli import _default_output_path, _load_dotenv


def test_dotenv_loads_missing_keys(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# a comment\nGEMINI_API_KEY=from_dotenv\nQUOTED='val ue'\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    _load_dotenv()

    import os

    assert os.environ["GEMINI_API_KEY"] == "from_dotenv"
    assert os.environ["QUOTED"] == "val ue"


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "real_env_wins")

    _load_dotenv()

    import os

    assert os.environ["GEMINI_API_KEY"] == "real_env_wins"


def test_dotenv_absent_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    _load_dotenv()  # must not raise


def test_dotenv_handles_export_prefix(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("export GEMINI_API_KEY=exp_val\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    _load_dotenv()

    import os

    assert os.environ["GEMINI_API_KEY"] == "exp_val"


def test_dotenv_strips_unquoted_inline_comment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=val123 # my key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    _load_dotenv()

    import os

    assert os.environ["GEMINI_API_KEY"] == "val123"


def test_dotenv_keeps_hash_inside_quotes(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('GEMINI_API_KEY="a#b#c"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    _load_dotenv()

    import os

    assert os.environ["GEMINI_API_KEY"] == "a#b#c"


def test_default_output_keeps_safe_suffix():
    assert _default_output_path(Path("a/photo.jpg")).name == "photo_boxed.jpg"
    assert _default_output_path(Path("a/photo.png")).name == "photo_boxed.png"


def test_default_output_falls_back_to_png_for_unsafe():
    # HEIC/GIF can't always be written by Pillow — annotated copy becomes PNG.
    assert _default_output_path(Path("a/photo.heic")).name == "photo_boxed.png"
    assert _default_output_path(Path("a/clip.gif")).name == "clip_boxed.png"
