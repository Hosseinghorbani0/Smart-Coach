from pathlib import Path

from core.config import Config


def test_storage_paths_are_created_under_project_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SMART_COACH_BASE_DIR", str(tmp_path))
    monkeypatch.delenv("SMART_COACH_OPENAI_API_KEY", raising=False)

    project_root = Path(tmp_path)
    voice_dir = project_root / "storage" / "voice"
    chat_dir = project_root / "storage" / "chat"

    Config.ensure_storage_dirs(project_root)

    assert voice_dir.exists()
    assert chat_dir.exists()


def test_openai_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SMART_COACH_OPENAI_API_KEY", "test-key")

    assert Config.get_openai_api_key() == "test-key"
