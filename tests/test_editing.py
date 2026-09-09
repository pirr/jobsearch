from pathlib import Path


def test_open_in_editor_writes_current_text_and_returns_edited_result(monkeypatch):
    from jobsearch.editing import open_in_editor

    monkeypatch.setenv("EDITOR", "fake-editor")
    seen = {}

    def fake_run(cmd, check=True):
        assert cmd[0] == "fake-editor"
        path = Path(cmd[1])
        assert path.read_text(encoding="utf-8") == "original text"
        path.write_text("edited text", encoding="utf-8")
        seen["path"] = path

    result = open_in_editor("original text", run=fake_run)

    assert result == "edited text"
    assert not seen["path"].exists()  # tempfile cleaned up


def test_open_in_editor_defaults_to_vi(monkeypatch):
    from jobsearch.editing import open_in_editor

    monkeypatch.delenv("EDITOR", raising=False)
    seen = {}

    def fake_run(cmd, check=True):
        seen["editor"] = cmd[0]
        Path(cmd[1]).write_text("x", encoding="utf-8")

    open_in_editor("", run=fake_run)
    assert seen["editor"] == "vi"


def test_open_in_editor_strips_trailing_whitespace(monkeypatch):
    from jobsearch.editing import open_in_editor

    monkeypatch.setenv("EDITOR", "fake-editor")

    def fake_run(cmd, check=True):
        Path(cmd[1]).write_text("edited text\n\n", encoding="utf-8")

    assert open_in_editor("x", run=fake_run) == "edited text"
