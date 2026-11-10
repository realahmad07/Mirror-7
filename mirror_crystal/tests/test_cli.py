"""Regression tests for the Mirror Crystal command-line interface."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirror_crystal import cli


class FakeEngine:
    instances = []

    def __init__(self):
        self.calls = []
        FakeEngine.instances.append(self)

    @classmethod
    def load(cls, _path):
        return cls()

    def ingest_jsonl(self, path, max_rows=None, fold_synonyms=True):
        self.calls.append(("jsonl", Path(path).suffix, max_rows, fold_synonyms))
        return {"file": Path(path).name, "rows_ingested": 1}

    def ingest_text_file(self, path, fold_synonyms=True):
        self.calls.append(("text", Path(path).suffix, fold_synonyms))
        return {"file": Path(path).name, "lines_read": 1}

    def save(self, path):
        Path(path).write_bytes(b"fake-mcrystal")
        return len(b"fake-mcrystal")


class FakeDiskVault:
    instance = None

    def __init__(self, path):
        self.path = Path(path)
        self.calls = []
        self.closed = False
        self.stats = {"pairs_seen": 2}
        FakeDiskVault.instance = self

    def ingest_jsonl(self, path, max_rows=None, fold_synonyms=True):
        self.calls.append(("jsonl", Path(path).suffix, max_rows, fold_synonyms))
        return {"file": Path(path).name, "rows_ingested": 1}

    def ingest_source_file(self, path):
        self.calls.append(("source", Path(path).suffix))
        return {"file": Path(path).name, "records": 1}

    def optimize_disk(self):
        return 1024

    def rule_count(self):
        return 2

    def close(self):
        self.closed = True


def test_train_directory_ingests_supported_files_recursively(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "a.jsonl").write_text(json.dumps({"prompt": "Q", "answer": "A"}) + "\n")
    (nested / "b.md").write_text("# note\n")
    (source / "c.py").write_text("print('ok')\n")
    (source / "ignored.csv").write_text("a,b\n")

    FakeEngine.instances.clear()
    monkeypatch.setattr(cli, "MirrorCrystalEngine", FakeEngine)
    exit_code = cli.main(["train", "--input", str(source), "--save", str(tmp_path / "out.mcrystal")])

    assert exit_code == 0
    engine = FakeEngine.instances[-1]
    assert [call[0] for call in engine.calls] == ["jsonl", "text", "text"]
    assert (tmp_path / "out.mcrystal").is_file()
    summary = json.loads(capsys.readouterr().out)
    assert len(summary["files_ingested"]) == 3


def test_train_disk_vault_closes_and_uses_document_ingestion(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "notes.txt").write_text("Mirror note\n")
    save = tmp_path / "vault.dcrystal"
    monkeypatch.setattr(cli, "DiskCrystalVault", FakeDiskVault)

    assert cli.main(["train", "--input", str(source), "--save", str(save)]) == 0
    assert FakeDiskVault.instance.closed is True
    assert FakeDiskVault.instance.calls == [("source", ".txt")]


def test_train_missing_input_fails_with_cli_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "train",
            "--input",
            str(tmp_path / "missing"),
            "--save",
            str(tmp_path / "out.mcrystal"),
        ])
    assert exc.value.code == 2


def test_query_disk_brain_closes_after_response_error(monkeypatch, tmp_path):
    brain = tmp_path / "brain.dcrystal"
    brain.write_bytes(b"sqlite-placeholder")

    class BrokenVault(FakeDiskVault):
        def respond(self, _prompt):
            raise RuntimeError("expected response failure")

    monkeypatch.setattr(cli, "DiskCrystalVault", BrokenVault)
    with pytest.raises(RuntimeError, match="expected response failure"):
        cli.main(["query", "--brain", str(brain), "--prompt", "test"])
    assert BrokenVault.instance.closed is True


def test_benchmark_returns_failure_when_required_check_fails(monkeypatch, capsys):
    report = {
        "unseen_number_slot_generalization_exact": True,
        "continual_100_step_zero_forgetting": False,
        "multi_hop_lock_chaining_verified": True,
        "multi_turn_session_reality_check_verified": True,
    }
    monkeypatch.setattr(cli, "run_hardware_benchmark", lambda **_kwargs: report)
    assert cli.main(["benchmark"]) == 1
    assert json.loads(capsys.readouterr().out) == report
