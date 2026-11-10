"""Regression tests for the repository-root Mirror 7 CLI."""
from __future__ import annotations

import builtins

import mirror_cli


class FakeService:
    def __init__(self, **_kwargs):
        self.crystal_vault = None
        self.teach_calls = []
        self.code_calls = []

    def status(self):
        return {"crystal": {"rule_count": 123, "vault_path": "test-vault.dcrystal"}}

    def crystal_coverage_map(self):
        return {
            "active_domains": {"math": {}, "reasoning": {}},
            "telemetry": {"abstention_rate_pct": 12.5},
        }

    def memory_list(self, _session_id):
        return []

    def crystal_teach(self, question, answer, **kwargs):
        self.teach_calls.append((question, answer, kwargs))
        return True

    def tdd_generate_and_verify(self, *, prompt):
        self.code_calls.append(prompt)
        return {
            "ok": True,
            "verified": True,
            "final_code": "print('hello')",
            "verification": {"tests_passed": 2, "tests_generated": 2},
        }

    def crystal_ask(self, prompt, session_id="default"):
        return {
            "answer": f"answer for {prompt}",
            "confidence": 0.9,
            "latency_ms": 1.0,
            "mode": "test",
            "abstained": False,
            "reasoning_trace": [],
        }


def test_interactive_teach_code_and_status_use_real_backend_contract(monkeypatch, capsys):
    service = FakeService()
    monkeypatch.setattr(mirror_cli, "BackendService", lambda **kwargs: service)
    commands = iter([
        "/teach What is a qubit? | A two-state quantum system.",
        "/code print hello",
        "/status",
        "/exit",
    ])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(commands))

    exit_code = mirror_cli.run_interactive("cli-regression")

    output = capsys.readouterr().out
    assert exit_code == 0
    assert service.teach_calls == [
        ("What is a qubit?", "A two-state quantum system.", {"domain": "local_cli_teach"})
    ]
    assert service.code_calls == ["print hello"]
    assert "[TEACH SUCCESS]" in output
    assert "[TDD ENGINE] Verified: True" in output
    assert "Tests Passed: 2/2" in output
    assert "Total Domains:      2" in output
    assert "Abstention Rate:    12.5%" in output


def test_duplicate_teaching_is_reported_as_noop(monkeypatch, capsys):
    service = FakeService()
    service.crystal_teach = lambda *_args, **_kwargs: False
    monkeypatch.setattr(mirror_cli, "BackendService", lambda **kwargs: service)
    commands = iter(["/teach Duplicate? | Existing answer.", "/exit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(commands))

    assert mirror_cli.run_interactive() == 0
    output = capsys.readouterr().out
    assert "[TEACH NO-OP]" in output
    assert "crystallized into Vault" not in output


def test_one_shot_closes_a_mounted_vault(monkeypatch, capsys):
    class ClosableVault:
        closed = False

        def close(self):
            self.closed = True

    service = FakeService()
    service.crystal_vault = ClosableVault()
    monkeypatch.setattr(mirror_cli, "BackendService", lambda **kwargs: service)

    assert mirror_cli.run_oneshot("hello", session_id="test-session") == 0
    assert "answer for hello" in capsys.readouterr().out
    assert service.crystal_vault.closed is True


def test_main_accepts_existing_vault_override(monkeypatch, tmp_path):
    vault = tmp_path / "test.dcrystal"
    vault.write_bytes(b"sqlite fixture")
    monkeypatch.setenv("MIRROR_CRYSTAL_VAULT", "previous-vault.dcrystal")
    called = {}
    monkeypatch.setattr(
        mirror_cli,
        "run_oneshot",
        lambda prompt, session_id: called.update(prompt=prompt, session_id=session_id) or 0,
    )

    assert mirror_cli.main(["--vault", str(vault), "--session", "s1", "hello", "there"]) == 0
    assert called == {"prompt": "hello there", "session_id": "s1"}
    assert __import__("os").environ["MIRROR_CRYSTAL_VAULT"] == "previous-vault.dcrystal"
