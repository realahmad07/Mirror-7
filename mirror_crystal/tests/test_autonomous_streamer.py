"""
Unit tests for AutonomousDatasetStreamer:
Pulse execution, lifecycle controls, hygiene filtering, and batch vault ingestion.
"""
from pathlib import Path
import tempfile
import time

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.autonomous_streamer import AutonomousDatasetStreamer


def test_streamer_pulse_and_flush(monkeypatch):
    from types import SimpleNamespace
    from mirror_crystal.autonomous_streamer import StreamCandidate

    candidates = [
        StreamCandidate(
            source="unit_test_stream",
            domain="test_domain",
            prompt="Explain deterministic test candidate alpha.",
            response="Deterministic candidate alpha has a stable, bounded explanation.",
            spdx_license="Apache-2.0",
        ),
        StreamCandidate(
            source="unit_test_stream",
            domain="test_domain",
            prompt="Explain deterministic test candidate beta.",
            response="Deterministic candidate beta has a separate, stable explanation.",
            spdx_license="Apache-2.0",
        ),
    ]
    # Make the stream-source selection deterministic. This test covers buffering
    # and flush accounting, not random choice among external candidate generators.
    monkeypatch.setattr(
        "mirror_crystal.autonomous_streamer.random.choice",
        lambda _generators: (lambda: candidates),
    )
    with tempfile.TemporaryDirectory() as td:
        vpath = Path(td) / "test_stream_vault.dcrystal"
        vault = DiskCrystalVault(vpath)
        try:
            streamer = AutonomousDatasetStreamer(vault, cadence_seconds=1.0, flush_batch_size=2)
            monkeypatch.setattr(
                streamer.hygiene,
                "audit_record",
                lambda *_args, **_kwargs: SimpleNamespace(passed=True, rejection_gate=None),
            )

            # Execute pulse
            res = streamer.pulse_once()
            assert res["evaluated"] == 2
            assert res["admitted"] == 2

            # The first pulse reaches the deterministic batch size; a second pulse
            # exercises repeated flush behavior without depending on live randomness.
            second = streamer.pulse_once()
            assert second["evaluated"] == 2
            assert second["admitted"] == 2
            streamer._flush_buffer_locked()

            assert vault.rule_count() >= 1

            status = streamer.get_status()
            assert status["total_evaluated"] == 4
            assert status["total_admitted"] == 4
            assert status["vault_total_rules"] == vault.rule_count()
        finally:
            vault.close()


def test_streamer_thread_lifecycle():
    with tempfile.TemporaryDirectory() as td:
        vpath = Path(td) / "test_stream_vault.dcrystal"
        vault = DiskCrystalVault(vpath)
        try:
            streamer = AutonomousDatasetStreamer(vault, cadence_seconds=0.2, flush_batch_size=2)

            # 1. Start daemon
            ok_start = streamer.start()
            assert ok_start is True
            assert streamer.get_status()["is_running"] is True

            # Let it run 2 pulses
            time.sleep(0.5)

            # 2. Pause
            streamer.pause()
            time.sleep(0.1)

            # 3. Resume
            streamer.resume()
            time.sleep(0.1)

            # 4. Stop
            ok_stop = streamer.stop()
            assert ok_stop is True
            assert streamer.get_status()["is_running"] is False

            assert vault.rule_count() >= 1
        finally:
            vault.close()


def test_streamer_nonblocking_status():
    import threading
    with tempfile.TemporaryDirectory() as td:
        vpath = Path(td) / "test_stream_vault.dcrystal"
        vault = DiskCrystalVault(vpath)
        try:
            streamer = AutonomousDatasetStreamer(vault, cadence_seconds=0.05, flush_batch_size=2)
            streamer.start()
            results = []
            def reader():
                for _ in range(20):
                    t0 = time.perf_counter()
                    st = streamer.get_status()
                    elapsed = time.perf_counter() - t0
                    results.append((st["is_running"], elapsed))
                    time.sleep(0.005)

            threads = [threading.Thread(target=reader) for _ in range(4)]
            for t in threads: t.start()
            for t in threads: t.join()

            assert len(results) == 80
            max_latency = max(r[1] for r in results)
            assert max_latency < 0.1, f"Status call took too long: {max_latency}s"
            streamer.stop()
        finally:
            vault.close()


def test_streamer_deduplication_and_live_fetcher():
    from mirror_crystal.autonomous_streamer import StreamDeduplicationStore, ArxivLiveApiFetcher
    with tempfile.TemporaryDirectory() as td:
        vpath = Path(td) / "test_stream_vault.dcrystal"
        dedup_path = Path(td) / "test_dedup.sqlite"
        vault = DiskCrystalVault(vpath)
        try:
            # 1. Deduplication store verification
            store = StreamDeduplicationStore(dedup_path)
            assert not store.is_seen("paper_sample_alpha")
            store.mark_seen("paper_sample_alpha", "unit_test")
            assert store.is_seen("paper_sample_alpha")
            assert store.count() == 1
            store.close()

            # 2. Live fetcher with fallback verification
            cands = ArxivLiveApiFetcher.fetch_live_candidates(max_results=1, timeout=3.0)
            assert len(cands) >= 1
            cand = cands[0]
            assert "arXiv" in cand.prompt or "research" in cand.prompt or len(cand.prompt) > 10
            assert cand.spdx_license in ("CC-BY-4.0", "Apache-2.0")

            # 3. Streamer with live fetch enabled
            streamer = AutonomousDatasetStreamer(
                vault,
                cadence_seconds=0.5,
                flush_batch_size=1,
                enable_live_fetch=True,
                dedup_db_path=dedup_path,
            )
            res = streamer.pulse_once()
            assert res["evaluated"] >= 1
            assert res["admitted"] >= 1
            streamer.close()
        finally:
            vault.close()
