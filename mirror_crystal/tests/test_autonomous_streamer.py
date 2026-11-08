"""
Unit tests for AutonomousDatasetStreamer:
Pulse execution, lifecycle controls, hygiene filtering, and batch vault ingestion.
"""
from pathlib import Path
import tempfile
import time

from mirror_crystal.disk_vault import DiskCrystalVault
from mirror_crystal.autonomous_streamer import AutonomousDatasetStreamer


def test_streamer_pulse_and_flush():
    with tempfile.TemporaryDirectory() as td:
        vpath = Path(td) / "test_stream_vault.dcrystal"
        vault = DiskCrystalVault(vpath)
        try:
            streamer = AutonomousDatasetStreamer(vault, cadence_seconds=1.0, flush_batch_size=2)

            # Execute pulse
            res = streamer.pulse_once()
            assert res["evaluated"] >= 1
            assert res["admitted"] >= 1

            # Second pulse to trigger flush (batch_size=2)
            streamer.pulse_once()
            streamer._flush_buffer_locked()

            assert vault.rule_count() >= 1

            status = streamer.get_status()
            assert status["total_evaluated"] >= 2
            assert status["total_admitted"] >= 2
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
