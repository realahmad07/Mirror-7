"""Stage 12: Production Readiness Test Suite.

Validates all six pillars of Stage 12:
1. Stable Inference: Execution timeouts, concurrency limits (Semaphore), and graceful fallback.
2. API Reliability: Circuit Breaker state machine (Closed/Open/Half-Open), Token-Bucket Rate Limiter,
   and live HTTP health probes (/api/livez, /api/readyz, /api/healthz, /metrics, X-Request-ID).
3. Structured Logging: JSON formatted events, trace/request correlation, credential scrubbing, and audit trails.
4. Monitoring & Telemetry: Prometheus metric exposition (counters, gauges, latency summaries).
5. Deployment Architecture: Strict configuration validation (ProductionConfig), and GracefulShutdownManager lifecycle.
6. Disaster Recovery & Self-Healing: Vault SQLite quick_check integrity, 368,270 rules invariant, auto-reconnection,
   and online hot backup verification.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import urllib.request
import urllib.error
import pytest

from mirror_crystal.production_readiness import (
    ProductionConfig,
    StructuredLogger,
    MetricsCollector,
    CircuitBreakerState,
    CircuitBreakerOpenError,
    CircuitBreaker,
    TokenBucketRateLimiter,
    InferenceCapacityExceededError,
    InferenceResult,
    InferenceStabilizer,
    VaultSelfHealingManager,
    GracefulShutdownManager,
)
from mirror7_backend.http_server import create_server
from mirror7_backend.service import BackendService

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAULT_PATH = REPO_ROOT / "restorevalutecrytaled" / "mirror_crystal_all_datasets.dcrystal"


# ============================================================================
# Section 1: Stable Inference (Concurrency, Deadlines & Graceful Fallback)
# ============================================================================

def test_stage12_stable_inference_concurrency_and_deadlines():
    """Verifies that InferenceStabilizer enforces strict concurrency limits and timeouts."""
    logger = StructuredLogger(name="test_inference", level="DEBUG")
    metrics = MetricsCollector()
    stabilizer = InferenceStabilizer(
        max_concurrent=2,
        timeout_seconds=0.25,
        logger=logger,
        metrics=metrics,
    )

    # 1. Successful fast inference
    def fast_solver(q: str) -> str:
        return f"Processed: {q}"

    res = stabilizer.execute_stable_inference("What is 2+2?", fast_solver)
    assert res.ok is True
    assert res.answer == "Processed: What is 2+2?"
    assert res.fallback_applied is False
    assert res.latency_ms > 0
    assert len(res.trace_id) == 16

    # 2. Timeout protection with graceful fallback
    def slow_solver(q: str) -> str:
        time.sleep(0.60)
        return "Slow response"

    def fallback_solver(q: str) -> str:
        return "I do not have sufficient authoritative data to answer within timeout."

    res_timeout = stabilizer.execute_stable_inference(
        "Complex slow query",
        slow_solver,
        fallback_solver=fallback_solver,
    )
    assert res_timeout.ok is True
    assert res_timeout.fallback_applied is True
    assert "sufficient authoritative data" in res_timeout.answer
    assert res_timeout.engine_used == "fallback_honest_abstention"

    # 3. Concurrency capacity rejection
    stabilizer_c = InferenceStabilizer(
        max_concurrent=2,
        timeout_seconds=5.0,
        logger=logger,
        metrics=metrics,
    )
    slots_occupied = threading.Semaphore(0)
    unblock = threading.Event()

    def blocking_solver(q: str) -> str:
        slots_occupied.release()
        unblock.wait(timeout=5.0)
        return "Done"

    threads = []
    # Start 2 threads filling max_concurrent=2 slots
    for i in range(2):
        t = threading.Thread(
            target=lambda name=f"block_{i}": stabilizer_c.execute_stable_inference(name, blocking_solver),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Ensure both worker threads have entered blocking_solver
    assert slots_occupied.acquire(timeout=2.0)
    assert slots_occupied.acquire(timeout=2.0)

    # 3rd request must be rejected immediately with InferenceCapacityExceededError
    with pytest.raises(InferenceCapacityExceededError) as exc_info:
        stabilizer_c.execute_stable_inference("overflow_query", fast_solver)
    assert "capacity" in str(exc_info.value).lower()

    unblock.set()
    for t in threads:
        t.join(timeout=2.0)
    stabilizer_c.shutdown()
    stabilizer.shutdown()


# ============================================================================
# Section 2: API Reliability (Circuit Breaker, Rate Limiting & Health Probes)
# ============================================================================

def test_stage12_circuit_breaker_lifecycle():
    """Verifies Closed -> Open -> Half-Open -> Closed state transitions."""
    cb = CircuitBreaker(
        name="test_cb",
        failure_threshold=3,
        recovery_timeout=0.20,
        half_open_probes=2,
    )

    assert cb.state == CircuitBreakerState.CLOSED

    def failing_call():
        raise RuntimeError("Transient database error")

    # Trip breaker after 3 failures
    for i in range(3):
        with pytest.raises(RuntimeError):
            cb.execute(failing_call)

    assert cb.state == CircuitBreakerState.OPEN

    # Breaker is OPEN: immediate fail-fast without invoking target
    called = False
    def should_not_run():
        nonlocal called
        called = True
        return "ok"

    with pytest.raises(CircuitBreakerOpenError):
        cb.execute(should_not_run)
    assert called is False

    # Wait for recovery timeout to transition to HALF_OPEN
    time.sleep(0.25)
    assert cb.allow_request() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # 2 successful probes in HALF_OPEN should close the breaker
    assert cb.execute(lambda: "probe_1") == "probe_1"
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert cb.execute(lambda: "probe_2") == "probe_2"
    assert cb.state == CircuitBreakerState.CLOSED


def test_stage12_token_bucket_rate_limiter():
    """Verifies token bucket rate limiting and idle client cleanup."""
    limiter = TokenBucketRateLimiter(rps=10.0, burst=5, idle_ttl_seconds=0.10)

    # Burst of 5 should all succeed
    for _ in range(5):
        allowed, retry_after = limiter.acquire("client_A")
        assert allowed is True
        assert retry_after == 0.0

    # 6th request exceeds burst capacity
    allowed, retry_after = limiter.acquire("client_A")
    assert allowed is False
    assert retry_after > 0.0

    # Independent client_B has its own fresh burst bucket
    allowed_b, _ = limiter.acquire("client_B")
    assert allowed_b is True

    # Idle cleanup
    time.sleep(0.15)
    pruned = limiter.prune_idle()
    assert pruned >= 1


def test_stage12_live_http_health_probes_and_trace_headers():
    """Verifies /api/livez, /api/readyz, /api/healthz, /metrics, and X-Request-ID headers."""
    service = BackendService()
    server = create_server(host="127.0.0.1", port=0, service=service)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        # 1. /api/livez (Liveness probe)
        with urllib.request.urlopen(f"{base_url}/api/livez", timeout=3) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["ok"] is True
            assert data["status"] == "alive"
            # Trace correlation header
            assert "X-Request-ID" in resp.headers

        # 2. /api/readyz (Readiness probe)
        with urllib.request.urlopen(f"{base_url}/api/readyz", timeout=3) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["ok"] is True
            assert data["status"] == "ready"

        # 3. /api/healthz (Comprehensive health check)
        with urllib.request.urlopen(f"{base_url}/api/healthz", timeout=3) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["ok"] is True
            assert data["service"] == "mirror7"
            assert "uptime_seconds" in data
            assert "sessions" in data

        # 4. /metrics (Prometheus exposition format)
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=3) as resp:
            assert resp.status == 200
            assert "text/plain" in resp.headers.get("Content-Type", "")
            metrics_text = resp.read().decode("utf-8")
            assert "mirror_uptime_seconds" in metrics_text
            assert "mirror_vault_rules_count" in metrics_text

        # 5. Client-supplied X-Request-ID propagation
        client_req_id = "test-custom-trace-uuid-999"
        req = urllib.request.Request(
            f"{base_url}/api/livez",
            headers={"X-Request-ID": client_req_id},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.headers.get("X-Request-ID") == client_req_id

    finally:
        server.server_close()


# ============================================================================
# Section 3: Structured Logging & Audit Trails
# ============================================================================

def test_stage12_structured_logging_and_credential_sanitization():
    """Verifies JSON structured logging, trace propagation, and secret redaction."""
    logger = StructuredLogger(name="mirror_audit", level="DEBUG")
    logger.clear_buffer()

    # Log with sensitive Bearer token and password
    entry = logger.info(
        "User authenticated with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token and password='supersecret123'",
        trace_id="trace_abc123",
        request_id="req_xyz789",
        duration_ms=42.5,
    )
    assert entry["level"] == "INFO"
    assert entry["trace_id"] == "trace_abc123"
    assert entry["duration_ms"] == 42.5
    assert "[REDACTED]" in entry["message"]
    assert "eyJhbGciOiJIUzI1Ni" not in entry["message"]
    assert "supersecret123" not in entry["message"]

    # Audit event
    audit_entry = logger.audit(
        action="VAULT_RECOVERY_ATTEMPT",
        actor="system_health_daemon",
        status="SUCCESS",
        details={"restored_rules": 368270},
    )
    assert "AUDIT_EVENT" in audit_entry["message"]
    assert audit_entry["extra"]["audit"]["action"] == "VAULT_RECOVERY_ATTEMPT"
    assert audit_entry["extra"]["audit"]["details"]["restored_rules"] == 368270


# ============================================================================
# Section 4: Monitoring & Telemetry (Prometheus Metric Collector)
# ============================================================================

def test_stage12_prometheus_metrics_collector():
    """Verifies counter increments, gauge updates, and histogram percentile summaries."""
    collector = MetricsCollector()

    collector.increment_counter("http_requests_total", 10)
    collector.increment_counter("inference_requests_total", 5)
    collector.set_gauge("active_sessions", 4.0)
    collector.set_gauge("vault_rules_count", 368270.0)

    # Record latency observations
    for d in [0.005, 0.010, 0.015, 0.020, 0.050]:
        collector.record_duration("inference_duration_seconds", d)

    prom_text = collector.render_prometheus_text()
    assert "mirror_http_requests_total 10" in prom_text
    assert "mirror_inference_requests_total 5" in prom_text
    assert "mirror_active_sessions 4.0" in prom_text
    assert "mirror_vault_rules_count 368270.0" in prom_text
    assert 'mirror_inference_duration_seconds{quantile="0.5"}' in prom_text
    assert 'mirror_inference_duration_seconds{quantile="0.9"}' in prom_text

    stats = collector.get_summary_stats("inference_duration_seconds")
    assert stats["count"] == 5.0
    assert stats["p50"] == 0.015
    assert stats["p99"] == 0.050


# ============================================================================
# Section 5: Deployment Architecture & Configuration Validation
# ============================================================================

def test_stage12_deployment_config_and_graceful_shutdown():
    """Verifies ProductionConfig validation and GracefulShutdownManager callbacks."""
    # 1. Valid configuration
    valid_cfg = ProductionConfig.from_env({
        "MIRROR_PORT": "9000",
        "MIRROR_LOG_LEVEL": "DEBUG",
        "MIRROR_MAX_WORKERS": "16",
        "MIRROR_RATE_LIMIT_RPS": "100.0",
    })
    ok, errors = valid_cfg.validate()
    assert ok is True
    assert len(errors) == 0
    assert valid_cfg.port == 9000
    assert valid_cfg.log_level == "DEBUG"

    # 2. Invalid configuration detection
    bad_cfg = ProductionConfig(
        port=70000,  # Invalid port > 65535
        log_level="VERBOSE",  # Invalid level
        max_workers=0,  # Invalid
        rate_limit_rps=-5.0,  # Invalid
    )
    bad_ok, bad_errors = bad_cfg.validate()
    assert bad_ok is False
    assert len(bad_errors) >= 3

    # 3. GracefulShutdownManager lifecycle
    shutdown_mgr = GracefulShutdownManager(timeout_seconds=2.0)
    cleanup_called = False

    def test_cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    shutdown_mgr.register_callback("test_component_cleanup", test_cleanup)
    assert shutdown_mgr.is_shutting_down() is False

    # Trigger shutdown
    shutdown_mgr.initiate_shutdown("SIGTERM")
    assert shutdown_mgr.is_shutting_down() is True
    assert cleanup_called is True
    assert shutdown_mgr.wait_for_shutdown() is True


# ============================================================================
# Section 6: Disaster Recovery & Self-Healing
# ============================================================================

def test_stage12_vault_self_healing_and_hot_backup():
    """Verifies SQLite integrity check, auto-reconnect, and verified hot online backup."""
    if not VAULT_PATH.is_file():
        pytest.skip(f"Vault file missing: {VAULT_PATH}")

    healing_mgr = VaultSelfHealingManager(VAULT_PATH)

    # 1. Physical SQLite integrity and rule count invariant
    ok_int, details = healing_mgr.check_vault_integrity()
    assert ok_int is True, f"Integrity check failed: {details}"
    assert details["quick_check"] == "ok"
    assert details["rule_count"] == 368270

    # 2. Connection self-healing
    conn1 = healing_mgr.get_connection()
    assert conn1 is not None
    # Force close connection to test self-healing re-connect
    conn1.close()
    conn2 = healing_mgr.get_connection()
    assert conn2 is not None
    cur = conn2.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchone()[0] == 1

    # 3. Non-blocking Online Hot Backup verification
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_file = Path(tmpdir) / "test_vault_backup.dcrystal"
        ok_backup, backup_msg = healing_mgr.create_online_hot_backup(backup_file)
        assert ok_backup is True, f"Hot backup failed: {backup_msg}"
        assert backup_file.is_file()
        assert backup_file.stat().st_size > 1024

        # Validate backed up database independently
        test_conn = sqlite3.connect(str(backup_file))
        assert test_conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert test_conn.execute("SELECT count(*) FROM rules").fetchone()[0] == 368270
        test_conn.close()

    healing_mgr.close()
