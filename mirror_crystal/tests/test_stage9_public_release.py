"""Stage 9: Public Release Validation Test Suite.

Validates the complete production release boundary:
1. Package Installation & Wheel Metadata (pyproject.toml, package namespaces, CLI entrypoints).
2. Live HTTP REST API Endpoints & Epistemic Contracts (health, crystal status, query, teach, WASM).
3. Live Studio UI Integration (studio.html DOM integrity, assets, offline safety).
4. Security Guard, Secret Scan & Sandbox Quarantine (no hardcoded keys, zero-trust AST execution).
5. Containerization & Deployment Integrity (Dockerfile, docker-compose.yml syntax and configuration).
6. Documentation, Licensing & Provenance Traceability (LICENSE, README.md, restorevalutecrytaled/README.md).
"""
import ast
import json
from pathlib import Path
import re
import threading
import urllib.error
import urllib.request
import pytest

from mirror7_backend.http_server import create_server
from mirror7_backend.service import BackendService
from mirror_crystal import MirrorCrystalEngine
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def make_request(server, method: str, path: str, payload: dict | None = None, is_bytes: bool = False):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        content = response.read()
        if is_bytes:
            return response.status, content, response.headers.get("Content-Type")
        return response.status, json.loads(content.decode("utf-8")), response.headers.get("Content-Type")


def test_stage9_package_build_and_metadata():
    """Validates pyproject.toml specification, console scripts, and Python package layout."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")

    # Name and version
    assert 'name = "mirror7-backend"' in pyproject_text
    assert 'version = "0.1.0"' in pyproject_text
    assert 'requires-python = ">=3.11"' in pyproject_text
    assert "cryptography>=" in pyproject_text

    # Console script entrypoint for CLI execution
    assert "[project.scripts]" in pyproject_text
    assert 'mirror7-server = "mirror7_backend.http_server:serve"' in pyproject_text

    # Packages to include
    for pkg in ("mirror7_backend", "mirror_crystal", "mirror_eval"):
        assert (REPO_ROOT / pkg).is_dir(), f"Package directory {pkg} must exist"
        assert (REPO_ROOT / pkg / "__init__.py").is_file(), f"{pkg}/__init__.py must exist"


def test_stage9_api_endpoints_and_contracts():
    """Validates full REST API endpoints against ephemeral in-process HTTP server."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair("What is the speed of light?", "The speed of light in vacuum is approximately 299,792,458 m/s.")

    service = BackendService(crystal_vault=eng, enable_crystal_auto_load=False)
    server = create_server(host="127.0.0.1", port=0, service=service)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # 1. GET /api/health
        code_h, data_h, _ = make_request(server, "GET", "/api/health")
        assert code_h == 200
        assert data_h["ok"] is True
        assert data_h["service"] == "mirror7"
        assert "sessions" in data_h["status"]

        # 2. GET /api/crystal/status
        code_s, data_s, _ = make_request(server, "GET", "/api/crystal/status")
        assert code_s == 200
        assert data_s["ok"] is True
        assert data_s["crystal"]["active"] is True
        assert data_s["crystal"]["rule_count"] >= 1

        # 3. POST /api/crystal/ask
        code_a, data_a, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": "What is the speed of light?"})
        assert code_a == 200
        assert "299,792,458" in data_a["answer"]
        assert "epistemic_tier" in data_a
        assert data_a["answer_qualification"] == "unverified_candidate"

        # 4. POST /api/crystal/teach
        code_t, data_t, _ = make_request(
            server,
            "POST",
            "/api/crystal/teach",
            {"question": "What is Planck constant?", "answer": "Planck constant h is approximately 6.626e-34 J s."},
        )
        assert code_t == 200
        assert data_t["ok"] is True

        # Verify query after teaching
        code_at, data_at, _ = make_request(server, "POST", "/api/crystal/ask", {"prompt": "What is Planck constant?"})
        assert "6.626e-34" in data_at["answer"]

        # 5. GET /api/crystal/wasm
        code_w, wasm_bytes, ct_w = make_request(server, "GET", "/api/crystal/wasm", is_bytes=True)
        assert code_w == 200
        assert wasm_bytes[:4] == b"\x00asm"
        assert "application/wasm" in ct_w

        # 6. GET / and /studio UI serving
        code_u1, html_bytes1, ct1 = make_request(server, "GET", "/", is_bytes=True)
        assert code_u1 == 200
        assert b"Mirror Crystal Live Studio" in html_bytes1
        assert "text/html" in ct1

        code_u2, html_bytes2, ct2 = make_request(server, "GET", "/studio", is_bytes=True)
        assert code_u2 == 200
        assert b"Mirror Crystal Live Studio" in html_bytes2
    finally:
        server.shutdown()
        server.server_close()


def test_stage9_ui_studio_integration():
    """Validates studio.html DOM structure, offline assets, and interaction elements."""
    studio_path = REPO_ROOT / "mirror7_backend" / "studio.html"
    assert studio_path.is_file(), "mirror7_backend/studio.html must exist"
    assert studio_path.stat().st_size > 10000, "studio.html should be substantial (>10KB)"

    html_text = studio_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text or "<!doctype html>" in html_text.lower()
    assert "<title>" in html_text
    assert "Mirror Crystal" in html_text

    # Critical UI element IDs / classes
    assert "prompt" in html_text.lower()
    assert "btn" in html_text.lower() or "button" in html_text.lower()
    assert "status" in html_text.lower()
    assert "theme" in html_text.lower() or "dark" in html_text.lower()

    # Verify scripts: only allow declared UI libraries (tailwindcss, lucide)
    external_scripts = re.findall(r'<script\s+[^>]*src=["\'](http[s]?://[^"\']+)["\']', html_text, re.IGNORECASE)
    allowed_ui_cdns = {"https://cdn.tailwindcss.com", "https://unpkg.com/lucide@latest"}
    unauthorized_cdns = set(external_scripts) - allowed_ui_cdns
    assert len(unauthorized_cdns) == 0, f"Found unauthorized external script CDNs: {unauthorized_cdns}"


def test_stage9_security_and_sandbox_integrity():
    """Validates security guard policies: zero hardcoded secrets and strict sandbox AST enforcement."""
    # 1. AST Sandbox Judge tests
    dangerous_payloads = [
        "import os; os.system('rm -rf /')",
        "__import__('subprocess').call(['ls'])",
        "[c for c in ().__class__.__bases__[0].__subclasses__() if c.__name__ == 'BuiltinImporter']",
        "eval('2 + 2')",
        "exec('print(1)')",
        "open('/etc/passwd').read()",
    ]
    for bad_code in dangerous_payloads:
        is_safe, msg = ZeroTrustSandboxJudge.is_ast_safe(bad_code)
        assert is_safe is False, f"Sandbox failed to reject dangerous code: {bad_code} (msg: {msg})"

    # 2. Secret Scan across repository source files (excluding virtual environments and build artifacts)
    secret_patterns = [
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bghp_[0-9a-zA-Z]{36}\b"),
    ]
    scan_exts = {".py", ".json", ".yml", ".yaml", ".md", ".toml", ".sh"}
    ignored_dirs = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "dist", "build"}
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in scan_exts:
            continue
        # Skip git metadata, virtual environments, build artifacts, and test caches
        if any(ignored in p.parts for ignored in ignored_dirs):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in secret_patterns:
            matches = pat.findall(content)
            assert not matches, f"Potential sensitive credential detected in {p}: {matches}"


def test_stage9_deployment_artifacts():
    """Validates Dockerfile and docker-compose.yml configuration and syntax."""
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.is_file(), "Dockerfile must exist"
    df_content = dockerfile.read_text(encoding="utf-8")

    assert "FROM python:3.11" in df_content or "FROM python:3.12" in df_content
    assert "WORKDIR /app" in df_content
    assert "COPY . /app" in df_content
    assert "EXPOSE 8787" in df_content
    assert "HEALTHCHECK" in df_content
    assert "mirror7_backend.http_server" in df_content

    compose_file = REPO_ROOT / "docker-compose.yml"
    assert compose_file.is_file(), "docker-compose.yml must exist"
    compose_content = compose_file.read_text(encoding="utf-8")
    assert "mirror7-backend:" in compose_content
    assert "8787:8787" in compose_content
    assert "MIRROR7_PORT" in compose_content


def test_stage9_documentation_and_vault_traceability():
    """Validates LICENSE, README.md, and production vault documentation."""
    # 1. License
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.is_file(), "LICENSE must exist"
    lic_text = license_path.read_text(encoding="utf-8")
    assert "Apache License" in lic_text or "Version 2.0" in lic_text

    # 2. Main README.md
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file(), "README.md must exist"
    readme_text = readme_path.read_text(encoding="utf-8")
    assert len(readme_text) > 1000, "README.md must be comprehensive"
    assert "Mirror" in readme_text
    assert "Stage" in readme_text

    # 3. Vault documentation
    vault_readme = REPO_ROOT / "restorevalutecrytaled" / "README.md"
    assert vault_readme.is_file(), "restorevalutecrytaled/README.md must exist"
    v_text = vault_readme.read_text(encoding="utf-8")
    assert "mirror_crystal_all_datasets.dcrystal" in v_text
    assert "368,270" in v_text
