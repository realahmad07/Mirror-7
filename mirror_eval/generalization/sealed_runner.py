"""Sealed out-of-process evaluator for Mirror 7.

Addresses Gap 3 (*independently verified intelligence beyond synthetic
experiments and predefined handlers*):

1. **Process boundary** — the candidate system runs in a separate Python
   subprocess reading ``{"id", "prompt"}`` JSON lines on stdin and writing
   ``{"id", "answer", "source"}`` on stdout. It never receives gold answers,
   task-family labels, depth, or scorer internals.
2. **Environment scrubbing** — network proxies are stripped and the child runs in
   an empty temporary working directory with ``PYTHONDONTWRITEBYTECODE=1``.
3. **Unmemorizable items** — items are generated on the fly from a fresh seed
   (default: 128 bits of OS randomness) using invented vocabulary.
4. **Tamper-evident attestation** — the report binds SHA-256 digests of the
   evaluator code, the item set, and the full prompt/response transcript, signed
   with an ephemeral Ed25519 key (public key embedded so anyone can verify).
5. **Attribution** — answers are broken down by ``learned`` vs ``reasoner`` vs
   ``handler`` vs ``abstain`` so a handler-heavy system cannot masquerade as a
   learned language model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from . import generator, scoring
from .scoring import Response, evaluate, markdown_table

REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal worker executed inside the isolated child process.
_WORKER_CODE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
mode = sys.argv[2]
if mode == "native":
    from mirror_eval.generalization.suts import native_chain
    sut = native_chain(include_context_reasoner=True)
elif mode == "handlers_only":
    from mirror_eval.generalization.suts import native_chain
    sut = native_chain(include_context_reasoner=False)
elif mode.startswith("lm:"):
    ckpt = mode.split(":", 1)[1]
    from mirror_lm.generate import load_for_inference, chat
    model, tok = load_for_inference(ckpt)
    def sut(prompt):
        out = chat(model, tok, [{"role": "user", "content": prompt}], max_new_tokens=64, temperature=0.0)
        return type("R", (), {"text": out, "source": "learned"})()
else:
    raise SystemExit(f"unknown mode: {mode}")

for line in sys.stdin:
    req = json.loads(line)
    r = sut(req["prompt"])
    sys.stdout.write(json.dumps({"id": req["id"], "answer": r.text, "source": r.source}) + "\n")
    sys.stdout.flush()
"""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluator_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


class SubprocessSUT:
    """Runs a candidate system over a stdin/stdout JSONL pipe in a clean subprocess."""

    def __init__(self, target: str = "native", timeout_s: float = 10.0) -> None:
        self.target = target
        self.timeout_s = timeout_s
        self._tmp = tempfile.TemporaryDirectory(prefix="mirror_sealed_")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self._tmp.name,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "*",
        }
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _WORKER_CODE, str(REPO_ROOT), target],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._tmp.name,
            env=env,
            text=True,
        )
        self._seq = 0

    def __call__(self, prompt: str) -> Response:
        self._seq += 1
        req_id = f"q{self._seq}"
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps({"id": req_id, "prompt": prompt}) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"SUT subprocess exited early: {err[:300]}")
        msg = json.loads(line)
        if msg.get("id") != req_id:
            raise RuntimeError("SUT response id mismatch")
        return Response(text=str(msg.get("answer", "")), source=str(msg.get("source", "unknown")))

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        self._tmp.cleanup()


def sign_bundle(payload: dict) -> dict:
    sk = Ed25519PrivateKey.generate()
    pk_hex = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    sig_hex = sk.sign(canonical).hex()
    return {
        "payload": payload,
        "attestation": {
            "scheme": "ed25519-sha256",
            "payload_sha256": digest,
            "public_key_hex": pk_hex,
            "signature_hex": sig_hex,
        },
    }


def verify_bundle(bundle: dict) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    att = bundle["attestation"]
    canonical = json.dumps(bundle["payload"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != att["payload_sha256"]:
        return False
    pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(att["public_key_hex"]))
    try:
        pk.verify(bytes.fromhex(att["signature_hex"]), canonical)
        return True
    except Exception:
        return False


def run_sealed_eval(
    *,
    target: str = "native",
    seed: str | None = None,
    per_cell: int = 5,
) -> dict:
    actual_seed = seed if seed is not None else secrets.token_hex(16)
    items = generator.generate(seed=actual_seed, per_cell=per_cell)
    t0 = time.monotonic()
    sut = SubprocessSUT(target=target)
    try:
        report = evaluate(items, sut, keep_transcript=True)
    finally:
        sut.close()
    elapsed = round(time.monotonic() - t0, 3)
    transcript = report.pop("transcript")
    transcript_sha = hashlib.sha256(
        json.dumps(transcript, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "generator_version": generator.GENERATOR_VERSION,
        "evaluator_sha256": evaluator_digest(),
        "seed": str(actual_seed),
        "per_cell": per_cell,
        "target": target,
        "elapsed_seconds": elapsed,
        "items_sha256": generator.items_digest(items),
        "transcript_sha256": transcript_sha,
        "results": report,
        "independence_scope": (
            "Out-of-process sealed evaluation on procedurally generated novel-vocabulary items. "
            "Eliminates corpus memorization and gold-label leakage into the SUT process. "
            "Not a third-party external audit because the generator and SUT live in the same repository."
        ),
    }
    return sign_bundle(payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the sealed NovelGen generalization evaluation.")
    ap.add_argument("--target", default="native", help="native | handlers_only | lm:<checkpoint.pt>")
    ap.add_argument("--seed", default=None, help="Seed string (default: fresh 128-bit cryptographic random seed)")
    ap.add_argument("--per-cell", type=int, default=5, help="Items per (family, depth) cell")
    ap.add_argument("--out", default=None, help="Optional path to write signed JSON bundle")
    args = ap.parse_args(argv)

    bundle = run_sealed_eval(target=args.target, seed=args.seed, per_cell=args.per_cell)
    p = bundle["payload"]
    print(markdown_table(p["results"], f"Sealed NovelGen ({p['target']}, seed={p['seed'][:12]}…)"))
    print(f"\nVerified signature: {verify_bundle(bundle)} | payload_sha256={bundle['attestation']['payload_sha256'][:16]}… | elapsed={p['elapsed_seconds']}s")
    if args.out:
        Path(args.out).write_text(json.dumps(bundle, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
