#!/usr/bin/env python3
"""Out-of-band signing tool for independent external reviewers.

SECURITY NOTICE:
This utility runs strictly out-of-band on the auditor's local workstation.
It expects the reviewer's private key via the environment variable
`REVIEWER_ED25519_PRIVATE_KEY_HEX` or secure prompt.
DO NOT store or commit private keys into this repository.
"""
from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519

from mirror_eval.stage10_3_trust_infrastructure import (
    DigitalAttestationEngine,
    ReviewerIdentityRegistry,
)


def main():
    reviewer_id = os.environ.get("REVIEWER_ID")
    if not reviewer_id and len(sys.argv) > 1:
        reviewer_id = sys.argv[1]

    if not reviewer_id:
        print("Usage: python3 scripts/reviewer_sign_attestation.py <reviewer_id>", file=sys.stderr)
        sys.exit(1)

    ident = ReviewerIdentityRegistry.get_identity(reviewer_id)
    if not ident:
        print(f"Error: Unknown reviewer identity '{reviewer_id}' in registry.", file=sys.stderr)
        sys.exit(1)

    # Secure key intake
    priv_hex = os.environ.get("REVIEWER_ED25519_PRIVATE_KEY_HEX")
    if not priv_hex:
        priv_hex = getpass.getpass(f"Enter Ed25519 private key hex for {ident.name}: ").strip()

    try:
        priv_bytes = bytes.fromhex(priv_hex)
        priv_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    except Exception as e:
        print(f"Error: Invalid private key format: {e}", file=sys.stderr)
        sys.exit(1)

    pub_hex = priv_key.public_key().public_bytes_raw().hex()
    if pub_hex.lower() != ident.public_key_hex.lower():
        print(f"Error: Supplied private key does not match registered public key for {ident.name}!", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully authenticated out-of-band private key for {ident.name} ({ident.role}).")


if __name__ == "__main__":
    main()
