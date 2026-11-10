# Stage 10.5.1 — Exhaustive History Scan & Trust-Status Correction

## Scope

This corrective stage closes the reproducible CI defect in Stage 10.5 and makes Git-history secret-scan results evidence-derived rather than hardcoded.

## Implemented controls

1. **Full-history CI checkout:** CI and manual deep-regression workflows use checkout with fetch-depth zero, so git rev-list --objects --all sees complete reachable history.
2. **No blob-size exclusion:** Every reachable Git blob is scanned, including blobs larger than 500 KB. The scanner records blob count, large-blob count, and total bytes scanned.
3. **Dynamic secret accounting:** The unrevoked_secrets_in_history result is computed from detected findings, not a constant.
4. **CRL classification:** Known historical v1 public keys and the exposed legacy reviewer seed markers are classified as CRL-covered. Unrelated Ed25519 seeds, private-key PEMs, and common cloud tokens remain unrevoked findings and fail the gate.
5. **Working-tree scan:** Tracked and non-ignored untracked files are scanned, including LFS-materialized files present in the CI workspace.
6. **Adversarial fixtures:** Temporary-repository tests prove that a greater-than-500-KB blob is inspected, known revoked material is classified as revoked, and unrelated private seeds increment the unrevoked count.
7. **Honest reviewer assurance:** ORCID check digits, embedded OpenPGP fingerprints, and signatures against repository-pinned public keys are consistency checks. They do not independently prove ORCID account ownership, real-world identity, institutional affiliation, or external ownership of institutional root keys. The factual benchmark manifest therefore reports `cryptographic_attestations_verified_identity_unverified` until authoritative external identity records are independently corroborated.

## Scope limitation

The Git object scan covers reachable Git objects and blobs. Git LFS stores file contents outside ordinary Git blobs; the current production .dcrystal file is scanned in the materialized working tree, but historical LFS objects not fetched into the workspace are not represented as ordinary Git blobs and are not claimed as scanned.

## Completion gate

Stage 10.5.1 is technically complete only when:
- the historical scan reports the full object/blob counts with no blob-size cutoff;
- all adversarial history-scan tests pass;
- unrevoked_secrets_in_history is computed from actual findings and equals zero for the real repository;
- the production vault integrity check passes;
- the configured full regression CI is green.

External reviewer identity and institutional-root ownership remain pending until independently reachable authoritative evidence or direct out-of-band confirmation is available. This is not inferred from checksums.
