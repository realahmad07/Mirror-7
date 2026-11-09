# Source-backed blueprint curation, evaluation and execution security

## 1. Legacy knowledge migration and curation

The legacy migration is explicit and resumable. It never interprets \`trust_tier\`, pass counts, or the mere presence of a test string as factual proof.

Dry-run an existing vault first:

\`\`\`bash
python scripts/migrate_legacy_blueprints.py --vault artifacts/mirror_crystal_all_datasets.dcrystal
\`\`\`

Apply the additive, paginated conversion only after reviewing the dry-run:

\`\`\`bash
python scripts/migrate_legacy_blueprints.py --vault artifacts/mirror_crystal_all_datasets.dcrystal --apply --page-size 500
\`\`\`

Legacy blueprints remain \`unverified\`. To promote a fact, create a JSONL review manifest containing one \`ClaimEvidence\` row per independent reviewer and per atomic claim. Each row must include a stable HTTPS source, source locator, quoted evidence excerpt and excerpt SHA-256, retrieval timestamp, SPDX licensing label, reviewer ID, independent review method, decision and review timestamp. The same reviewer cannot count twice.

A manifest row also supplies \`expected_claim_ids\`, which freezes the blueprint's claim scope. Promote only after all expected claims have at least two distinct independent approvals:

\`\`\`bash
python scripts/migrate_legacy_blueprints.py \\
  --vault artifacts/mirror_crystal_all_datasets.dcrystal \\
  --apply \\
  --review-manifest approved_claim_reviews.jsonl \\
  --promote-reviewed
\`\`\`

The runtime checks that claim text is present in the blueprint response template, that its evidence excerpt digest is intact, and that every frozen claim has the required independent reviews. Rejected claims, missing evidence and altered claim scopes block promotion. The review metadata is auditable data, not a cryptographic identity system; high-assurance deployments should sign reviewer approvals.

## 2. Claim-level reliability

Fact blueprints cannot be labelled reliable merely because their blueprint-level provenance fields or evidence level look complete. Every fact in the dependency chain must have a frozen claim scope and two independent source-review approvals for each claim.

The evidence ledger stores reviewers separately by blueprint, claim and reviewer ID. A promoted fact blueprint records source references, license expression, verification checks, named reviewers and verification time. Contradiction checks still apply across same-domain/same-intent blueprint records.

The implementation does not fetch every source URL at response time and does not semantically prove that an excerpt entails a claim. Reviewers must compare the quoted source passage and claim. The claim matcher also checks the exact normalized claim text against the answer template. An independent entailment service can later be added behind the same evidence contract.

## 3. Frozen source-backed factual evaluation

The evaluation-only v1 dataset is:

- \`mirror_eval/data/source_backed_factual_v1.jsonl\`
- \`mirror_eval/data/source_backed_factual_v1.manifest.json\`

The set contains seven source-backed factual cases spanning HTTP, Python, SI metrology and JSON—including two qualification/near-miss cases—plus one unanswerable private-token negative control. It is marked \`pending_independent_review\`: no reviewer approvals have been fabricated. The pinned dataset digest detects accidental edits. The data must never enter training or normal vault ingestion.

Validate structure, source links, quote hashes and frozen digest:

\`\`\`bash
python -m mirror_eval.factual_benchmark \\
  --dataset mirror_eval/data/source_backed_factual_v1.jsonl \\
  --manifest mirror_eval/data/source_backed_factual_v1.manifest.json \\
  --validate-only
\`\`\`

Scoring is intentionally blocked until two distinct independent reviewers approve every factual claim and every case's answerability. After reviews are committed, run:

\`\`\`bash
python -m mirror_eval.factual_benchmark \\
  --dataset mirror_eval/data/source_backed_factual_v1.jsonl \\
  --manifest mirror_eval/data/source_backed_factual_v1.manifest.json \\
  --vault artifacts/mirror_crystal_all_datasets.dcrystal \\
  --output artifacts/factual_v1_results.json
\`\`\`

Metrics include factual accuracy on answerable cases, accuracy among answered cases, coverage, unsupported-answer rate, claim support rate, abstention precision/recall, Brier score and expected calibration error. Calibration is computed only over answered responses because abstentions have no answer-confidence score. Each report retains each raw answer, response SHA-256, response mode, confidence, latency and per-case outcome, plus the dataset digest. The built-in claim judge is exact required-fragment matching: it is deterministic but not semantic entailment and cannot detect all extra unsupported claims. It supports alternative written forms for a single expected unit/phrase where explicitly declared. Published scores should add independent claim extraction and entailment annotations, preserve raw model answers, and report judge limitations.

The test split is frozen. Do not place benchmark prompts, answers, or excerpts in training, curation prompts, or knowledge vaults.

## 4. Container/VM execution boundary

Production execution defaults to Docker or Podman via \`MIRROR_SANDBOX_MODE=container\`. A candidate gets a fresh container with:

- network disabled (\`--network=none\`), read-only root filesystem and no host volume mounts;
- all Linux capabilities dropped, no-new-privileges and non-root UID/GID;
- cgroup CPU and memory limits, PID limit, file-size and file-descriptor limits;
- a small no-exec/no-suid/no-device temporary filesystem;
- a pinned Python minor/base image configuration and an isolated interpreter.

Pull the configured image before use:

\`\`\`bash
docker pull python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
\`\`\`

If Docker/Podman or the image is missing, production execution fails closed. It does not silently fall back to same-process execution. \`MIRROR_SANDBOX_MODE=process\` is an explicitly lower-isolation development/test override only. The CI job uses this override for broad fast unit tests and separately pulls the image and runs a real container smoke test with \`MIRROR_REQUIRE_CONTAINER_SANDBOX=1\`.

A container is a useful OS boundary, not a VM. For hostile multi-tenant or high-value workloads, use a dedicated VM/microVM runtime, hardened seccomp/AppArmor/SELinux policy, user namespaces, daemon isolation and host-level monitoring. Docker socket access must never be exposed to the model or candidate process.

## 5. Measurement policy

Keep these signals separate:
- factual accuracy: proportion of answerable test cases answered correctly;
- unsupported-answer rate: incorrect/unanswerable claims among non-abstained responses;
- claim support: source-backed gold fragments supported in output;
- abstention precision and recall: whether refusals occur on the right cases;
- calibration: Brier score and expected calibration error from the response confidence;
- coverage and latency: reported alongside accuracy, never substituted for it.

The initial source-backed set is a small seed benchmark, not enough for statistically strong conclusions. Expand it with licensed primary sources, domain-balanced and multilingual cases, adversarial near-miss prompts, temporal/freshness cases and independently reviewed unanswerable cases. Keep frozen versioned test sets and publish confidence intervals as sample size grows.
