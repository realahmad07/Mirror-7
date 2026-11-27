"""Verified external dataset and license audit registry for Stage S11.0.

GOVERNANCE & LICENSING RULES:
- Every external dataset MUST have its upstream license verified before use.
- Zero raw external benchmark text is copied or redistributed inside this Apache-2.0 repository.
- Active evaluation datasets are downloaded at runtime into gitignored ``.external_eval_cache/``
  and verified against pinned URLs, Git revisions, byte lengths, and SHA-256 digests.
- Datasets with NonCommercial (CC-BY-NC), unspecified/other/NOASSERTION licenses, or legal
  unavailability (HTTP 451) are explicitly REJECTED and excluded from evaluation manifests.
"""
from __future__ import annotations

from typing import Any

# Complete 14-dataset empirical audit verified directly against Hugging Face & GitHub APIs on 2026-11-27.
DATASET_LICENSE_REGISTRY: tuple[dict[str, Any], ...] = (
    # ---- APPROVED ACTIVE STAGE S11.0 BENCHMARKS (3 real-world benchmarks) ----
    {
        "dataset_id": "balanced_copa",
        "display_name": "Balanced COPA (Kavumba et al., ACL 2019; extends Roemmele et al., 2011)",
        "domain_role": "real_world_causal_commonsense_language",
        "status": "APPROVED_ACTIVE",
        "verified_license": "CC-BY-4.0 (extensions) + BSD-2-Clause (original COPA)",
        "commercial_safe": True,
        "share_alike": False,
        "redistribution_policy": "runtime_fetch_only_zero_redistribution_in_git",
        "source_repo": "https://huggingface.co/datasets/pkavumba/balanced-copa",
        "pinned_revision": "813bd03cd6e07d9bd8d7333896ad5d40abb95ea9",
        "files": [
            {
                "name": "train.csv",
                "url": "https://huggingface.co/datasets/pkavumba/balanced-copa/resolve/813bd03cd6e07d9bd8d7333896ad5d40abb95ea9/train.csv",
                "size_bytes": 111753,
                "sha256": "650ec76d0b0b46511bcf68702f9e371cc3d9ad3074f947659d98155cc19d56ef",
            },
            {
                "name": "test.csv",
                "url": "https://huggingface.co/datasets/pkavumba/balanced-copa/resolve/813bd03cd6e07d9bd8d7333896ad5d40abb95ea9/test.csv",
                "size_bytes": 53898,
                "sha256": "b57415ab077856be06f083fd39f72a3341e393630d6af1ce4b46a8be2a095972",
            },
        ],
        "source_disjoint_unit": "choice_pair_cluster (keeps original + mirrored counterfactual pairs strictly in same split; official test.csv has 0 choice-pair overlap with train.csv)",
        "rationale": (
            "Organic real-world English causal commonsense (cause/effect) with mirrored counterfactual "
            "premises that expose superficial lexical shortcuts; clean CC-BY-4.0 license; pure CSV "
            "readable by Python stdlib with zero external dependencies."
        ),
    },
    {
        "dataset_id": "squad_v2",
        "display_name": "SQuAD 2.0 Development Set (Rajpurkar, Jia & Liang, ACL 2018)",
        "domain_role": "real_world_evidence_grounded_qa_and_abstention",
        "status": "APPROVED_ACTIVE",
        "verified_license": "CC-BY-SA-4.0",
        "commercial_safe": True,
        "share_alike": True,
        "redistribution_policy": "runtime_fetch_only_zero_redistribution_in_git_attribution_preserved",
        "source_repo": "https://rajpurkar.github.io/SQuAD-explorer/ (mirror: https://huggingface.co/datasets/rajpurkar/squad_v2)",
        "pinned_revision": "dev-v2.0.json (sha256-pinned)",
        "files": [
            {
                "name": "dev-v2.0.json",
                "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
                "size_bytes": 4370528,
                "sha256": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
            }
        ],
        "source_disjoint_unit": "wikipedia_article_title (35 distinct topic domains partitioned 14 train / 7 val / 14 hidden_eval with 0 shared articles)",
        "rationale": (
            "Real-world unstructured Wikipedia prose with exact character-offset gold evidence spans "
            "AND ~50% human-authored unanswerable questions (is_impossible=True), enabling real "
            "evidence grounding, cross-topic transfer, and non-synthetic abstention measurement. "
            "Because license is CC-BY-SA-4.0, raw text is NEVER committed into the Apache-2.0 repo."
        ),
    },
    {
        "dataset_id": "longmemeval",
        "display_name": "LongMemEval (Wu et al., ICLR 2025 — xiaowu0162/longmemeval-cleaned)",
        "domain_role": "real_world_long_term_multi_session_memory",
        "status": "APPROVED_ACTIVE",
        "verified_license": "MIT (verified on both GitHub SPDX and Hugging Face dataset card)",
        "commercial_safe": True,
        "share_alike": False,
        "redistribution_policy": "runtime_fetch_only_zero_redistribution_in_git",
        "source_repo": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (code: https://github.com/xiaowu0162/LongMemEval)",
        "pinned_revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "files": [
            {
                "name": "longmemeval_oracle.json",
                "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json",
                "size_bytes": 15388478,
                "sha256": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
            }
        ],
        "optional_full_haystack_file": {
            "name": "longmemeval_s_cleaned.json",
            "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json",
            "size_bytes": 277383467,
            "note": "Full 277 MB ~40-session haystack audited & pinned for optional offline heavy runs; S11.0 CI uses 15.4 MB oracle file with split-internal K=10 chronological distractor-session augmentation.",
        },
        "source_disjoint_unit": "connected_component_over_shared_haystack_session_ids (resolves 8 cross-question shared sessions among 940 unique sessions; zero session-ID overlap across splits)",
        "rationale": (
            "Real multi-session conversational memory benchmark (500 episodes, 6 memory categories: "
            "single-session user/assistant/preference, multi-session, temporal-reasoning, "
            "knowledge-update) with session/turn gold evidence grounding (answer_session_ids + has_answer) "
            "and 30 genuine unanswerable abstention questions (*_abs). Permissive MIT license."
        ),
    },
    # ---- APPROVED SECONDARY / DEFERRED CANDIDATES (S11.1+ expansion) ----
    {
        "dataset_id": "ruletaker",
        "display_name": "RuleTaker (Clark, Tafjord & Richardson, IJCAI 2020 — tasksource/ruletaker)",
        "domain_role": "controlled_natural_language_deduction_reference",
        "status": "APPROVED_DEFERRED",
        "verified_license": "Apache-2.0",
        "commercial_safe": True,
        "share_alike": False,
        "redistribution_policy": "runtime_fetch_only",
        "source_repo": "https://huggingface.co/datasets/tasksource/ruletaker",
        "pinned_revision": "64d8c52fddf8054b91b0791fe6535b9b5b6e2b96",
        "rationale": (
            "Clean Apache-2.0 license (also manifest-pinned in PR #43), but templated/controlled "
            "synthetic English — does not test organic open-world language. Deferred as secondary "
            "controlled-deduction diagnostic in S11.1+."
        ),
    },
    {
        "dataset_id": "babi_qa",
        "display_name": "bAbI QA (Weston et al., Facebook AI Research)",
        "domain_role": "synthetic_micro_world_qa_reference",
        "status": "APPROVED_DEFERRED",
        "verified_license": "CC-BY-3.0",
        "commercial_safe": True,
        "share_alike": False,
        "redistribution_policy": "runtime_fetch_only",
        "source_repo": "https://huggingface.co/datasets/facebook/babi_qa",
        "pinned_revision": "021d7aeb7307b7856dd0632f92827bc607dc2f1b",
        "rationale": (
            "Permissive CC-BY-3.0, already used as a narrow parser probe in five-organ external eval, "
            "but purely synthetic templated stories. Not sufficient as an open-world benchmark."
        ),
    },
    {
        "dataset_id": "boolq",
        "display_name": "BoolQ (Clark et al., NAACL 2019 — google/boolq)",
        "domain_role": "real_world_yes_no_reading_comprehension",
        "status": "APPROVED_DEFERRED",
        "verified_license": "CC-BY-SA-3.0",
        "commercial_safe": True,
        "share_alike": True,
        "redistribution_policy": "runtime_fetch_only_zero_redistribution_in_git",
        "source_repo": "https://huggingface.co/datasets/google/boolq",
        "pinned_revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "rationale": (
            "Valid real-world Wikipedia QA under CC-BY-SA-3.0, but HF release uses Parquet only and "
            "lacks sentence-offset evidence spans and explicit unanswerable abstention questions "
            "compared to SQuAD 2.0."
        ),
    },
    {
        "dataset_id": "ai2_arc",
        "display_name": "AI2 ARC Easy & Challenge (Clark et al., 2018 — allenai/ai2_arc)",
        "domain_role": "real_world_science_commonsense_mcqa",
        "status": "APPROVED_DEFERRED",
        "verified_license": "CC-BY-SA-4.0",
        "commercial_safe": True,
        "share_alike": True,
        "redistribution_policy": "runtime_fetch_only_zero_redistribution_in_git",
        "source_repo": "https://huggingface.co/datasets/allenai/ai2_arc",
        "pinned_revision": "210d026faf9955653af8916fad021475a3f00453",
        "rationale": (
            "Strong real-world science commonsense benchmark under CC-BY-SA-4.0; lacks passage "
            "evidence spans in the core split and uses HF Parquet. Reserved for S11.2+ world-knowledge "
            "transfer once a verified corpus-learning loop exists."
        ),
    },
    {
        "dataset_id": "hotpotqa",
        "display_name": "HotpotQA (Yang et al., EMNLP 2018 — hotpotqa/hotpot_qa)",
        "domain_role": "multi_hop_evidence_grounded_qa",
        "status": "APPROVED_DEFERRED",
        "verified_license": "CC-BY-SA-4.0 (dataset) / Apache-2.0 (upstream code)",
        "commercial_safe": True,
        "share_alike": True,
        "redistribution_policy": "runtime_fetch_only_zero_redistribution_in_git",
        "source_repo": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "pinned_revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "rationale": (
            "High-value multi-hop evidence grounding benchmark, but primary CMU JSON host returned "
            "DNS resolution failure (Errno 11001) during audit and HF mirror is Parquet-only. "
            "Deferred to S11.1+ once pinned JSON mirror or optional parquet reader is approved."
        ),
    },
    {
        "dataset_id": "commonsenseqa",
        "display_name": "CommonsenseQA (Talmor et al., NAACL 2019 — tau/commonsense_qa)",
        "domain_role": "five_way_commonsense_mcqa",
        "status": "APPROVED_DEFERRED",
        "verified_license": "MIT (HF card; upstream ConceptNet derived under CC-BY-SA-4.0; upstream GH repo has no LICENSE file)",
        "commercial_safe": True,
        "share_alike": True,
        "redistribution_policy": "runtime_fetch_only_pending_upstream_conceptnet_sa_review",
        "source_repo": "https://huggingface.co/datasets/tau/commonsense_qa",
        "pinned_revision": "94630fe30dad47192a8546eb75f094926d47e155",
        "rationale": (
            "HF card declares MIT, but upstream GitHub repo has no SPDX license file and questions "
            "derive from ConceptNet (CC-BY-SA-4.0). Deferred in favor of cleaner CC-BY-4.0 Balanced COPA."
        ),
    },
    {
        "dataset_id": "winogrande",
        "display_name": "WinoGrande (Sakaguchi et al., AAAI 2020 — allenai/winogrande)",
        "domain_role": "commonsense_coreference_disambiguation",
        "status": "APPROVED_DEFERRED",
        "verified_license": "Apache-2.0 (upstream GitHub allenai/winogrande; HF card missing license tag)",
        "commercial_safe": True,
        "share_alike": False,
        "redistribution_policy": "runtime_fetch_only_after_official_archive_checksum_pinning",
        "source_repo": "https://github.com/allenai/winogrande",
        "pinned_revision": "master (checksum pending)",
        "rationale": (
            "Clean Apache-2.0 upstream repository; HF dataset card lacked license metadata during audit. "
            "Strong S11.1+ candidate for coreference/pronoun commonsense."
        ),
    },
    # ---- REJECTED / EXCLUDED DATASETS ----
    {
        "dataset_id": "clutrr",
        "display_name": "CLUTRR (Sinha et al., EMNLP 2019 — facebookresearch/clutrr)",
        "domain_role": "kinship_relational_reasoning",
        "status": "REJECTED_LICENSE",
        "verified_license": "CC-BY-NC-4.0 (upstream repo; HF card: unknown; GH SPDX: NOASSERTION)",
        "commercial_safe": False,
        "share_alike": False,
        "redistribution_policy": "FORBIDDEN",
        "source_repo": "https://github.com/facebookresearch/clutrr",
        "pinned_revision": "excluded",
        "rationale": (
            "Non-commercial CC-BY-NC-4.0 restriction and missing HF license tag; consistently "
            "excluded in both PR #43 and Stage S11.0."
        ),
    },
    {
        "dataset_id": "rocstories_cloze",
        "display_name": "ROCStories / StoryCloze (Mostafazadeh et al. — qikp/rocstories-cloze)",
        "domain_role": "narrative_commonsense_ending_ranking",
        "status": "REJECTED_LICENSE",
        "verified_license": "other / unspecified (HF mirror); upstream University of Rochester gated research terms",
        "commercial_safe": False,
        "share_alike": False,
        "redistribution_policy": "FORBIDDEN_NO_REDISTRIBUTION",
        "source_repo": "https://huggingface.co/datasets/qikp/rocstories-cloze",
        "pinned_revision": "excluded_from_s11",
        "rationale": (
            "HF mirror declares license='other' and upstream Rochester release requires non-redistributable "
            "research-only registration. Excluded from Stage S11 manifests (preserved only as historical "
            "context in scripts/run_five_organ_external_eval.py)."
        ),
    },
    {
        "dataset_id": "locomo",
        "display_name": "LoCoMo (Maharana et al., ACL 2024 — snap-research/locomo)",
        "domain_role": "long_term_conversational_memory",
        "status": "REJECTED_LICENSE",
        "verified_license": "NOASSERTION (GitHub API reports Other/NOASSERTION; LICENSE file HTTP 404)",
        "commercial_safe": False,
        "share_alike": False,
        "redistribution_policy": "FORBIDDEN_UNTIL_VERIFIED",
        "source_repo": "https://github.com/snap-research/locomo",
        "pinned_revision": "excluded",
        "rationale": (
            "Upstream repository lacks a verified standard SPDX license file (NOASSERTION / 404 on "
            "LICENSE). Excluded in favor of cleanly MIT-licensed LongMemEval."
        ),
    },
    {
        "dataset_id": "hellaswag",
        "display_name": "HellaSwag (Zellers et al., ACL 2019 — rowanz/hellaswag)",
        "domain_role": "commonsense_grounded_continuation",
        "status": "REJECTED_UNAVAILABLE",
        "verified_license": "UNVERIFIED (GitHub upstream returned HTTP 451 Unavailable For Legal Reasons; HF card missing license)",
        "commercial_safe": False,
        "share_alike": False,
        "redistribution_policy": "FORBIDDEN",
        "source_repo": "https://huggingface.co/datasets/Rowan/hellaswag",
        "pinned_revision": "excluded",
        "rationale": (
            "Upstream GitHub repository returned HTTP 451 (Unavailable For Legal Reasons) and HF card "
            "has no license metadata. Excluded."
        ),
    },
)


def get_active_dataset_specs() -> tuple[dict[str, Any], ...]:
    """Return the three license-verified datasets approved for active Stage S11.0 evaluation."""
    return tuple(d for d in DATASET_LICENSE_REGISTRY if d["status"] == "APPROVED_ACTIVE")


def audit_dataset_licenses() -> dict[str, Any]:
    """Return a structured summary of the 14-dataset external license and provenance audit."""
    by_status: dict[str, list[str]] = {}
    for item in DATASET_LICENSE_REGISTRY:
        by_status.setdefault(item["status"], []).append(item["dataset_id"])

    active = get_active_dataset_specs()
    # Hard governance assertion: zero non-commercial or rejected datasets in active set.
    active_all_commercial_safe = all(bool(d.get("commercial_safe")) for d in active)
    active_zero_git_redistribution = all(
        "zero_redistribution_in_git" in str(d.get("redistribution_policy", ""))
        for d in active
    )
    active_all_sha256_pinned = all(
        all(len(str(f.get("sha256", ""))) == 64 for f in d.get("files", []))
        for d in active
    )

    return {
        "audit_date": "2026-11-27",
        "total_datasets_reviewed": len(DATASET_LICENSE_REGISTRY),
        "counts_by_status": {k: len(v) for k, v in sorted(by_status.items())},
        "datasets_by_status": by_status,
        "active_s11_0_benchmarks": [d["dataset_id"] for d in active],
        "license_compliance_checks": {
            "active_all_commercial_safe": active_all_commercial_safe,
            "active_zero_git_redistribution": active_zero_git_redistribution,
            "active_all_sha256_pinned": active_all_sha256_pinned,
            "rejected_noncommercial_or_unclear_excluded": all(
                d["dataset_id"] not in [a["dataset_id"] for a in active]
                for d in DATASET_LICENSE_REGISTRY
                if d["status"].startswith("REJECTED")
            ),
        },
        "registry": list(DATASET_LICENSE_REGISTRY),
    }
