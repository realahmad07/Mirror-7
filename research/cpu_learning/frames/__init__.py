"""Learner-facing L0/L2 typed state frames, partial observation masks, and role-frame parsers.

STRICT ISOLATION: Modules in this package MUST NEVER import ``hidden_actions``,
``oracle``, or ``generators``, and MUST NEVER receive gold transition schemas or
evaluator hidden labels.
"""
from __future__ import annotations

from .frame_parser import (
    BaseFrameParser,
    BilingualCCGFrameAdapter,
    ByteToken,
    TypedRoleFrameParser,
    tokenize_utf8_bytes,
)
from .state_frame import (
    CandidateActionFrame,
    FactEpistemicStatus,
    FrameParseResult,
    MaskedDeltaReport,
    TypedEntityFrame,
    TypedStateFrame,
    assert_no_hidden_fields,
    compute_canonical_state_action_key,
    extract_masked_sparse_delta,
    verify_unobserved_preservation,
)

__all__ = [
    "BaseFrameParser",
    "BilingualCCGFrameAdapter",
    "ByteToken",
    "CandidateActionFrame",
    "FactEpistemicStatus",
    "FrameParseResult",
    "MaskedDeltaReport",
    "TypedEntityFrame",
    "TypedRoleFrameParser",
    "TypedStateFrame",
    "assert_no_hidden_fields",
    "compute_canonical_state_action_key",
    "extract_masked_sparse_delta",
    "tokenize_utf8_bytes",
    "verify_unobserved_preservation",
]
