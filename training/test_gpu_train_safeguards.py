from __future__ import annotations

import pytest

from training.gpu_train import (
    DynamicBucketSampler, accumulation_group_size, verify_resume_fingerprint,
)
from training.knowledge_transfer.collector import _split_for_prompt
from training.schema import TrainingExample


def test_bucket_sort_uses_longer_of_prompt_and_target():
    examples = [
        TrainingExample(example_id="short", user_text="x", target_text="y"),
        TrainingExample(example_id="long-prompt", user_text="x" * 30, target_text="y"),
        TrainingExample(example_id="long-target", user_text="x", target_text="y" * 20),
    ]
    batches = DynamicBucketSampler(examples, batch_size=1, max_bytes=64).get_batches(shuffle=False)
    assert [batch[0].example_id for batch in batches] == ["short", "long-target", "long-prompt"]


def test_accumulation_group_handles_partial_final_group():
    assert [accumulation_group_size(5, i, 2) for i in range(5)] == [2, 2, 2, 2, 1]
    assert [accumulation_group_size(3, i, 4) for i in range(3)] == [3, 3, 3]
    with pytest.raises(ValueError):
        accumulation_group_size(3, 0, 0)


def test_resume_requires_matching_dataset_fingerprint():
    verify_resume_fingerprint("abc", "abc")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_resume_fingerprint("abc", "def")
    with pytest.raises(ValueError, match="no dataset fingerprint"):
        verify_resume_fingerprint(None, "abc")
    verify_resume_fingerprint(None, "abc", allow_unverified=True)


def test_prompt_split_is_stable_across_task_id_and_whitespace_variants():
    ratio = (0.8, 0.1, 0.1)
    assert _split_for_prompt("What is 2 + 2?", ratio) == _split_for_prompt("  WHAT   IS 2 + 2?  ", ratio)
