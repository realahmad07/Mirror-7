from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.dataset import load_jsonl, validate_split_isolation
from training.native_conversation_v5 import (
    load_checkpoint,
    train as run_train_v5,
)


class TestCPUSmokePipelineV5(unittest.TestCase):
    """CPU smoke training pipeline test for V5."""

    def test_minimal_cpu_training_epoch_and_reload_v5(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "smoke_data.jsonl"
            output_ckpt = tmp_path / "cpu_v5_checkpoint.pt"

            rows = [
                {"example_id": "tr-1", "user_text": "hi", "target_text": "hello", "split": "train"},
                {"example_id": "tr-2", "user_text": "bye", "target_text": "goodbye", "split": "train"},
                {"example_id": "va-1", "user_text": "hey", "target_text": "hello there", "split": "validation"},
            ]
            with dataset_path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            examples = load_jsonl(dataset_path)
            validate_split_isolation(examples)

            class Args:
                dataset = str(dataset_path)
                output = str(output_ckpt)
                epochs = 2
                batch_size = 2
                lr = 0.001
                weight_decay = 0.01
                grad_clip = 1.0
                embedding_dim = 32
                hidden_dim = 64
                state_dim = 32
                attention_dim = 32
                dropout = 0.0
                max_bytes = 32
                device = "cpu"
                resume = None
                shuffle = True
                use_scheduler = False
                label_smoothing = 0.0

            model, history = run_train_v5(Args())
            self.assertEqual(len(history), 2)
            self.assertTrue(output_ckpt.is_file())

            reloaded_model, payload = load_checkpoint(output_ckpt, device="cpu")
            self.assertEqual(payload["format"], "mirror7-native-conversation-v5-input-conditioned")

            resp = reloaded_model.generate_text("hi", max_new_tokens=16, temperature=0.0)
            self.assertIsInstance(resp, str)


if __name__ == "__main__":
    unittest.main()
