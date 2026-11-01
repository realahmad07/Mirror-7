from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import torch

from training.dataset import load_jsonl, validate_split_isolation
from training.native_conversation import (
    NativeConversationConfig,
    build_native_conversation,
    load_checkpoint,
    train as run_train,
)


class TestCPUSmokePipeline(unittest.TestCase):
    """PHASE 5 — Verification of CPU-only training pipeline and real inference."""

    def test_minimal_cpu_training_epoch_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "smoke_data.jsonl"
            output_ckpt = tmp_path / "cpu_checkpoint.pt"

            # 1. Prepare minimal valid dataset conforming to schema
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
            self.assertEqual(len(examples), 3)

            # 2. Run minimal CPU training
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
                layers = 1
                dropout = 0.0
                attention_dim = 32
                max_bytes = 32
                device = "cpu"

            model, history = run_train(Args())

            # 3. Verify history and checkpoint
            self.assertEqual(len(history), 2)
            self.assertIn("train_loss", history[0])
            self.assertIn("validation_loss", history[0])
            self.assertTrue(output_ckpt.is_file())

            # 4. Reload using real inference path
            reloaded_model, payload = load_checkpoint(output_ckpt, device="cpu")
            self.assertEqual(payload["format"], "mirror7-native-conversation-v3-cross-attention")
            self.assertEqual(payload["config"]["embedding_dim"], 32)

            # 5. Generate response successfully
            response = reloaded_model.generate_text("hi", max_new_tokens=16, temperature=0.0)
            self.assertIsInstance(response, str)
            self.assertGreaterEqual(len(response), 0)


if __name__ == "__main__":
    unittest.main()
