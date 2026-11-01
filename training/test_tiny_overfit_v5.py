from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from training.native_brain import PAD, _batch, _encode
from training.native_conversation_v5 import (
    NativeConversationV5Config,
    _prepare_targets,
    build_native_conversation_v5,
    load_checkpoint,
    save_checkpoint,
)


class TestTinyOverfitV5Gate(unittest.TestCase):
    """Tiny-overfit gate for V5 Input-Conditioned Recurrent Decoder."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.train_examples = [
            ("hello", "Hello!"),
            ("how are you", "I am good."),
            ("what is mirror 7", "Mirror 7 is an agent."),
            ("who are you", "I am your AI assistant."),
            ("summarize", "Ready to summarize."),
            ("help", "I can assist you."),
            ("repeat", "Echoing your input."),
            ("bye", "Goodbye!"),
        ]
        self.config = NativeConversationV5Config(
            embedding_dim=64,
            hidden_dim=128,
            state_dim=64,
            attention_dim=64,
            dropout=0.0,
            max_bytes=64,
        )
        self.model = build_native_conversation_v5(self.config).to(self.device)

    def test_tiny_overfit_and_generation_gate(self):
        ui, um = _batch(
            [_encode(x[0], self.config.max_bytes) for x in self.train_examples],
            self.config.max_bytes,
        )
        ti, _ = _batch(
            _prepare_targets([x[1] for x in self.train_examples], self.config.max_bytes),
            self.config.max_bytes,
        )
        ui, um, ti = ui.to(self.device), um.to(self.device), ti.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.005, weight_decay=0.0)

        initial_loss = None
        final_loss = None

        self.model.train()
        for step in range(80):
            logits, _, _, _ = self.model(ui, um, ti[:, :-1])
            labels = ti[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=PAD,
            )
            if step == 0:
                initial_loss = float(loss.detach())
            final_loss = float(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        reduction_pct = 100.0 * (1.0 - (final_loss / initial_loss))
        self.assertGreaterEqual(
            reduction_pct, 90.0, f"Loss reduction {reduction_pct:.1f}% is below 90% gate!"
        )
        self.assertLess(
            final_loss, 0.10, f"Final loss {final_loss:.4f} is not below 0.10 threshold!"
        )

        # Autoregressive generation check
        self.model.eval()
        exact_matches = 0
        for prompt, target in self.train_examples:
            pred = self.model.generate_text(prompt, max_new_tokens=32, temperature=0.0).strip()
            if pred == target:
                exact_matches += 1

        exact_match_pct = 100.0 * (exact_matches / len(self.train_examples))
        self.assertGreaterEqual(
            exact_match_pct, 87.5, f"Exact match rate {exact_match_pct:.1f}% below 87.5%!"
        )

        # Checkpoint reload determinism
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "v5_overfit.pt"
            save_checkpoint(ckpt_path, self.model, self.config)
            reloaded_model, _ = load_checkpoint(ckpt_path, device="cpu")
            for prompt, _ in self.train_examples:
                orig_gen = self.model.generate_text(prompt, max_new_tokens=16, temperature=0.0)
                reload_gen = reloaded_model.generate_text(prompt, max_new_tokens=16, temperature=0.0)
                self.assertEqual(orig_gen, reload_gen)


if __name__ == "__main__":
    unittest.main()
