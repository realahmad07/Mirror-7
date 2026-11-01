from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import torch

from training.native_brain import PAD, _batch, _encode
from training.native_conversation import (
    NativeConversationConfig,
    _prepare_targets,
    build_native_conversation,
    load_checkpoint,
    save_checkpoint,
)


class TestTinyOverfitGate(unittest.TestCase):
    """PHASE 4 — Tiny-overfit gate on 8 deterministic training examples.

    Validates that:
    1. Gradients flow correctly through all layers.
    2. Model parameters update strictly from training data.
    3. Teacher-forced loss drops by >90% (justified criterion: final_loss < 0.10).
    4. Free-running autoregressive generation produces accurate, non-degenerate targets.
    5. Checkpoint save and reload maintains deterministic inference.
    """

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
        self.held_out_val = [
            ("good morning", "Good morning. What are you working on?"),
            ("tell me a story", "Once upon a time in Mirror 7."),
        ]

        self.config = NativeConversationConfig(
            embedding_dim=64,
            hidden_dim=128,
            state_dim=64,
            layers=1,
            dropout=0.0,
            max_bytes=64,
            attention_dim=64,
        )
        self.model = build_native_conversation(self.config).to(self.device)

    def test_tiny_overfit_and_generation_gate(self):
        # 1. Prepare training tensors
        ui, um = _batch([_encode(x[0], self.config.max_bytes) for x in self.train_examples], self.config.max_bytes)
        target_rows = _prepare_targets([x[1] for x in self.train_examples], self.config.max_bytes)
        ti, tm = _batch(target_rows, self.config.max_bytes)
        ui, um, ti = ui.to(self.device), um.to(self.device), ti.to(self.device)

        # 2. Record initial parameters and loss
        initial_params = {name: p.clone().detach() for name, p in self.model.named_parameters()}

        self.model.eval()
        with torch.no_grad():
            init_logits, _, _, _ = self.model(ui, um, ti[:, :-1])
            init_loss = float(
                torch.nn.functional.cross_entropy(
                    init_logits.reshape(-1, init_logits.shape[-1]),
                    ti[:, 1:].reshape(-1),
                    ignore_index=PAD,
                ).cpu()
            )

        self.assertGreater(init_loss, 4.0, f"Expected initial loss ~5.5, got {init_loss}")

        # 3. Optimize on training examples only
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.005)
        self.model.train()

        for step in range(80):
            optimizer.zero_grad(set_to_none=True)
            logits, _, _, _ = self.model(ui, um, ti[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                ti[:, 1:].reshape(-1),
                ignore_index=PAD,
            )
            self.assertTrue(torch.isfinite(loss), f"Loss became non-finite at step {step}")
            loss.backward()

            # Verify gradients are finite on every step
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Parameter {name} missing gradient")
                    self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient non-finite for {name}")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        final_loss = float(loss.detach().cpu())

        # 4. Predeclared, justified criterion for substantial loss reduction:
        # Predeclared target: loss must reduce by >= 90% and reach final_loss < 0.10.
        loss_reduction_pct = (init_loss - final_loss) / init_loss * 100.0
        self.assertGreaterEqual(
            loss_reduction_pct,
            90.0,
            f"Loss reduction was only {loss_reduction_pct:.2f}% (from {init_loss:.4f} to {final_loss:.4f})",
        )
        self.assertLess(
            final_loss,
            0.10,
            f"Final loss {final_loss:.4f} did not meet < 0.10 criterion",
        )

        # 5. Confirm parameter updates
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                diff = (param.detach() - initial_params[name]).abs().sum().item()
                self.assertGreater(diff, 0.0, f"Parameter {name} did not update!")

        # 6. Test free-running autoregressive generation
        self.model.eval()
        exact_matches = 0
        predictions = []

        for prompt, target in self.train_examples:
            pred = self.model.generate_text(prompt, max_new_tokens=48, temperature=0.0)
            predictions.append((prompt, target, pred))
            if pred.strip() == target.strip():
                exact_matches += 1

        exact_match_rate = exact_matches / len(self.train_examples)

        # Predeclared generation criterion: exact match rate on overfit set must be >= 87.5% (at least 7/8)
        self.assertGreaterEqual(
            exact_match_rate,
            0.875,
            f"Free-running generation exact match rate was {exact_match_rate * 100:.1f}%. Predictions: {predictions}",
        )

        # 7. Checkpoint save, reload, and deterministic inference verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "tiny_overfit.pt"
            save_checkpoint(
                ckpt_path,
                self.model,
                self.config,
                metadata={"init_loss": init_loss, "final_loss": final_loss},
                history=[{"step": 80, "loss": final_loss}],
            )
            reloaded_model, payload = load_checkpoint(ckpt_path, device="cpu")

            for prompt, target in self.train_examples:
                orig_pred = self.model.generate_text(prompt, temperature=0.0)
                reloaded_pred = reloaded_model.generate_text(prompt, temperature=0.0)
                self.assertEqual(orig_pred, reloaded_pred, f"Inference mismatch after reload for prompt {prompt!r}")


if __name__ == "__main__":
    unittest.main()
