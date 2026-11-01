from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from training.native_brain import PAD, _batch, _encode
from training.native_conversation_v5 import (
    BOS,
    EOS,
    TOTAL_VOCAB_SIZE,
    VOCAB_SIZE,
    NativeConversationV5,
    NativeConversationV5Config,
    _prepare_targets,
    build_native_conversation_v5,
    load_checkpoint,
    make_v5_enhanced_config,
    save_checkpoint,
)


class TestNativeConversationV5Enhanced(unittest.TestCase):
    """Correctness and regression suite for V5 Enhanced architecture (BiGRU, MHA, Gating, Scheduled Sampling)."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = make_v5_enhanced_config(
            embedding_dim=32,
            hidden_dim=64,
            state_dim=32,
            attention_dim=32,
            dropout=0.0,
            max_bytes=64,
            num_heads=2,
            scheduled_sampling_p=0.2,
        )
        self.model = build_native_conversation_v5(self.config)

    def test_enhanced_architecture_dimensions(self):
        self.assertTrue(self.config.bidirectional_encoder)
        self.assertEqual(self.config.num_heads, 2)
        self.assertTrue(self.config.context_gating)
        self.assertEqual(self.config.scheduled_sampling_p, 0.2)

        # Check submodules
        self.assertTrue(hasattr(self.model, "gate_linear"))
        self.assertTrue(hasattr(self.model, "attn_out"))
        self.assertTrue(self.model.encoder.bidirectional)

    def test_forward_and_backward_passes_finite_with_scheduled_sampling(self):
        input_ids = torch.tensor([[10, 20, 30, 40], [15, 25, PAD, PAD]], dtype=torch.long)
        input_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 50, 60, EOS], [BOS, 55, EOS, PAD]], dtype=torch.long)

        # Test in training mode (exercises scheduled sampling)
        self.model.train()
        logits, state, hidden, attentions = self.model(input_ids, input_mask, target_ids)

        self.assertEqual(logits.shape, (2, 4, TOTAL_VOCAB_SIZE))
        self.assertEqual(state.shape, (2, self.config.state_dim))
        self.assertEqual(hidden.shape, (2, self.config.hidden_dim))
        self.assertEqual(attentions.shape, (2, 4, 4))

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, TOTAL_VOCAB_SIZE),
            target_ids.reshape(-1),
            ignore_index=PAD,
        )
        self.assertTrue(torch.isfinite(loss).item())

        self.model.zero_grad()
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Param {name} received no gradient!")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Param {name} gradient has NaN/inf!")

    def test_multi_head_attention_mask_isolation(self):
        # Sample 0 length 3, Sample 1 length 6
        input_ids = torch.tensor(
            [[10, 20, 30, PAD, PAD, PAD], [11, 21, 31, 41, 51, 61]],
            dtype=torch.long,
        )
        input_mask = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        target_ids = torch.tensor([[BOS, 50], [BOS, 55]], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            _, _, _, attention = self.model(input_ids, input_mask, target_ids)

        # For sample 0, attention at masked positions [3, 4, 5] must be strictly 0.0 across all heads
        padded_attn = attention[0, :, 3:]
        self.assertTrue(torch.allclose(padded_attn, torch.zeros_like(padded_attn)))

    def test_recurrent_context_gate_is_wired_into_decoder(self):
        self.assertTrue(self.config.recurrent_context_gating)
        self.assertTrue(hasattr(self.model, "recurrent_gate_linear"))
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        input_mask = torch.ones((1, 3), dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 50, 60]], dtype=torch.long)
        self.model.train()
        logits, _, _, _ = self.model(input_ids, input_mask, target_ids)
        logits.sum().backward()
        grad = self.model.recurrent_gate_linear.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_configured_dropout_is_active_in_training_mode(self):
        config = make_v5_enhanced_config(
            embedding_dim=16, hidden_dim=32, state_dim=16,
            attention_dim=16, num_heads=2, dropout=0.5,
            max_bytes=32, scheduled_sampling_p=0.0,
        )
        model = build_native_conversation_v5(config)
        model.train()
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        input_mask = torch.ones((1, 4), dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 50, 60]], dtype=torch.long)
        first = model(input_ids, input_mask, target_ids)[0]
        second = model(input_ids, input_mask, target_ids)[0]
        self.assertFalse(torch.equal(first, second))

    def test_pre_recurrent_gate_checkpoint_remains_strict_loadable(self):
        legacy_config = NativeConversationV5Config(
            embedding_dim=32, hidden_dim=64, state_dim=32,
            attention_dim=32, dropout=0.0, max_bytes=64,
            bidirectional_encoder=True, num_heads=2, context_gating=True,
            scheduled_sampling_p=0.0,
        )
        legacy_model = build_native_conversation_v5(legacy_config)
        self.assertFalse(legacy_config.recurrent_context_gating)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "legacy_v5.pt"
            save_checkpoint(path, legacy_model, legacy_config)
            loaded, _ = load_checkpoint(path)
            self.assertFalse(loaded.config.recurrent_context_gating)

    def test_context_gating_active_range(self):
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        input_mask = torch.ones((1, 3), dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 50, 60]], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            logits, state, hidden, _ = self.model(input_ids, input_mask, target_ids)
        self.assertTrue(torch.isfinite(logits).all())

    def test_enhanced_tiny_overfit(self):
        """Verify enhanced architecture achieves >90% loss reduction and high exact match on tiny set."""
        train_examples = [
            ("hello", "Hello!"),
            ("how are you", "I am good."),
            ("what is mirror 7", "Mirror 7 is an agent."),
            ("who are you", "I am your AI assistant."),
            ("summarize", "Ready to summarize."),
            ("help", "I can assist you."),
            ("repeat", "Echoing your input."),
            ("bye", "Goodbye!"),
        ]
        ui, um = _batch(
            [_encode(x[0], self.config.max_bytes) for x in train_examples],
            self.config.max_bytes,
        )
        ti, _ = _batch(
            _prepare_targets([x[1] for x in train_examples], self.config.max_bytes),
            self.config.max_bytes,
        )

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
            reduction_pct, 90.0, f"Enhanced loss reduction {reduction_pct:.1f}% below 90%!"
        )
        self.assertLess(final_loss, 0.10, f"Final loss {final_loss:.4f} not below 0.10!")

        # Generation check
        self.model.eval()
        exact_matches = 0
        for prompt, target in train_examples:
            pred = self.model.generate_text(prompt, max_new_tokens=32, temperature=0.0).strip()
            if pred == target:
                exact_matches += 1

        exact_match_pct = 100.0 * (exact_matches / len(train_examples))
        self.assertGreaterEqual(
            exact_match_pct, 87.5, f"Exact match {exact_match_pct:.1f}% below 87.5%!"
        )

    def test_checkpoint_roundtrip_enhanced(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_v5_enhanced.pt"
            save_checkpoint(
                ckpt_path,
                self.model,
                self.config,
                metadata={"enhanced": True},
            )
            self.assertTrue(ckpt_path.is_file())

            reloaded_model, payload = load_checkpoint(ckpt_path, device="cpu")
            self.assertTrue(reloaded_model.config.bidirectional_encoder)
            self.assertEqual(reloaded_model.config.num_heads, 2)
            self.assertTrue(reloaded_model.config.context_gating)

            input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
            input_mask = torch.ones((1, 3), dtype=torch.float32)

            orig_out = self.model.generate(input_ids, input_mask, max_new_tokens=10, temperature=0.0)
            reload_out = reloaded_model.generate(input_ids, input_mask, max_new_tokens=10, temperature=0.0)
            self.assertTrue(torch.equal(orig_out, reload_out))


if __name__ == "__main__":
    unittest.main()
