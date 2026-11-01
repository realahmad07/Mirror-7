from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from training.native_brain import PAD
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
    save_checkpoint,
)


class TestNativeConversationV5Correctness(unittest.TestCase):
    """Regression and correctness suite for V5 Input-Conditioned Recurrent Decoder."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = NativeConversationV5Config(
            embedding_dim=32,
            hidden_dim=64,
            state_dim=32,
            attention_dim=32,
            dropout=0.0,
            max_bytes=64,
        )
        self.model = build_native_conversation_v5(self.config)

    def test_vocabulary_constants_and_dimensions(self):
        self.assertEqual(VOCAB_SIZE, 257)
        self.assertEqual(PAD, 256)
        self.assertEqual(BOS, 257)
        self.assertEqual(EOS, 258)
        self.assertEqual(TOTAL_VOCAB_SIZE, 259)
        self.assertEqual(self.model.embedding.num_embeddings, TOTAL_VOCAB_SIZE)
        self.assertEqual(self.model.decoder_embedding.num_embeddings, TOTAL_VOCAB_SIZE)
        self.assertEqual(self.model.head.out_features, TOTAL_VOCAB_SIZE)

    def test_forward_and_backward_passes_finite(self):
        input_ids = torch.tensor([[10, 20, 30, 40], [15, 25, PAD, PAD]], dtype=torch.long)
        input_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 50, 60, EOS], [BOS, 55, EOS, PAD]], dtype=torch.long)

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

    def test_context_influences_recurrent_state_updates(self):
        """CRITICAL V5 PROPERTY: Recurrent hidden state must be directly influenced by context."""
        input_ids_1 = torch.tensor([[10, 20, 30]], dtype=torch.long)
        input_ids_2 = torch.tensor([[90, 80, 70]], dtype=torch.long)
        input_mask = torch.ones((1, 3), dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 50, 60]], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            _, _, hidden1, _ = self.model(input_ids_1, input_mask, target_ids)
            _, _, hidden2, _ = self.model(input_ids_2, input_mask, target_ids)

        # Even with the EXACT SAME target tokens, the final recurrent hidden state
        # MUST differ because context was injected into the recurrent cell at every step!
        self.assertFalse(
            torch.allclose(hidden1, hidden2, atol=1e-4),
            "V5 recurrent hidden state failed to incorporate input context!",
        )

    def test_padding_and_attention_masking(self):
        # Input 1 length 3, Input 2 length 6
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

        # For sample 0, attention at masked positions [3, 4, 5] must be strictly 0.0
        padded_attn = attention[0, :, 3:]
        self.assertTrue(torch.allclose(padded_attn, torch.zeros_like(padded_attn)))

    def test_empty_mask_numerical_stability(self):
        input_ids = torch.full((1, 4), PAD, dtype=torch.long)
        input_mask = torch.zeros((1, 4), dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 10, EOS]], dtype=torch.long)

        logits, state, hidden, attention = self.model(input_ids, input_mask, target_ids)
        self.assertTrue(torch.isfinite(logits).all(), "Logits contain NaN/inf on empty mask!")
        self.assertTrue(torch.isfinite(state).all(), "State contains NaN/inf on empty mask!")
        self.assertTrue(torch.isfinite(hidden).all(), "Hidden contains NaN/inf on empty mask!")
        self.assertTrue(torch.allclose(attention, torch.zeros_like(attention)))

    def test_bos_eos_preservation_in_target_preparation(self):
        targets = ["Short text", "A" * 1000]
        max_bytes = 32
        rows = _prepare_targets(targets, max_bytes)
        for r in rows:
            self.assertEqual(r[0], BOS)
            self.assertEqual(r[-1], EOS)
            self.assertLessEqual(len(r), max_bytes)

    def test_teacher_forced_vs_step_by_step_generation_consistency(self):
        """Verify forward unrolling matches step-by-step autoregressive computation."""
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        input_mask = torch.ones((1, 3), dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 70, 71]], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            fwd_logits, _, _, _ = self.model(input_ids, input_mask, target_ids)

            # Manual step-by-step using encode and cell
            _, enc_mask, state, k, v = self.model.encode(input_ids, input_mask)
            h = self.model.init_decoder_hidden(state)
            step_logits = []
            for t in range(target_ids.shape[1]):
                context, _ = self.model._step_attention(h, k, v, enc_mask)
                tok_emb = self.model.decoder_embedding(target_ids[:, t : t + 1]).squeeze(1)
                h = self.model.cell(torch.cat([tok_emb, context], dim=-1), h)
                fused = self.model.output_norm(torch.cat([h, context], dim=-1))
                logits = self.model.head(fused)
                step_logits.append(logits)
            manual_logits = torch.stack(step_logits, dim=1)

        self.assertTrue(
            torch.allclose(fwd_logits, manual_logits, atol=1e-5),
            "Forward unrolling does not match step-by-step recurrent transition!",
        )

    def test_greedy_and_sampled_generation(self):
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        input_mask = torch.ones((1, 3), dtype=torch.float32)

        # Greedy decoding
        out_greedy = self.model.generate(
            input_ids, input_mask, max_new_tokens=16, temperature=0.0
        )
        self.assertGreater(out_greedy.shape[1], 0)
        self.assertNotIn(PAD, out_greedy[0].tolist()[: out_greedy.shape[1] - 1])
        self.assertNotIn(BOS, out_greedy[0].tolist())

        # Sampled decoding with nucleus and penalties
        out_sampled = self.model.generate(
            input_ids,
            input_mask,
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        self.assertGreater(out_sampled.shape[1], 0)

    def test_batched_generation_with_variable_termination(self):
        input_ids = torch.tensor([[70, 71, 72], [80, 81, PAD]], dtype=torch.long)
        input_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32)

        out = self.model.generate(
            input_ids, input_mask, max_new_tokens=24, temperature=0.0
        )
        self.assertEqual(out.shape[0], 2)
        self.assertGreater(out.shape[1], 0)

    def test_text_generation_interface(self):
        text = self.model.generate_text("Hello Mirror 7 V5!", max_new_tokens=20, temperature=0.0)
        self.assertIsInstance(text, str)
        empty_gen = self.model.generate_text("", max_new_tokens=10, temperature=0.0)
        self.assertIsInstance(empty_gen, str)

    def test_checkpoint_save_and_strict_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_v5_model.pt"
            save_checkpoint(
                ckpt_path,
                self.model,
                self.config,
                metadata={"test_run": True},
                history=[{"epoch": 1, "loss": 2.1}],
            )
            self.assertTrue(ckpt_path.is_file())

            reloaded_model, payload = load_checkpoint(ckpt_path, device="cpu")
            self.assertEqual(payload["format"], "mirror7-native-conversation-v5-input-conditioned")
            self.assertEqual(payload["metadata"]["test_run"], True)

            input_ids = torch.tensor([[100, 101, 102]], dtype=torch.long)
            input_mask = torch.ones((1, 3), dtype=torch.float32)

            orig_out = self.model.generate(input_ids, input_mask, max_new_tokens=12, temperature=0.0)
            reload_out = reloaded_model.generate(input_ids, input_mask, max_new_tokens=12, temperature=0.0)
            self.assertTrue(torch.equal(orig_out, reload_out))


if __name__ == "__main__":
    unittest.main()
