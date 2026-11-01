from __future__ import annotations

import math
from pathlib import Path
import unittest
import torch

from training.native_brain import PAD, VOCAB_SIZE, _batch, _encode
from training.native_conversation import (
    BOS,
    EOS,
    TOTAL_VOCAB_SIZE,
    NativeConversation,
    NativeConversationConfig,
    _prepare_targets,
    build_native_conversation,
    load_checkpoint,
    save_checkpoint,
)


class TestNativeConversationCorrectness(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = NativeConversationConfig(
            embedding_dim=32,
            hidden_dim=64,
            state_dim=48,
            layers=1,
            dropout=0.0,
            max_bytes=64,
            attention_dim=64,
        )
        self.model = build_native_conversation(self.config)

    def test_vocabulary_constants_and_dimensions(self):
        self.assertEqual(PAD, 256)
        self.assertEqual(VOCAB_SIZE, 257)
        self.assertEqual(BOS, 257)
        self.assertEqual(EOS, 258)
        self.assertEqual(TOTAL_VOCAB_SIZE, 259)

        # Check embedding and output dimensions
        self.assertEqual(self.model.embedding.num_embeddings, TOTAL_VOCAB_SIZE)
        self.assertEqual(self.model.decoder_embedding.num_embeddings, TOTAL_VOCAB_SIZE)
        self.assertEqual(self.model.head.out_features, TOTAL_VOCAB_SIZE)
        self.assertEqual(self.model.embedding.padding_idx, PAD)
        self.assertEqual(self.model.decoder_embedding.padding_idx, PAD)

    def test_forward_pass_and_tensor_shapes(self):
        batch_size = 3
        seq_len_in = 10
        seq_len_out = 8

        input_ids = torch.randint(0, 255, (batch_size, seq_len_in))
        input_mask = torch.ones((batch_size, seq_len_in))
        target_ids = torch.randint(0, 255, (batch_size, seq_len_out))

        logits, state, hidden, attention = self.model(
            input_ids, input_mask, target_ids
        )

        self.assertEqual(logits.shape, (batch_size, seq_len_out, TOTAL_VOCAB_SIZE))
        self.assertEqual(state.shape, (batch_size, self.config.state_dim))
        self.assertEqual(
            hidden.shape, (self.config.layers, batch_size, self.config.hidden_dim)
        )
        self.assertEqual(attention.shape, (batch_size, seq_len_out, seq_len_in))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(attention).all())

    def test_all_parameters_receive_gradients_including_state(self):
        self.model.train()
        input_ids = torch.randint(0, 255, (2, 8))
        input_mask = torch.ones((2, 8))
        target_ids = torch.randint(0, 255, (2, 6))

        logits, _, _, _ = self.model(input_ids, input_mask, target_ids[:, :-1])
        labels = target_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=PAD,
        )
        self.assertTrue(torch.isfinite(loss))

        loss.backward()

        # Check that EVERY single trainable parameter has a valid, finite gradient
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter '{name}' did not receive gradients!",
                )
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Parameter '{name}' has non-finite gradients!",
                )

        # Specifically verify self.state and self.decoder_init
        self.assertIsNotNone(self.model.state.weight.grad)
        self.assertIsNotNone(self.model.decoder_init.weight.grad)
        self.assertGreater(self.model.state.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.model.decoder_init.weight.grad.abs().sum().item(), 0.0)

    def test_consistent_decoder_initialization(self):
        input_ids = torch.randint(0, 255, (2, 10))
        input_mask = torch.ones((2, 10))

        # Forward initialization
        _, _, state, _ = self.model.encode(input_ids, input_mask)
        expected_h0 = self.model.init_decoder_hidden(state)

        # In generate
        self.model.eval()
        with torch.no_grad():
            _, _, gen_state, _ = self.model.encode(input_ids, input_mask)
            gen_h0 = self.model.init_decoder_hidden(gen_state)

        self.assertTrue(torch.equal(expected_h0, gen_h0))

    def test_bos_eos_preservation_in_target_preparation(self):
        max_bytes = 16
        long_text = "This is a very long sentence that exceeds max_bytes."
        exact_text = "a" * 14
        empty_text = ""

        rows = _prepare_targets([long_text, exact_text, empty_text], max_bytes)

        # Each row must start with BOS and end with EOS
        for row in rows:
            self.assertEqual(row[0], BOS)
            self.assertEqual(row[-1], EOS)
            self.assertLessEqual(len(row), max_bytes)

        # When batched, max_len must be <= max_bytes and EOS must not be truncated
        ti, tm = _batch(rows, max_bytes)
        self.assertLessEqual(ti.shape[1], max_bytes)

        for i in range(len(rows)):
            row_tokens = ti[i].tolist()
            self.assertEqual(row_tokens[0], BOS)
            self.assertIn(EOS, row_tokens)
            eos_idx = row_tokens.index(EOS)
            # All tokens after EOS must be PAD
            for pad_tok in row_tokens[eos_idx + 1 :]:
                self.assertEqual(pad_tok, PAD)

    def test_padding_and_attention_masking(self):
        # Input 1 length 4, Input 2 length 8
        input_ids = torch.tensor(
            [
                [10, 20, 30, 40, PAD, PAD, PAD, PAD],
                [11, 21, 31, 41, 51, 61, 71, 81],
            ],
            dtype=torch.long,
        )
        input_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        target_ids = torch.tensor([[BOS, 50, 60], [BOS, 55, 65]], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            _, _, _, attention = self.model(input_ids, input_mask, target_ids)

        # For sample 0, attention at positions 4, 5, 6, 7 must be strictly 0.0
        sample0_padded_attn = attention[0, :, 4:]
        self.assertTrue(torch.allclose(sample0_padded_attn, torch.zeros_like(sample0_padded_attn)))

        # Also verify that changing padded values does NOT affect encoder states or attention
        corrupted_ids = input_ids.clone()
        corrupted_ids[0, 4:] = torch.tensor([99, 101, 102, 103])
        with torch.no_grad():
            enc1, _, state1, _ = self.model.encode(input_ids, input_mask)
            enc2, _, state2, _ = self.model.encode(corrupted_ids, input_mask)

        # Valid encoder positions [0..3] and pooled state must be identical
        self.assertTrue(torch.allclose(enc1[0, :4], enc2[0, :4]))
        self.assertTrue(torch.allclose(state1[0], state2[0]))

    def test_empty_mask_numerical_stability(self):
        # Input where all tokens are masked out
        input_ids = torch.full((1, 8), PAD, dtype=torch.long)
        input_mask = torch.zeros((1, 8), dtype=torch.float32)
        target_ids = torch.tensor([[BOS, 10, EOS]], dtype=torch.long)

        logits, state, _, attention = self.model(input_ids, input_mask, target_ids)

        self.assertTrue(torch.isfinite(logits).all(), "Logits contain NaN/inf on empty mask!")
        self.assertTrue(torch.isfinite(state).all(), "State contains NaN/inf on empty mask!")
        self.assertTrue(torch.isfinite(attention).all(), "Attention contains NaN/inf on empty mask!")
        self.assertTrue(torch.allclose(attention, torch.zeros_like(attention)))

    def test_greedy_and_sampled_generation(self):
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        input_mask = torch.ones((1, 3), dtype=torch.float32)

        # Greedy decoding (temperature <= 0)
        out_greedy = self.model.generate(
            input_ids, input_mask, max_new_tokens=16, temperature=0.0
        )
        self.assertGreater(out_greedy.shape[1], 0)
        self.assertNotIn(PAD, out_greedy[0].tolist()[:out_greedy.shape[1] - 1])
        self.assertNotIn(BOS, out_greedy[0].tolist())

        # Determinism of greedy decoding
        out_greedy2 = self.model.generate(
            input_ids, input_mask, max_new_tokens=16, temperature=0.0
        )
        self.assertTrue(torch.equal(out_greedy, out_greedy2))

        # Sampled decoding (temperature > 0)
        out_sampled = self.model.generate(
            input_ids, input_mask, max_new_tokens=16, temperature=0.8, top_k=20
        )
        self.assertGreater(out_sampled.shape[1], 0)
        self.assertNotIn(BOS, out_sampled[0].tolist())

    def test_batched_generation_with_variable_termination(self):
        input_ids = torch.tensor(
            [[70, 71, 72], [80, 81, PAD]], dtype=torch.long
        )
        input_mask = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32
        )

        out = self.model.generate(
            input_ids, input_mask, max_new_tokens=24, temperature=0.0
        )
        self.assertEqual(out.shape[0], 2)
        self.assertGreater(out.shape[1], 0)

    def test_text_generation_interface_and_utf8(self):
        text = self.model.generate_text("Hello Mirror 7!", max_new_tokens=20, temperature=0.0)
        self.assertIsInstance(text, str)

        empty_gen = self.model.generate_text("", max_new_tokens=10, temperature=0.0)
        self.assertIsInstance(empty_gen, str)

    def test_checkpoint_save_and_strict_reload(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pt"
            save_checkpoint(
                ckpt_path,
                self.model,
                self.config,
                metadata={"test_run": True},
                history=[{"epoch": 1, "loss": 2.5}],
            )
            self.assertTrue(ckpt_path.is_file())

            reloaded_model, payload = load_checkpoint(ckpt_path, device="cpu")
            self.assertEqual(payload["format"], "mirror7-native-conversation-v3-cross-attention")
            self.assertEqual(payload["metadata"]["test_run"], True)
            self.assertEqual(len(payload["history"]), 1)

            # Test inference consistency between original and reloaded
            input_ids = torch.tensor([[100, 101, 102]], dtype=torch.long)
            input_mask = torch.ones((1, 3), dtype=torch.float32)

            orig_out = self.model.generate(input_ids, input_mask, max_new_tokens=12, temperature=0.0)
            reload_out = reloaded_model.generate(input_ids, input_mask, max_new_tokens=12, temperature=0.0)
            self.assertTrue(torch.equal(orig_out, reload_out))

    def test_generation_penalties_and_nucleus_sampling(self):
        input_ids = torch.tensor([[105, 106, 107]], dtype=torch.long)
        input_mask = torch.ones((1, 3), dtype=torch.float32)

        # Test min_new_tokens prevents immediate termination
        out_min = self.model.generate(
            input_ids, input_mask, max_new_tokens=10, min_new_tokens=5, temperature=0.0
        )
        self.assertGreaterEqual(out_min.shape[1], 5)

        # Test repetition penalty and n-gram blocking execute cleanly
        out_pen = self.model.generate(
            input_ids,
            input_mask,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
        self.assertGreater(out_pen.shape[1], 0)

        # Test generate_text with penalties
        text = self.model.generate_text(
            "What is the system?",
            max_new_tokens=25,
            min_new_tokens=3,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
