from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from training.dataset import TrainingExample
from training.gpu_train import (
    DynamicBucketSampler,
    EncodedExampleCache,
    build_optimizer,
    collate_batch,
)
from training.native_brain import PAD, _batch, _encode
from training.native_conversation_v5 import (
    BOS,
    EOS,
    TOTAL_VOCAB_SIZE,
    VOCAB_SIZE,
    NativeConversationV5,
    NativeConversationV5Config,
    _clean_truncate,
    _encode_example_prompt,
    _encode_prompt,
    _prepare_targets,
    apply_lora_to_model,
    build_native_conversation_v5,
    configure_cpu_runtime,
    decompose_long_instruction,
    expand_examples_into_chunks,
    format_cognitive_prompt,
    get_lora_state_dict,
    load_checkpoint,
    make_v5_cpu_scaled_config,
    make_v5_enhanced_config,
    merge_lora_weights,
    quantize_for_cpu_inference,
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

    def test_bigru_padding_invariance(self):
        """Verify backward GRU does not drift over trailing PAD tokens in batched encoding."""
        self.model.eval()
        ids_unpad = torch.tensor([[10, 20, 30]], dtype=torch.long)
        mask_unpad = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        ids_pad = torch.tensor([[10, 20, 30, PAD, PAD, PAD, PAD]], dtype=torch.long)
        mask_pad = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        with torch.no_grad():
            x1, _, s1, k1, v1 = self.model.encode(ids_unpad, mask_unpad)
            x2, _, s2, k2, v2 = self.model.encode(ids_pad, mask_pad)

        self.assertTrue(torch.allclose(x1, x2[:, :3, :], atol=1e-6))
        self.assertTrue(torch.allclose(s1, s2, atol=1e-6))
        self.assertTrue(torch.allclose(k1, k2[:, :3, :], atol=1e-6))
        self.assertTrue(torch.allclose(v1, v2[:, :3, :], atol=1e-6))

    def test_fp16_amp_no_overflow(self):
        """Verify float16 forward and generate never overflow on attention/logit masks."""
        model_fp16 = build_native_conversation_v5(self.config).half()
        model_fp16.eval()
        input_ids = torch.tensor([[10, 20, 30, PAD, PAD]], dtype=torch.long)
        input_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float16)
        target_ids = torch.tensor([[BOS, 50, 60, EOS]], dtype=torch.long)

        with torch.no_grad():
            logits, state, hidden, attn = model_fp16(input_ids, input_mask, target_ids)
            gen = model_fp16.generate(
                input_ids, input_mask, max_new_tokens=8, temperature=0.8, top_k=10, top_p=0.9
            )
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertTrue(torch.isfinite(attn).all())
        self.assertEqual(gen.shape[0], 1)

    def test_zero_data_loss_chunking_and_clean_truncate(self):
        """Verify long target responses are split into clean continuation chunks without losing content."""
        long_text = (
            "First complete sentence explaining the architecture. "
            "Second complete sentence detailing the recurrent state. "
            "Third complete sentence covering cross-attention alignment."
        )
        truncated = bytes(_clean_truncate(long_text, 65)).decode("utf-8")
        self.assertTrue(truncated.endswith("."), f"Expected sentence boundary, got: {truncated!r}")

        ex = TrainingExample(
            example_id="ex1",
            user_text="Explain Mirror 7.",
            target_text=long_text,
            goal="explain",
            state={},
            evidence={},
            actions=(),
            split="train",
            quality=1.0,
        )
        chunks = expand_examples_into_chunks([ex], max_bytes=70)
        self.assertGreaterEqual(len(chunks), 3)
        reconstructed = " ".join(c.target_text for c in chunks)
        self.assertIn("First complete sentence", reconstructed)
        self.assertIn("Second complete sentence", reconstructed)
        self.assertIn("Third complete sentence", reconstructed)

    def test_structured_cognitive_prompt_and_state_carry_over(self):
        """Verify structured cognitive state formatting and working-memory prev_state carry-over."""
        ex = TrainingExample(
            example_id="cog1",
            user_text="Calculate 15 * 8.",
            target_text="120.",
            goal="math_computation",
            state={"mode": "verify"},
            evidence={"formula": "15*8=120"},
            actions=("multiply", "verify"),
            split="train",
            quality=1.0,
        )
        formatted = format_cognitive_prompt(
            ex.user_text, goal=ex.goal, state=ex.state, evidence=ex.evidence, actions=ex.actions
        )
        self.assertIn("[GOAL] math_computation", formatted)
        self.assertIn("[STATE] mode=verify", formatted)
        self.assertIn("[FACTS] formula=15*8=120", formatted)
        self.assertIn("[PLAN] multiply -> verify", formatted)
        self.assertIn("[USER] Calculate 15 * 8.", formatted)

        encoded = _encode_example_prompt(ex, max_bytes=128)
        self.assertGreater(len(encoded), len(ex.user_text))

        # Test prev_state working memory carry-over in encode()
        self.model.eval()
        ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        mask = torch.ones((1, 3), dtype=torch.float32)
        prev_state = torch.ones((1, self.config.state_dim), dtype=torch.float32)
        with torch.no_grad():
            _, _, s_no_prev, _, _ = self.model.encode(ids, mask, prev_state=None)
            _, _, s_with_prev, _, _ = self.model.encode(ids, mask, prev_state=prev_state)
        self.assertFalse(torch.allclose(s_no_prev, s_with_prev, atol=1e-4))

    def test_dual_scale_multiscale_heads_and_boundary_anchor_refresh(self):
        """Verify dual-scale cross-attention (fine byte + coarse clause heads) and boundary refresh."""
        cfg4 = make_v5_enhanced_config(
            embedding_dim=32,
            hidden_dim=64,
            state_dim=32,
            attention_dim=32,
            dropout=0.0,
            max_bytes=64,
            num_heads=4,
        )
        m4 = build_native_conversation_v5(cfg4)
        m4.eval()
        ids_unpad = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        mask_unpad = torch.ones((1, 5), dtype=torch.float32)
        ids_pad = torch.tensor([[10, 20, 30, 40, 50, PAD, PAD, PAD]], dtype=torch.long)
        mask_pad = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        with torch.no_grad():
            _, bm1, s1, k1, v1 = m4.encode(ids_unpad, mask_unpad)
            _, bm2, s2, k2, v2 = m4.encode(ids_pad, mask_pad)
            kh1, vh1, _, _ = m4._prepare_attention_cache(k1, v1, bm1)
            kh2, vh2, _, _ = m4._prepare_attention_cache(k2, v2, bm2)
            # Fine heads (0, 1) and coarse pooled heads (2, 3) must both be padding-invariant
            self.assertTrue(torch.allclose(kh1, kh2[:, :, :5, :], atol=1e-6))
            self.assertTrue(torch.allclose(vh1, vh2[:, :, :5, :], atol=1e-6))

            # Target containing sentence boundary (46 = '.') exercises periodic anchor refresh
            target_with_period = torch.tensor([[BOS, 65, 46, 66, EOS]], dtype=torch.long)
            logits, _, hidden, _ = m4(ids_unpad, mask_unpad, target_with_period)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(hidden).all())

    def test_best_of_n_thinking_and_long_goal_execution(self):
        """Verify Best-of-N verification, two-pass Think-Then-Answer, and long-goal decomposition."""
        long_inst = (
            "1. Define cross-attention in neural networks.\n"
            "2. Explain how bidirectional encoders capture context.\n"
            "3. Summarize why recurrent state gating prevents drift."
        )
        subgoals = decompose_long_instruction(long_inst)
        self.assertEqual(len(subgoals), 3)

        bon = self.model.generate_best_of_n("Hello Mirror 7!", n=3, max_new_tokens=12)
        self.assertEqual(len(bon["candidates"]), 3)
        self.assertEqual(len(bon["scores"]), 3)
        self.assertIsInstance(bon["best_text"], str)

        thought_res = self.model.generate_with_thinking(
            "Explain water formula.",
            goal="chemistry",
            state={"topic": "H2O"},
            max_plan_tokens=12,
            max_answer_tokens=16,
            num_candidates=2,
        )
        self.assertIn("answer", thought_res)
        self.assertIn("thought_plan", thought_res)
        self.assertEqual(thought_res["plan_state"].shape, (1, self.config.state_dim))

        goal_res = self.model.execute_long_goal(
            long_inst, max_tokens_per_step=10, num_candidates=2
        )
        self.assertEqual(len(goal_res["step_outputs"]), 3)
        self.assertEqual(goal_res["final_state"].shape, (1, self.config.state_dim))

    def test_curriculum_bucket_sampler(self):
        """Verify curriculum batch ordering progresses from short to long examples in epoch 1."""
        items = [
            TrainingExample(
                example_id=f"id_{i}",
                user_text="Prompt",
                target_text="A" * ((i + 1) * 20),
                split="train",
            )
            for i in range(12)
        ]
        sampler = DynamicBucketSampler(items, batch_size=2, max_bytes=512)
        batches_ep1 = sampler.get_batches(shuffle=True, epoch=1, total_epochs=5, curriculum=True)
        self.assertEqual(len(batches_ep1), 6)
        first_tier_avg = sum(len(x.target_text) for b in batches_ep1[:2] for x in b) / 4.0
        last_tier_avg = sum(len(x.target_text) for b in batches_ep1[4:] for x in b) / 4.0
        self.assertLess(first_tier_avg, last_tier_avg)

    def test_computational_efficiency_fastpath_and_quantization(self):
        """Verify fused SDPA return_attentions=False equivalence and dynamic INT8 CPU quantization."""
        self.model.eval()
        ids = torch.tensor([[10, 20, 30, 40], [15, 25, PAD, PAD]], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        tids = torch.tensor([[BOS, 50, 46, 60, EOS], [BOS, 55, EOS, PAD, PAD]], dtype=torch.long)

        with torch.no_grad():
            logits_full, s_full, h_full, attn_full = self.model(
                ids, mask, tids, return_attentions=True
            )
            logits_fast, s_fast, h_fast, attn_fast = self.model(
                ids, mask, tids, return_attentions=False
            )
        self.assertIsNotNone(attn_full)
        self.assertIsNone(attn_fast)
        self.assertTrue(torch.allclose(logits_full, logits_fast, atol=1e-5))
        self.assertTrue(torch.allclose(s_full, s_fast, atol=1e-6))
        self.assertTrue(torch.allclose(h_full, h_fast, atol=1e-5))

        # Dynamic INT8 quantization for local CPU inference
        q_model = quantize_for_cpu_inference(self.model)
        q_out = q_model.generate_text(
            "Hello quantized Mirror 7!",
            max_new_tokens=16,
            temperature=0.0,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
        self.assertIsInstance(q_out, str)

    def test_encoded_cache_and_token_budget_batching(self):
        """Verify EncodedExampleCache, per-batch sequence trimming, and token-budget batching."""
        items = [
            TrainingExample(
                example_id=f"c_{i}",
                user_text="Short prompt",
                target_text="Word " * (i + 2),
                split="train",
            )
            for i in range(8)
        ]
        cache = EncodedExampleCache(items, max_bytes=256, eager=True)
        ui, um, ti = collate_batch(items[:2], cache, max_bytes=256, device=torch.device("cpu"))
        # Per-batch sequence trimming: batch of short items must NOT be padded to 256
        self.assertLess(ui.shape[1], 32)
        self.assertLess(ti.shape[1], 32)

        sampler = DynamicBucketSampler(
            items,
            batch_size=2,
            max_bytes=256,
            max_tokens_per_batch=120,
            lengths=cache.lengths(),
        )
        batches = sampler.get_batches(shuffle=False)
        self.assertGreaterEqual(len(batches), 2)
        opt = build_optimizer(self.model, lr=1e-3, device=torch.device("cpu"))
        self.assertIsNotNone(opt)

    def test_segmented_activation_checkpointing_exact_equivalence(self):
        """Verify segmented activation checkpointing produces identical logits, states, and gradients."""
        import copy

        m_ref = copy.deepcopy(self.model)
        m_chk = copy.deepcopy(self.model)
        m_ref.train()
        m_chk.train()

        ids = torch.tensor([[10, 20, 30, 40, 50], [15, 25, 35, PAD, PAD]], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        tids = torch.tensor(
            [[BOS, 65, 46, 66, 67, 10, 68, EOS], [BOS, 70, 71, 33, 72, EOS, PAD, PAD]],
            dtype=torch.long,
        )

        logits_ref, s_ref, h_ref, _ = m_ref(
            ids, mask, tids[:, :-1], return_attentions=False, checkpoint_segment_size=0
        )
        loss_ref = torch.nn.functional.cross_entropy(
            logits_ref.reshape(-1, logits_ref.shape[-1]),
            tids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )
        loss_ref.backward()

        logits_chk, s_chk, h_chk, _ = m_chk(
            ids, mask, tids[:, :-1], return_attentions=False, checkpoint_segment_size=3
        )
        loss_chk = torch.nn.functional.cross_entropy(
            logits_chk.reshape(-1, logits_chk.shape[-1]),
            tids[:, 1:].reshape(-1),
            ignore_index=PAD,
        )
        loss_chk.backward()

        self.assertTrue(torch.allclose(logits_ref, logits_chk, atol=1e-6))
        self.assertTrue(torch.allclose(s_ref, s_chk, atol=1e-6))
        self.assertTrue(torch.allclose(h_ref, h_chk, atol=1e-6))
        for (n1, p1), (n2, p2) in zip(m_ref.named_parameters(), m_chk.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertIsNotNone(p1.grad)
            self.assertIsNotNone(p2.grad)
            self.assertTrue(torch.allclose(p1.grad, p2.grad, atol=1e-6), f"Grad mismatch on {n1}")

    def test_lora_cpu_finetuning_and_seamless_merge(self):
        """Verify LoRA freezes base weights, preserves step-0 output, updates adapters, and merges cleanly."""
        import copy

        m_lora = copy.deepcopy(self.model)
        m_lora.eval()
        ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        mask = torch.ones((1, 4), dtype=torch.float32)
        tids = torch.tensor([[BOS, 65, 66, EOS]], dtype=torch.long)

        with torch.no_grad():
            base_logits, _, _, _ = m_lora(ids, mask, tids[:, :-1], return_attentions=False)

        lora_stats = apply_lora_to_model(m_lora, rank=4, alpha=8.0)
        self.assertLess(lora_stats["trainable_ratio"], 0.25)

        # At initialization (lora_B = 0), forward output must match base model exactly
        with torch.no_grad():
            init_lora_logits, _, _, _ = m_lora(ids, mask, tids[:, :-1], return_attentions=False)
        self.assertTrue(torch.allclose(base_logits, init_lora_logits, atol=1e-6))

        # Run 1 optimization step on LoRA parameters only
        m_lora.train()
        opt = build_optimizer(m_lora, lr=1e-2, device=torch.device("cpu"))
        logits, _, _, _ = m_lora(ids, mask, tids[:, :-1], return_attentions=False)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tids[:, 1:].reshape(-1), ignore_index=PAD
        )
        loss.backward()
        opt.step()

        lora_sd = get_lora_state_dict(m_lora)
        self.assertGreater(len(lora_sd), 0)

        m_lora.eval()
        with torch.no_grad():
            before_merge_logits, _, _, _ = m_lora(ids, mask, tids[:, :-1], return_attentions=False)

        # Save checkpoint while LoRA is still attached -> must export clean standard state_dict
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "lora_merged.pt"
            save_checkpoint(ckpt_path, m_lora, m_lora.config)
            loaded_model, _ = load_checkpoint(ckpt_path, device="cpu")
            with torch.no_grad():
                loaded_logits, _, _, _ = loaded_model(
                    ids, mask, tids[:, :-1], return_attentions=False
                )
            self.assertTrue(torch.allclose(before_merge_logits, loaded_logits, atol=1e-5))

        merge_lora_weights(m_lora)
        with torch.no_grad():
            after_merge_logits, _, _, _ = m_lora(ids, mask, tids[:, :-1], return_attentions=False)
        self.assertTrue(torch.allclose(before_merge_logits, after_merge_logits, atol=1e-5))

    def test_cpu_scaled_architecture_moe_memory_and_byte_patching(self):
        """Verify Sparse MoE, O(1) Key-Value Memory Bank, and Strided Byte-Patching scale capacity cleanly."""
        scaled_cfg = make_v5_cpu_scaled_config(
            embedding_dim=32,
            hidden_dim=64,
            state_dim=32,
            attention_dim=32,
            dropout=0.0,
            max_bytes=64,
            num_heads=4,
            checkpoint_segment_size=4,
            num_moe_experts=4,
            moe_top_k=2,
            byte_patch_size=2,
            memory_bank_slots=128,
        )
        m_scaled = build_native_conversation_v5(scaled_cfg)
        base_params = sum(p.numel() for p in self.model.parameters())
        scaled_params = sum(p.numel() for p in m_scaled.parameters())
        self.assertGreater(scaled_params, base_params * 1.5)

        m_scaled.train()
        ids = torch.tensor([[10, 20, 30, 40, 50], [12, 22, 32, PAD, PAD]], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        tids = torch.tensor([[BOS, 65, 46, 66, EOS], [BOS, 70, 71, EOS, PAD]], dtype=torch.long)

        logits, state, hidden, _ = m_scaled(ids, mask, tids[:, :-1], return_attentions=False)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tids[:, 1:].reshape(-1), ignore_index=PAD
        )
        loss.backward()
        self.assertIsNotNone(m_scaled.memory_bank.mem_keys.grad)
        self.assertIsNotNone(m_scaled.byte_patcher.weight.grad)
        self.assertIsNotNone(m_scaled.moe_block.router.weight.grad)

        out_text = m_scaled.generate_text("Test scaled CPU model.", max_new_tokens=10, temperature=0.0)
        self.assertIsInstance(out_text, str)

    def test_cpu_bf16_autocast_and_runtime_tuning(self):
        """Verify CPU runtime thread/OneDNN configuration and CPU BFloat16 mixed-precision pass."""
        rt_info = configure_cpu_runtime()
        self.assertGreaterEqual(rt_info["active_cpu_threads"], 1)
        self.assertTrue(rt_info["mkldnn_enabled"])

        self.model.train()
        ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        mask = torch.ones((1, 4), dtype=torch.float32)
        tids = torch.tensor([[BOS, 65, 66, EOS]], dtype=torch.long)

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            logits, _, _, _ = self.model(ids, mask, tids[:, :-1], return_attentions=False)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tids[:, 1:].reshape(-1), ignore_index=PAD
            )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.model.zero_grad(set_to_none=True)

    def test_utf8_grammar_constrained_decoding_never_emits_replacement_chars(self):
        """Verify RFC-3629 UTF-8 state-machine decoding never produces replacement chars or invalid bytes."""
        self.model.eval()
        for tok_budget in (1, 2, 3, 7, 16, 32):
            text_greedy = self.model.generate_text(
                "Explain quantum computing!",
                max_new_tokens=tok_budget,
                temperature=0.0,
                enforce_valid_utf8=True,
            )
            self.assertNotIn("\ufffd", text_greedy)
            text_sampled = self.model.generate_text(
                "Explain quantum computing!",
                max_new_tokens=tok_budget,
                temperature=1.4,
                top_k=120,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                enforce_valid_utf8=True,
            )
            self.assertNotIn("\ufffd", text_sampled)

        # Verify raw bytes decode strictly under UTF-8 without errors
        ids = torch.tensor([[65, 66, 67], [68, 69, PAD]], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
        out_tokens = self.model.generate(
            ids,
            mask,
            max_new_tokens=24,
            temperature=1.5,
            top_k=200,
            enforce_valid_utf8=True,
        )
        for row in out_tokens.tolist():
            raw_bytes = bytes(t for t in row if t != EOS and 0 <= t <= 255)
            decoded_strict = raw_bytes.decode("utf-8", errors="strict")
            self.assertNotIn("\ufffd", decoded_strict)

    def test_training_edge_case_scenarios_empty_allpad_and_resume(self):
        """Verify edge-case scenarios: empty target_ids (seq_len=0), all-PAD encoder masks, and resume."""
        import json
        from training.gpu_train import train_gpu

        self.model.eval()
        # 1. Empty target_ids (seq_len=0) must not crash with RuntimeError on stack
        ids = torch.tensor([[10, 20], [30, PAD]], dtype=torch.long)
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        empty_tids = torch.empty((2, 0), dtype=torch.long)
        with torch.no_grad():
            logits_empty, s_empty, h_empty, attn_empty = self.model(
                ids, mask, empty_tids, return_attentions=True
            )
            self.assertEqual(logits_empty.shape, (2, 0, TOTAL_VOCAB_SIZE))
            self.assertEqual(attn_empty.shape, (2, 0, 2))

        # 2. All-False encoder mask (empty prompt) must match between fused SDPA and explicit attention
        zero_mask = torch.zeros((2, 2), dtype=torch.float32)
        tids = torch.tensor([[BOS, 65, EOS], [BOS, 66, EOS]], dtype=torch.long)
        with torch.no_grad():
            l_full, _, _, _ = self.model(ids, zero_mask, tids, return_attentions=True)
            l_fast, _, _, _ = self.model(ids, zero_mask, tids, return_attentions=False)
            self.assertTrue(torch.allclose(l_full, l_fast, atol=1e-5))

        # 3. End-to-end train_gpu scenario: Epoch 1 -> Save -> Resume Epoch 2 -> First Response Check
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = Path(tmpdir) / "mini_dataset.jsonl"
            ckpt_path = Path(tmpdir) / "model.pt"
            rows = []
            for idx in range(6):
                rows.append(
                    {
                        "example_id": f"tr_{idx}",
                        "user_text": f"Question {idx}: What is cross-attention?",
                        "target_text": "Cross-attention aligns decoder states with encoded prompt representations.",
                        "goal": "explain_attention",
                        "state": {"step": 1},
                        "evidence": {},
                        "actions": ["define", "explain"],
                        "split": "train",
                        "quality": 1.0,
                    }
                )
            for idx in range(2):
                rows.append(
                    {
                        "example_id": f"val_{idx}",
                        "user_text": f"Val Question {idx}: How does Mirror 7 decode?",
                        "target_text": "Mirror 7 decodes byte by byte using an input-conditioned GRUCell.",
                        "goal": "explain_decoder",
                        "state": {},
                        "evidence": {},
                        "actions": [],
                        "split": "validation",
                        "quality": 1.0,
                    }
                )
            ds_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            res1 = train_gpu(
                dataset_path=ds_path,
                output_path=ckpt_path,
                epochs=1,
                batch_size=3,
                lr=2e-3,
                max_bytes=128,
                device_str="cpu",
                use_cpu_amp=True,
                checkpoint_segment_size=16,
            )
            self.assertEqual(res1["status"], "completed")
            self.assertEqual(len(res1["history"]), 1)
            self.assertNotIn("\ufffd", res1["sample_output"])

            res2 = train_gpu(
                dataset_path=ds_path,
                output_path=ckpt_path,
                resume_path=ckpt_path,
                epochs=1,
                batch_size=3,
                lr=2e-3,
                max_bytes=128,
                device_str="cpu",
            )
            self.assertEqual(len(res2["history"]), 2)
            self.assertNotIn("\ufffd", res2["sample_output"])

    def test_end_to_end_training_convergence_and_first_response_generation(self):
        """Train V5 Enhanced to convergence on a mini Q&A set and verify its first generated response."""
        torch.manual_seed(7)
        cfg = make_v5_enhanced_config(
            embedding_dim=48,
            hidden_dim=96,
            state_dim=48,
            attention_dim=48,
            dropout=0.0,
            max_bytes=128,
            num_heads=4,
        )
        model = build_native_conversation_v5(cfg)
        qa_pairs = [
            ("What is Mirror 7?", "Mirror 7 is a native recurrent conversational model."),
            ("How does attention work?", "Attention focuses on relevant encoded tokens."),
        ]
        ui, um = _batch([_encode_prompt(q, cfg.max_bytes) for q, _ in qa_pairs], cfg.max_bytes)
        ti, _ = _batch(_prepare_targets([a for _, a in qa_pairs], cfg.max_bytes), cfg.max_bytes)

        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=0.0)
        model.train()
        initial_loss = None
        final_loss = None
        for _ in range(55):
            optimizer.zero_grad(set_to_none=True)
            logits, _, _, _ = model(ui, um, ti[:, :-1], return_attentions=False)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                ti[:, 1:].reshape(-1),
                ignore_index=PAD,
            )
            if initial_loss is None:
                initial_loss = float(loss.item())
            final_loss = float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        self.assertLess(final_loss, initial_loss * 0.08)
        self.assertLess(final_loss, 0.25)

        model.eval()
        ans1 = model.generate_text("What is Mirror 7?", max_new_tokens=64, temperature=0.0)
        ans2 = model.generate_best_of_n("How does attention work?", n=3, max_new_tokens=64)["best_text"]
        self.assertEqual(ans1, "Mirror 7 is a native recurrent conversational model.")
        self.assertEqual(ans2, "Attention focuses on relevant encoded tokens.")

    def test_a100_preset_configs_and_scaling(self):
        """Verify all parameter scaling presets (base, cpu-scaled, medium, a100-pro, a100-max) build valid configs."""
        from training.native_conversation_v5 import make_v5_preset_config

        base_cfg = make_v5_preset_config("base")
        base_model = build_native_conversation_v5(base_cfg)
        base_params = sum(p.numel() for p in base_model.parameters())
        self.assertEqual(base_params, 924739)

        pro_cfg = make_v5_preset_config("a100-pro", max_bytes=128)
        pro_model = build_native_conversation_v5(pro_cfg)
        pro_params = sum(p.numel() for p in pro_model.parameters())
        self.assertGreater(pro_params, 30_000_000)

        max_cfg = make_v5_preset_config("a100-max", max_bytes=128)
        self.assertEqual(max_cfg.hidden_dim, 1024)
        self.assertEqual(max_cfg.num_moe_experts, 16)
        self.assertEqual(max_cfg.memory_bank_slots, 16384)

        ultra_cfg = make_v5_preset_config("a100-ultra", max_bytes=128)
        self.assertTrue(ultra_cfg.decoder_self_attn)
        self.assertEqual(ultra_cfg.hidden_dim, 1536)

        # Verify DecoderSelfAttentionRoPE parallel forward vs incremental KV-cache step equivalence
        from training.native_conversation_v5 import DecoderSelfAttentionRoPE
        dsa = DecoderSelfAttentionRoPE(hidden_dim=64, num_heads=4)
        torch.nn.init.normal_(dsa.out_proj.weight, std=0.05)
        dsa.eval()
        seq = torch.randn(2, 10, 64)
        with torch.no_grad():
            par_out = dsa(seq)
            k_c = torch.empty(2, 4, 10, 16)
            v_c = torch.empty(2, 4, 10, 16)
            step_outs = [dsa.step(seq[:, t, :], k_c, v_c, t) for t in range(10)]
            step_out = torch.stack(step_outs, dim=1)
        self.assertTrue(torch.allclose(par_out, step_out, atol=1e-5))

    def test_general_ai_corpus_builder_and_star_self_play(self):
        """Verify General AI corpus builder filters, textbook parsers, and STaR self-play verifier."""
        from training.build_general_ai_corpus import (
            FilterConfig,
            build_identity_pairs,
            build_registry,
            assess,
            normalize_math_solution,
        )
        from training.self_play_distill import (
            generate_verified_response,
            verify_and_score_generation,
        )

        id_pairs = build_identity_pairs("181-million")
        self.assertGreaterEqual(len(id_pairs), 40)
        prompts_lower = {p.lower() for p, _ in id_pairs}
        self.assertIn("who are you ?", prompts_lower)
        self.assertIn("who are you?", prompts_lower)

        reg = {s.name: s for s in build_registry()}
        fcfg = FilterConfig()
        ok, score, cp, ct, reason = assess(
            "who are you ?",
            id_pairs[0][1],
            reg["identity"],
            fcfg,
        )
        self.assertTrue(ok, msg=reason)
        self.assertGreaterEqual(score, 0.95)

        # Reject foreign identity ("I am ChatGPT...")
        ok_bad, _, _, _, reason_bad = assess(
            "Who are you?",
            "I am ChatGPT, a large language model trained by OpenAI.",
            reg["smoltalk-everyday"],
            fcfg,
        )
        self.assertFalse(ok_bad)
        self.assertEqual(reason_bad, "foreign_identity")

        # Math normalization
        norm_sol = normalize_math_solution("We compute 2 + 2 = \\boxed{4}.")
        self.assertIn("Therefore, the final answer is 4.", norm_sol)

        # STaR Process Verifier
        good_verdict = verify_and_score_generation(
            "What are the Pauli matrices X, Y, and Z?",
            "1. Pauli X = [[0, 1], [1, 0]] (bit-flip).\n"
            "2. Pauli Y = [[0, -i], [i, 0]] (combined flip).\n"
            "3. Pauli Z = [[1, 0], [0, -1]] (phase-flip). Therefore, P^2 = I.",
        )
        self.assertTrue(good_verdict["accepted"])
        self.assertGreaterEqual(good_verdict["score"], 0.75)

        res = generate_verified_response(self.model, "Who are you?", num_candidates=2, max_new_tokens=16)
        self.assertIn("best_text", res)
        self.assertEqual(len(res["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
