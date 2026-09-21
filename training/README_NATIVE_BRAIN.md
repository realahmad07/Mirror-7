# Mirror 7 Native Brain — Experiment 1

Phase 367 begins a new research track: Mirror learns its own internal semantic/state representation without Qwen, a pretrained encoder, or a conventional next-byte generation objective.

The first experiment trains one shared byte encoder on user/response pairs with two coupled objectives:
1. In-batch contrastive alignment: matching input/response states should be close while other responses are negatives.
2. State-transition prediction: the input state predicts the corresponding response state.

This is deliberately non-generative. The first question is whether a small Mirror-native network can learn useful internal state from experience.

Research motivation: TinyStories showed that task-appropriate data can make sub-10M models produce coherent behavior; latent-prediction work explores learning representations rather than reconstructing every raw observation; Mamba demonstrates that sequence modeling can use selective recurrent state without Transformer attention. Sources are linked in the project research notes.

Colab command:
python -m training.native_brain --dataset /content/drive/MyDrive/Mirror7/data/mirror_oasst2_curated_v1.jsonl --output /content/drive/MyDrive/Mirror7/mirror7_native_brain_v1.pt --epochs 5 --batch-size 64 --max-bytes 512

Success gate:
- validation loss improves reproducibly;
- matching input/response cosine similarity separates from random pairs;
- response retrieval Top-1 beats the random baseline by a substantial margin;
- transition prediction beats a shuffled control;
- results reproduce on an unseen seed.

Loss alone is not a success criterion. If the gate fails, stop and change the learning mechanism before scaling.

Roadmap:
367 semantic/state learning
368 learned state-transition dynamics
369 memory retrieval and consolidation
370 native decoder conditioned on Mirror state
371 joint brain + decoder training
372+ compare GRU, selective-state-space and compact-attention variants

Qwen is optional for evaluation/distillation only; it is not required by the native brain.
