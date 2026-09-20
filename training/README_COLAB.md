# Mirror 7 — Colab Run Guide

1. Open training/colab/mirror7_training.ipynb from the GitHub repository in Google Colab.
2. Select a GPU runtime.
3. Run the dependency/setup cell.
4. Run the dataset validation cell.
5. Run the training cell.
6. Run the held-out evaluation cell.
7. Copy the checkpoint and evaluation JSON to persistent storage.
8. Do not connect the checkpoint to the public website until the post-training regression gate passes.

The notebook uses the GitHub repository as the source of code and treats the trained checkpoint as a separate artifact.
