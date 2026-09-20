from __future__ import annotations

import argparse
import json

try:
    from .dataset import load_jsonl, summarize, validate_split_isolation
except ImportError:
    # Support both:
    #   python -m training.validate_dataset
    # and:
    #   python training/validate_dataset.py
    from dataset import load_jsonl, summarize, validate_split_isolation


def main():
    parser = argparse.ArgumentParser(description="Validate Mirror 7 training JSONL.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    examples = load_jsonl(args.dataset)
    validate_split_isolation(examples)
    print(json.dumps(summarize(examples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
