from __future__ import annotations

from pathlib import Path
import json


ROWS = [
    {"example_id":"train-001","user_text":"hello","target_text":"Hello. How can I help?","goal":"respond","state":{"mode":"conversation"},"split":"train"},
    {"example_id":"train-002","user_text":"what can you do?","target_text":"I can reason over the available context and help with tasks.","goal":"respond","state":{"mode":"conversation"},"split":"train"},
    {"example_id":"train-003","user_text":"summarize this","target_text":"Please provide the material you want summarized.","goal":"request_input","state":{"mode":"conversation"},"split":"train"},
    {"example_id":"train-004","user_text":"thanks","target_text":"You're welcome.","goal":"respond","state":{"mode":"conversation"},"split":"train"},
    {"example_id":"val-001","user_text":"good morning","target_text":"Good morning. What are you working on?","goal":"respond","state":{"mode":"conversation"},"split":"validation"},
    {"example_id":"val-002","user_text":"help me","target_text":"Tell me what you need help with.","goal":"request_input","state":{"mode":"conversation"},"split":"validation"},
    {"example_id":"test-001","user_text":"bye","target_text":"Goodbye.","goal":"respond","state":{"mode":"conversation"},"split":"test"},
    {"example_id":"test-002","user_text":"I need an example","target_text":"Sure. Tell me the topic or task.","goal":"request_input","state":{"mode":"conversation"},"split":"test"},
]


def main():
    output = Path("training/data/smoke.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in ROWS:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
