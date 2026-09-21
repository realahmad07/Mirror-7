# Mirror 7

> Open-source research architecture for explicit state, reasoning, memory, planning, action, and learned language realization.

<div align="center">

[![CI](https://github.com/realahmad07/Mirror-7/actions/workflows/ci.yml/badge.svg)](https://github.com/realahmad07/Mirror-7/actions/workflows/ci.yml)
[![Training CI](https://github.com/realahmad07/Mirror-7/actions/workflows/training-native.yml/badge.svg)](https://github.com/realahmad07/Mirror-7/actions/workflows/training-native.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](./pyproject.toml)

</div>

## What Mirror 7 is

Mirror 7 is an open-source research project exploring whether useful machine intelligence can be built from explicit, inspectable mechanisms for representation, state, prediction, discrepancy handling, reasoning, memory, planning, action, and learning rather than treating next-token prediction as the entire cognitive architecture.

The repository now has two intentionally separated layers:

| Layer | Purpose |
|---|---|
| Mirror runtime / backend | Structured state, reasoning, planning, memory, actions, persistence, evaluation boundaries, HTTP service, and response-generation interfaces. |
| Learned research | Native semantic-state experiments and an optional learned language realization layer. |

## Read the project as pages

| Start here | Engineering | Research |
|---|---|---|
| [Project overview](docs/01-project-overview.md) | [Backend](docs/03-backend.md) | [Training](docs/04-training.md) |
| [Architecture](docs/02-system-architecture.md) | [Testing and CI](docs/06-testing-ci.md) | [Evaluation](docs/05-evaluation.md) |
| [Repository map](docs/07-repository-organization.md) | [Limitations](docs/08-limitations.md) | [Experiments 1–7](docs/research/README.md) |
| [Documentation index](docs/README.md) | [Contributing](CONTRIBUTING.md) | [Roadmap](docs/09-roadmap.md) |

## Current status

| Area | State | Meaning |
|---|:---:|---|
| Core research corpus | ✅ | Phase implementations and acceptance records are preserved. |
| Backend service | ✅ | Sessions, persistence, actions, HTTP API, metrics, and response generation are implemented. |
| Response-generation boundary | ✅ | Contract, adapter, lifecycle, configuration, validation, and E2E layers exist. |
| Pretrained response realizer | ✅ | Phase 366 provides an optional language realization component. |
| Native semantic-state research | ✅ | Experiments 1–7 are recorded and replicated on bounded benchmarks. |
| Scratch language-model route | ⚠️ | 10.57M-class ByteGRU trained, but free-running held-out generation remained weak. |
| Public scientific claim | ⚠️ | This is a research system, not evidence of AGI or a ChatGPT-level standalone model. |

## Training snapshot

| Item | Recorded value |
|---|---|
| Curated corpus | mirror_oasst2_curated_v1.jsonl |
| Total examples | **14,627** |
| Train / validation / test | **11,703 / 1,455 / 1,469** |
| Dataset fingerprint | e7d66d84725bfc9347ea7b8656bfb815625f80adc81cd05d1ea868e6585c4715a |
| Scratch target | **10,574,855 parameters** |
| Scratch model | ByteGRU, 256 embedding, 1024 hidden, 2 layers, 0.10 dropout |
| Native-brain default | 96 embedding, 256 hidden, 2 layers, 128 state |
| Training hardware | Google Colab, NVIDIA A100-SXM4-40GB |
| Rule | Quality over quantity; no blind scaling after a failed hypothesis. |

## Native-brain results

| Experiment | Verified result |
|---|---|
| Exp 1 | Three-seed semantic-state signal: Top-1 14.27%, Top-5 25.47%, semantic margin 0.2209, transition margin 0.1150. |
| Exp 2 | Transition-heavy objective raised transition margin to 0.1605 mean, while Top-1 fell to 12.93%. |
| Exp 3 | Explicit state transformations became learnable on the defined synthetic benchmark. |
| Exp 4B | Transformation-specific contrastive objective reached 87.18% held-out Top-1 on the defined compositional benchmark. |
| Exp 5 | Variable-length chains reached 60.71% mean accuracy across three seeds. |
| Exp 6 | Opaque learned operations exposed a depth/generalization bottleneck. |
| Exp 7 | Residual transition matched the Exp6 length-5 baseline at 63.33% and slightly improved unseen-concept mean to 65.56%; not a decisive overall win. |

## What failed

The first training cycle also tested 1.7M and 10.57M scratch autoregressive models, scheduled sampling, response plans, curricula, semantic lexical plans, BPE-GRU, a small non-autoregressive Transformer, retrieval, reranking, prompt-to-prompt retrieval, and explicit response-state designs.

The important negative result is consistent across diagnostics: supervised/teacher-forced learning was healthy enough to show prompt sensitivity and tiny-set overfitting, but free-running generation remained weak. That is now treated as a research boundary rather than a reason to hide the result or blindly increase parameter count.

## Backend surface

The backend is UI-independent and includes session lifecycle, persistence, allow-listed actions, structured realization contracts, replaceable response-model adapters, output validation, metrics, health/status endpoints, and an HTTP API for the external UI.

## Local commands

~~~bash
python -m pytest -q mirror7_backend
python -m compileall -q mirror7_backend training
python -m pytest -q training/test_training_foundation.py
python -m build
~~~

The complete historical regression remains available through the manual Deep Regression GitHub Actions workflow.

## Deployment

~~~bash
python -m mirror7_backend.http_server
~~~

or:

~~~bash
docker build -t mirror7-backend .
docker run --rm -p 8787:8787 mirror7-backend
~~~

## Research boundary

Mirror 7 should currently be described as an open-source research AI system with a structured reasoning/state architecture, an HTTP backend boundary, and a validated experimental native semantic-state subsystem, with an optional learned language realization layer.

It should not currently be described as AGI, human-level intelligence, or a proven standalone foundation model.
