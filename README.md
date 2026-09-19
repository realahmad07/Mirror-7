# MIRROR 7

> **An open-source research architecture for explicit representation, state, prediction, reasoning, memory, planning, and self-hosted computation.**

[![Bootstrap](https://img.shields.io/badge/Bootstrap-COMPLETE-2ea44f?style=for-the-badge)](./BOOTSTRAP_PROGRESS.md)
[![Phase 31](https://img.shields.io/badge/Phase%2031-PASS-2ea44f?style=for-the-badge)](./phase31_bootstrap)
[![Phase 32](https://img.shields.io/badge/Phase%2032-PASS-2ea44f?style=for-the-badge)](./phase32_prediction)
[![Phase 54](https://img.shields.io/badge/Phase%2054-18%2F18%20PASS-2ea44f?style=for-the-badge)](./INDEPENDENT_EVALUATION_2026-09-19.md)
[![Phase 55](https://img.shields.io/badge/Phase%2055-27%2F27%20PASS-2ea44f?style=for-the-badge)](./PHASE55_ACCEPTANCE.md)
[![Phase 56](https://img.shields.io/badge/Phase%2056-PASS-2ea44f?style=for-the-badge)](./PHASE56_ACCEPTANCE.md)
[![Phase 57](https://img.shields.io/badge/Phase%2057-PASS-2ea44f?style=for-the-badge)](./PHASE57_ACCEPTANCE.md)
[![Phase 58](https://img.shields.io/badge/Phase%2058-PASS-2ea44f?style=for-the-badge)](./PHASE58_ACCEPTANCE.md)
[![Phase 59](https://img.shields.io/badge/Phase%2059-PASS-2ea44f?style=for-the-badge)](./PHASE59_ACCEPTANCE.md)
[![Phase 60](https://img.shields.io/badge/Phase%2060-PASS-2ea44f?style=for-the-badge)](./PHASE60_ACCEPTANCE.md)
[![Phase 61](https://img.shields.io/badge/Phase%2061-PASS-2ea44f?style=for-the-badge)](./PHASE61_ACCEPTANCE.md)
[![Phase 62](https://img.shields.io/badge/Phase%2062-PASS-2ea44f?style=for-the-badge)](./PHASE62_ACCEPTANCE.md)
[![Phase 63](https://img.shields.io/badge/Phase%2063-PASS-2ea44f?style=for-the-badge)](./PHASE63_ACCEPTANCE.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge)](./LICENSE)

---

## 🧭 What is Mirror 7?

Mirror 7 is a research project exploring an alternative route toward machine intelligence:

```text
raw observation
      ↓
discover structure
      ↓
stable state
      ↓
temporal identity
      ↓
transition memory
      ↓
prediction
      ↓
discrepancy
      ↓
causal / world-model update
      ↓
goals + reasoning
      ↓
planning
      ↓
action
      ↓
observe result
      ↺
learn / revise / remember
```

The project emphasizes **explicit, inspectable mechanisms** wherever the research question permits, rather than assuming that intelligence must be implemented as next-token prediction.

> **Important boundary:** a passing phase demonstrates the tested mechanism and its acceptance criteria. It does **not** by itself prove AGI, human-level intelligence, consciousness, or general-world competence.

---

## 🟢 Current Project Status

### Verified boundary

| Area | Current state |
|---|:---:|
| Self-hosting / compiler bootstrap | ✅ Complete |
| Representation discovery | ✅ Verified |
| State + temporal identity | ✅ Verified |
| Transition memory + prediction | ✅ Verified |
| Causal reasoning | ✅ Verified |
| Reasoning / planning mechanism | ✅ Verified |
| Action + closed loop | ✅ Verified |
| Memory | ✅ Verified |
| World model | ✅ Verified |
| Composition | ✅ Verified |
| Hierarchical planning | ✅ Verified |
| Tools | ✅ Verified |
| Language grounding | ✅ Verified |
| Counterfactuals | ✅ Verified |
| Continual learning | ✅ Verified |
| Meta-reasoning | ✅ Verified |
| Robustness / transfer | ✅ Verified |
| Full integration | ✅ Verified |
| Advanced reasoning | ✅ Verified |
| Open-ended black-box learner | ✅ Implemented |
| Phase 53 blind benchmark | ✅ 21/21 |
| Phase 54 locked fresh evaluation | ✅ 18/18 |
| Phase 54 held-out | ✅ 9/9 |
| Phase 54 invalid actions | ✅ 0 |
| Phase 55 representation independence | ✅ 27/27 |
| Phase 56 raw concept acquisition | ✅ PASS |
| Phase 57 cross-view concept acquisition | ✅ PASS |
| Phase 58 hierarchical concept abstraction | ✅ PASS |
| Phase 59 predictive concept learning | ✅ PASS |
| Phase 60 long-horizon world model | ✅ PASS |
| Phase 61 predictive closed-loop autonomy | ✅ PASS |
| Phase 62 partial observability + active information | ✅ PASS |
| Phase 63 nonstationary world + experiment design | ✅ PASS |

### Current frontier

Phases 58–61 are now complete at their bounded acceptance boundaries: hierarchical abstraction **8/8**, predictive concept learning **8/8**, long-horizon world modeling **8/8**, and predictive closed-loop autonomy **9/9**. Phase 61 integrates the Phase 58 hierarchy, Phase 59 predictor, and Phase 60 world model inside an online goal-directed control loop. Phase 63 adds nonstationary-regime tracking, repeated-mismatch drift detection, and autonomous experiment selection from competing transition hypotheses. The next research boundary is open-ended experiment design, richer changing environments, longer unseen tasks, and independent external evaluation.

---

## 🧠 The Mirror 7 Capability Stack

```mermaid
flowchart TD
    A["RAW OBSERVATION"] --> B["DISCOVER STRUCTURE<br/>Phase 31"]
    B --> C["STABLE STATE"]
    C --> D["TEMPORAL IDENTITY"]
    D --> E["TRANSITION MEMORY<br/>Phase 32"]
    E --> F["PREDICTION"]
    F --> G{"DISCREPANCY?"}
    G -- "No" --> H["ROLLOUT / CONTINUE"]
    G -- "Yes" --> I["UPDATE MODEL"]
    I --> J["CAUSAL STRUCTURE<br/>Phase 33"]
    J --> K["GOALS + REASONING"]
    K --> L["PLANNING"]
    L --> M["ACTION"]
    M --> N["OBSERVE RESULT"]
    N --> F
```

The later capability layers extend the same loop with:

```text
memory
  + world model
  + composition
  + hierarchy
  + tools
  + language grounding
  + counterfactual simulation
  + continual learning
  + meta-reasoning
  + robustness
  + transfer
```

---

## 🔬 What has actually been demonstrated

### 1. Computational substrate / bootstrap

Phases 24–30 established the self-hosting compiler substrate:

```mermaid
flowchart LR
    A["MIRR source"] --> B["Compiler A"]
    B --> C["Compiler B"]
    C --> D["Executable artifact"]
    D --> E["Independent rebuild"]
    E --> F{"Byte-identical?"}
    F -- "Yes" --> G["Bootstrap COMPLETE"]
    F -- "No" --> H["FAIL / investigate"]
```

Recorded reproducibility identifiers:

- Primary artifact SHA256: `ec48f82db766b3fa4bbcad83bc5f9193aedb2212329122e8b380a1a68d3a5c50`
- Independent rebuild SHA256: `ec48f82db766b3fa4bbcad83bc5f9193aedb2212329122e8b380a1a68d3a5c50`
- Fixed-point compiled-B SHA256: `4d395c63b6e4364fcad2f198e74e305b639996cc649bbf58d66bb2e46e597d25`

See:
- [Bootstrap progress](./BOOTSTRAP_PROGRESS.md)
- [Detailed status](./STATUS.md)

### 2. Representation → prediction

Phases 31–32 established:

```text
raw observation
      ↓
structural discovery
      ↓
stable state
      ↓
temporal identity
      ↓
learned transition
      ↓
next-state prediction
      ↓
discrepancy
      ↓
online update
```

Phase 31 acceptance: **37/37 tests passed**.

Phase 32 covers multi-seed validation, held-out cases, ambiguity, online learning, rollout, perturbation testing, serialization, integration, and regression.

### 3. Causal → planning → action

The Phase 33–51 capability line demonstrates explicit mechanisms for:

```text
CAUSE
 ↓
GOAL
 ↓
REASON
 ↓
PLAN
 ↓
ACT
 ↓
MEMORY
 ↓
WORLD MODEL
 ↓
COUNTERFACTUAL
 ↓
REVISE
```

See [Phase 34 acceptance](./PHASE34_ACCEPTANCE.md) and [Phase 35–51 verification](./PHASE35_51_VERIFICATION.md).

### 4. Open-ended black-box learning

Phase 52 provides a learner that receives only:

```text
observation
goal
legal actions
step limit
transition feedback
```

It does not receive task-family labels, hidden parameters, or expected transitions.

The learner discovers action consequences online, predicts effects, updates memory, and replans.

See [Phase 52 runtime](./phase52_open_learning/README.md).

### 5. Blind generalization

Phase 53:

- 21/21 benchmark episodes solved
- 12/12 held-out episodes solved
- 0 invalid actions

See [Phase 53 report](./PHASE53_REPORT.md).

### 6. Phase 54 fresh evaluation

The locked Phase 54 evaluator uses six fresh task families with randomized opaque action identifiers.

Result:

```text
Training:     9 / 9
Held-out:     9 / 9
Overall:     18 / 18
Invalid:       0
```

Evaluator SHA256:

```text
c821c963ab8b80f5a28611c79f971deec6133eeee9836d210f293f325fcaed7c
```

The evaluator remained unchanged while the learner was fixed.

See [Phase 54 evaluation](./INDEPENDENT_EVALUATION_2026-09-19.md).

### 7. Phase 55 representation independence

Phase 55 tests whether the same bounded relational structure survives three substantially different byte encodings:

```text
edge list ↔ neighbor map ↔ binary adjacency matrix
                 ↓
       canonical unlabeled graph
                 ↓
          same fingerprint
```

Result:

```text
Progressive: 27 / 27
Held-out:     2 / 2
Adversarial:   6 / 6
Pytest:       11 passed
```

See [Phase 55 acceptance](./PHASE55_ACCEPTANCE.md).

### 8. Phase 56 raw concept acquisition

Phase 56 extends representation work into bounded concept discovery from undifferentiated byte streams:

```text
raw bytes
   ↓
canonical recurring motifs
   ↓
cross-episode support
   ↓
reusable concepts
   ↓
ordered relations / transition-events
```

Result:

```text
Progressive: 3 families × 3 seeds
Held-out:    2 / 2
Adversarial: 3 / 3
Pytest:      7 passed
```

See [Phase 56 acceptance](./PHASE56_ACCEPTANCE.md).

### 9. Phase 57 cross-view raw concept acquisition

Phase 57 removes the single-byte-stream boundary by receiving an unlabeled bundle of raw views and canonicalizing reusable relational structure across 1-D sequences and 2-D grids:

```text
unlabeled raw views
       ↓
relational atoms
       ↓
cross-view / cross-episode support
       ↓
permutation-null filtering
       ↓
compact concept vocabulary
       ↓
view-invariant relations + within-view events
```

Result:

```text
Progressive: 3 families × 3 seeds
Held-out:     2 / 2
Adversarial: 3 / 3
Scaling:      1 / 1
Pytest:      10 passed
```

See [Phase 57 acceptance](./PHASE57_ACCEPTANCE.md).

---

### 10. Phase 58 hierarchical concept abstraction

Phase 58 recursively discovers repeated compositions of previously stable concepts:

```text
stable concepts
      ↓
recurrent compositions
      ↓
compact abstraction
      ↓
higher-level concepts
      ↓
recursive hierarchy
```

Result:

```text
Progressive: 3 families × 3 seeds
Held-out:     2 / 2
Adversarial: 2 / 2
Pytest:      8 passed
```

See [Phase 58 acceptance](./PHASE58_ACCEPTANCE.md).

### 11. Phase 59 predictive concept learning

Phase 59 learns context-conditioned concept transitions with longest-supported context, deterministic backoff, and explicit abstention under ambiguity:

```text
hierarchical concepts
        ↓
context history
        ↓
prediction + confidence
        ↓
abstain / back off when evidence is weak
```

Result:

```text
Progressive: 3 families × 3 seeds
Held-out:     2 / 2
Adversarial: 2 / 2
Pytest:      8 passed
```

See [Phase 59 acceptance](./PHASE59_ACCEPTANCE.md).

### 12. Phase 60 long-horizon world modeling

Phase 60 adds exact transition memory, factorized action rules, numeric-delta generalization, discrepancy handling, and fail-closed rollout:

```text
state + action
      ↓
exact evidence
      ↓
factorized rule
      ↓
delta generalization
      ↓
long-horizon rollout
      ↓
observe / compare / update
```

Result:

```text
Progressive: 3 horizons × 3 seeds
Held-out:     2 / 2
Adversarial: 2 / 2
Integration: Phase 58 + Phase 59
Pytest:      8 passed
```

See [Phase 60 acceptance](./PHASE60_ACCEPTANCE.md).


### 13. Phase 61 predictive closed-loop autonomy

Phase 61 connects the Phase 58–60 mechanisms into an online goal-directed control loop:

```text
observe
  ↓
predict supported consequences
  ↓
bounded model-based plan
  ↓
goal-improving action
  ↓
safe exploration of unknown legal action
  ↓
observe result
  ↓
discrepancy detection
  ↓
replan / update
  ↺
```

Successful traces are also compressed through the hierarchy and used by the predictive concept model as a policy prior.

Result:

```text
Progressive: 3 families × 3 seeds
Held-out:     2 / 2
Adversarial: 3 / 3
Integration: Phase 58 + 59 + 60
Pytest:      9 passed
```

See [Phase 61 acceptance](./PHASE61_ACCEPTANCE.md).



### 14. Phase 62 partial observability + active information

Phase 62 removes the full-state visibility assumption:

```text
partial observation
       ↓
hidden goal-relevant variables
       ↓
infer information actions from consequences
       ↓
active information selection
       ↓
revealed state
       ↓
model-based planning
       ↓
observe / revise / fail closed
```

Result:

```text
Progressive: 3 families × 3 seeds
Held-out:     2 / 2
Adversarial: 3 / 3
Integration: Phase 61 handoff contract
Pytest:      9 passed
```

See [Phase 62 acceptance](./PHASE62_ACCEPTANCE.md).


### 15. Phase 63 nonstationary world + autonomous experiment design

Phase 63 removes the fixed-dynamics assumption:

```text
learn competing regimes
        ↓
predict
        ↓
detect repeated mismatch
        ↓
mark model stale
        ↓
design discriminating experiment
        ↓
identify active regime
        ↓
continue goal-directed control
```

Result:

```text
Progressive: 3 change levels × 3 seeds
Held-out:     2 / 2
Adversarial: 3 / 3
Integration: Phase 62 handoff contract
Pytest:      9 passed
```

See [Phase 63 acceptance](./PHASE63_ACCEPTANCE.md).



## 🔁 The Core Learning Loop

```mermaid
flowchart LR
    O["OBSERVE"] --> R["REPRESENT"]
    R --> S["STATE"]
    S --> P["PREDICT"]
    P --> X{"EXPECTED?"}
    X -- "Yes" --> A["ACT / CONTINUE"]
    X -- "No" --> D["DISCREPANCY"]
    D --> U["UPDATE TRANSITION / WORLD MODEL"]
    U --> M["REMEMBER"]
    M --> G["REPLAN"]
    G --> A
    A --> O
```

This loop is the central architectural idea behind the project.

---

## ✅ Verification Philosophy

Mirror 7 follows a strict promotion rule:

```text
IMPLEMENT
   ↓
FOCUSED TEST
   ↓
PROGRESSIVE TESTS
   ↓
MULTI-SEED
   ↓
HELD-OUT / UNSEEN
   ↓
ADVERSARIAL CONTROL
   ↓
FULL REGRESSION
   ↓
PROMOTE ONLY IF GREEN
```

When something fails:

```mermaid
flowchart TD
    A["FAIL"] --> B["REPRODUCE"]
    B --> C["MINIMIZE"]
    C --> D["TRACE EXACT FAILURE"]
    D --> E["SMALLEST JUSTIFIED FIX"]
    E --> F["RERUN FAILED CASE"]
    F --> G["FULL REGRESSION"]
    G --> H{"GREEN?"}
    H -- "No" --> B
    H -- "Yes" --> I["PROMOTE"]
```

Rules:

- Do not weaken an evaluator to obtain PASS.
- Do not remove failing seeds.
- Do not hardcode benchmark answers.
- Do not leak hidden task information into the learner.
- Do not call a toy benchmark result AGI evidence.
- Keep implementation, verification, integration, and research claims separate.

---

## 📊 Phase Map

<details>
<summary><strong>Phases 24–30 — Bootstrap</strong></summary>

| Phase | Result |
|---|:---:|
| 24 | ✅ Runtime dictionary |
| 25 | ✅ Tokenization / lookup / compilation |
| 26 | ✅ Native compiler path |
| 27 | ✅ Structured relocation / control flow |
| 28 | ✅ Surface parser/compiler integration |
| 29 | ✅ MIRR source closure |
| 30 | ✅ Compiler entirely in MIRR |
| Bootstrap chain | ✅ Self-recompile / fixed-point / independent rebuild |

</details>

<details>
<summary><strong>Phases 31–34 — Intelligence Foundations</strong></summary>

| Phase | Result |
|---|:---:|
| 31 | ✅ Representation / state / temporal identity |
| 32 | ✅ Prediction / transition memory / discrepancy |
| 33 | ✅ Causal structure / interventions |
| 34 | ✅ Goal-directed reasoning / planning |

</details>

<details>
<summary><strong>Phases 35–51 — Capability Expansion</strong></summary>

| Phase | Capability | Result |
|---|---|:---:|
| 35 | Action model | ✅ |
| 36 | Closed loop | ✅ |
| 37 | Working memory | ✅ |
| 38 | Episodic memory | ✅ |
| 39 | World model | ✅ |
| 40 | Composition | ✅ |
| 41 | Hierarchical planning | ✅ |
| 42 | Tool use | ✅ |
| 43 | Language grounding | ✅ |
| 44 | Counterfactual simulation | ✅ |
| 45 | Continual learning | ✅ |
| 46 | Meta-reasoning | ✅ |
| 47 | Efficiency mechanisms | ✅ |
| 48 | Robustness | ✅ |
| 49 | Transfer | ✅ |
| 50 | Complete integration | ✅ |
| 51 | Advanced reasoning | ✅ |

See [full Phase 35–51 verification](./PHASE35_51_VERIFICATION.md).

</details>

<details>
<summary><strong>Phases 52–54 — Generalization Boundary</strong></summary>

| Phase | Result |
|---|:---:|
| 52 | ✅ Open-ended black-box learner |
| 53 | ✅ Internal blind benchmark — 21/21 |
| 54 | ✅ Fresh locked evaluator — 18/18 |
| Phase 54 held-out | ✅ 9/9 |
| Phase 54 invalid actions | ✅ 0 |

See [Phase 53 report](./PHASE53_REPORT.md) and [Phase 54 evaluation](./INDEPENDENT_EVALUATION_2026-09-19.md).

</details>

<details>
<summary><strong>Phase 55 — Representation Independence</strong></summary>

| Gate | Result |
|---|:---:|
| Progressive cross-representation comparisons | ✅ 27/27 |
| Held-out graph families | ✅ 2/2 |
| Adversarial controls | ✅ 6/6 |
| Pytest | ✅ 11 passed |
| Determinism / invariance | ✅ |

See [Phase 55 acceptance](./PHASE55_ACCEPTANCE.md).

</details>

<details>
<summary><strong>Phase 56 — Raw Concept Acquisition</strong></summary>

| Gate | Result |
|---|:---:|
| Progressive concept families | ✅ 3 families × 3 seeds |
| Held-out recombination | ✅ |
| Held-out raw encoding change | ✅ |
| Adversarial controls | ✅ 3/3 |
| Pytest | ✅ 7 passed |
| Determinism | ✅ |

See [Phase 56 acceptance](./PHASE56_ACCEPTANCE.md).

</details>

---

## 🚧 What is still unsolved

Passing the current gates does **not** mean the project has reached AGI.

The major open research problems are:

### Raw multimodal concept acquisition

Can it discover entities, relations, events, and useful state from less regular raw observations, including modalities beyond byte streams, without developers hand-defining the ontology?

### Scaling

How does search and memory behave as:

```text
entities ↑
hypotheses ↑
state size ↑
planning depth ↑
distractors ↑
```

### Long-horizon autonomy

Can the system maintain reliable model → plan → act → observe → revise loops across much longer tasks?

### Strong abstraction

Can a computational principle learned in one environment transfer to another environment where the surface semantics are substantially different?

### Continual learning at scale

Can it acquire large amounts of new knowledge without corrupting old knowledge?

### Real-world grounding

Can the mechanisms operate robustly with external language, tools, files, sensors, and dynamic environments?

### Compute efficiency

What is the actual runtime/memory/sample-efficiency profile compared with strong neural baselines?

### Safety and reliability

Can failure, uncertainty, bad actions, model corruption, and irreversible outcomes be detected and contained?

### Independent reproduction

Can an unrelated implementation/team reproduce the important results from the frozen repository?

---

## 🎯 What Mirror 7 can do today

At the demonstrated boundary, Mirror 7 can:

```mermaid
flowchart TD
    A["Observe an environment"] --> B["Discover / represent structure"]
    B --> C["Build a stable state"]
    C --> D["Learn transitions"]
    D --> E["Predict consequences"]
    E --> F["Detect discrepancy"]
    F --> G["Update memory / model"]
    G --> H["Infer causal relationships"]
    H --> I["Form and evaluate goals"]
    I --> J["Plan"]
    J --> K["Act"]
    K --> L["Observe actual result"]
    L --> G
```

It also has tested mechanisms for:

- working and episodic memory;
- world-model updates;
- compositional reasoning;
- hierarchical planning;
- counterfactual simulation;
- tool execution;
- language-to-structure grounding;
- continual learning;
- epistemic bookkeeping and contradiction checks;
- open-ended action-effect discovery;
- generalization to unseen benchmark task families.

The **scope statement** is important: these capabilities are demonstrated in bounded, structured research environments and should not be read as a claim that Mirror 7 already has unrestricted real-world intelligence.

---

## 🧪 Research Boundaries & Known Caveats

### Phase 34 source publication

Phase 34 is now represented by a dedicated repository implementation under `phase34_reasoning_planning/`. Its executable acceptance gate covers 3 progressive task families × 3 seeds, 3 held-out cases, 3 adversarial controls, exact-plan validation, and 8/8 regression. The gate and pytest suite were repeated locally with the same result. This establishes the bounded reasoning/planning mechanism; it does not claim unrestricted planning or AGI.

### Phase 51 benchmark caveat

Several Phase 51 comparison values are synthetic/mock baseline values. The mechanism and executable gate are verified, but those values are not treated as independent scientific comparison results.

### Phase 55 representation-independence boundary

Phase 55 demonstrates representation invariance for bounded relational graphs across three generic byte encodings with opaque node identifiers and reordered serialization. The implementation uses exact canonicalization only within a bounded eight-node reference domain; this is not yet a claim of unrestricted multimodal representation learning.

### Phase 56 raw-concept boundary

Phase 56 demonstrates bounded unsupervised structural concept acquisition from raw byte streams. The discovered concepts are recurring structural motifs, while relations/events are derived from ordered concept co-occurrence. This is not yet semantic open-world grounding, multimodal perception, or unrestricted ontology induction.

### Phase 54 independence caveat

Phase 54 is an **independent-style** evaluation, not a third-party scientific replication. The evaluator was kept outside the learner implementation and remained locked, but both benchmark construction and learner development occurred within the same broader project workflow.

---

## 📚 Where to go next

| Document | Purpose |
|---|---|
| [STATUS.md](./STATUS.md) | Detailed acceptance board and evidence |
| [ROADMAP.md](./ROADMAP.md) | Capability roadmap and remaining research gaps |
| [BOOTSTRAP_PROGRESS.md](./BOOTSTRAP_PROGRESS.md) | Bootstrap / self-hosting chain |
| [PHASE34_ACCEPTANCE.md](./PHASE34_ACCEPTANCE.md) | Phase 34 acceptance record |
| [phase34_reasoning_planning/](./phase34_reasoning_planning/) | Phase 34 reasoning/planning implementation and tests |
| [PHASE35_51_VERIFICATION.md](./PHASE35_51_VERIFICATION.md) | Phase 35–51 verification record |
| [PHASE55_ACCEPTANCE.md](./PHASE55_ACCEPTANCE.md) | Phase 55 acceptance record |
| [phase55_representation_independence/](./phase55_representation_independence/) | Phase 55 implementation and tests |
| [PHASE56_ACCEPTANCE.md](./PHASE56_ACCEPTANCE.md) | Phase 56 acceptance record |
| [phase56_raw_concept_acquisition/](./phase56_raw_concept_acquisition/) | Phase 56 implementation and tests |
| [PHASE57_ACCEPTANCE.md](./PHASE57_ACCEPTANCE.md) | Phase 57 acceptance record |
| [phase57_multiview_concept_acquisition/](./phase57_multiview_concept_acquisition/) | Phase 57 implementation and tests |
| [PHASE58_ACCEPTANCE.md](./PHASE58_ACCEPTANCE.md) | Phase 58 acceptance record |
| [phase58_hierarchical_abstraction/](./phase58_hierarchical_abstraction/) | Phase 58 implementation and tests |
| [PHASE59_ACCEPTANCE.md](./PHASE59_ACCEPTANCE.md) | Phase 59 acceptance record |
| [phase59_predictive_concepts/](./phase59_predictive_concepts/) | Phase 59 implementation and tests |
| [PHASE60_ACCEPTANCE.md](./PHASE60_ACCEPTANCE.md) | Phase 60 acceptance record |
| [phase60_long_horizon_world_model/](./phase60_long_horizon_world_model/) | Phase 60 implementation and tests |
| [PHASE61_ACCEPTANCE.md](./PHASE61_ACCEPTANCE.md) | Phase 61 acceptance record |
| [phase61_closed_loop_autonomy/](./phase61_closed_loop_autonomy/) | Phase 61 implementation and tests |
| [PHASE62_ACCEPTANCE.md](./PHASE62_ACCEPTANCE.md) | Phase 62 acceptance record |
| [phase62_active_information/](./phase62_active_information/) | Phase 62 implementation and tests |
| [PHASE63_ACCEPTANCE.md](./PHASE63_ACCEPTANCE.md) | Phase 63 acceptance record |
| [phase63_nonstationary_experiments/](./phase63_nonstationary_experiments/) | Phase 63 implementation and tests |
| [PHASE53_REPORT.md](./PHASE53_REPORT.md) | Phase 53 blind benchmark |
| [INDEPENDENT_EVALUATION_2026-09-19.md](./INDEPENDENT_EVALUATION_2026-09-19.md) | Phase 54 locked evaluation |
| [phase52_open_learning/](./phase52_open_learning/) | Open-ended learner |
| [phase53_independent_eval/](./phase53_independent_eval/) | Blind evaluator |
| [LICENSE](./LICENSE) | Project license |

---

## ▶️ Running the project

The repository contains multiple phase-specific gates. Start with the relevant phase documentation rather than assuming one global command covers the entire project.

For the open-ended learner:

```bash
python -m pytest phase52_open_learning -q
```

For the Phase 53 evaluator:

```bash
python -m pytest phase53_independent_eval -q
```

For the Phase 56 acceptance gate:

```bash
python -m phase56_raw_concept_acquisition.run_phase56_gate
python -m pytest phase56_raw_concept_acquisition/test_phase56.py -q
```

For the Phase 57 acceptance gate:

```bash
python -m phase57_multiview_concept_acquisition.run_phase57_gate
python -m pytest phase57_multiview_concept_acquisition/test_phase57.py -q
```

For the Phase 55 acceptance gate:

```bash
python -m phase55_representation_independence.run_phase55_gate
python -m pytest phase55_representation_independence/test_phase55.py -q
```

For the Phase 34 acceptance gate:

```bash
python -m phase34_reasoning_planning.run_phase34_gate
python -m pytest phase34_reasoning_planning/test_phase34.py -q
```

For the Phase 32 acceptance gate:

```bash
python phase32_prediction/test_phase32_gate.py
```

See the individual phase READMEs and [STATUS.md](./STATUS.md) for the authoritative verification boundary.

---

## 🌱 Research Direction

The next milestone should not simply be “more benchmark rows.”

The strongest next question is:

> **Can Mirror 7 discover useful concepts from increasingly unstructured observations while scaling the same mechanism without exploding search or memory costs?**

That points toward three connected research tracks:

```mermaid
flowchart LR
    A["Raw concept acquisition"] --> D["General abstraction"]
    B["Scaling / computational analysis"] --> D
    C["Independent reproduction"] --> D
    D --> E["Broader real-world evaluation"]
```

Only after those experiments should the project commit to a specialized new runtime, language, compiler, or hardware substrate.

---

## 🛡️ License

Mirror 7 is released under the **Apache License 2.0**.

The repository contains the full license text in [LICENSE](./LICENSE).

Official license reference: https://www.apache.org/licenses/LICENSE-2.0

---

## 📌 Project Principle

```text
DO NOT ASK:
"How many features can we add?"

ASK:
"What general mechanism has actually been demonstrated?"
```

Mirror 7 is an ongoing research program.

The repository should always make it easy to distinguish:

```text
implemented
   ≠
verified
   ≠
integrated
   ≠
generalized
   ≠
independently reproduced
   ≠
AGI
```

That distinction is part of the project itself.
