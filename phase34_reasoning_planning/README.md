# Mirror 7 — Phase 34: Goal-Directed Reasoning / Planning

Phase 34 supplies the missing repository source artifact for the project's earlier reported reasoning/planning boundary.

## Mechanism

\`\`\`text
learned transition evidence
        ↓
state + goal
        ↓
goal-distance heuristic
        ↓
candidate predicted successors
        ↓
cycle / dominance checks
        ↓
bounded best-first search
        ↓
validated action sequence
\`\`\`

The planner does not call an environment and does not contain task-specific answers. \`TransitionModel\` is populated from observed \`(state, action, next_state)\` evidence. Exact observations are preferred; consistent numeric effects are generalized conservatively using repeated evidence and observed source-value guards.

## Phase 34 acceptance gate

The executable gate requires:

- 3 progressively harder planning tasks;
- 3 random seeds per progressive task;
- 3 held-out cases;
- 3 adversarial negative controls;
- exact-plan validation against the learned model;
- deterministic regression tests.

Run:

\`\`\`bash
python -m phase34_reasoning_planning.run_phase34_gate
python -m pytest phase34_reasoning_planning/test_phase34.py -q
\`\`\`

Expected final line:

\`\`\`text
PHASE 34 ACCEPTANCE GATE: PASS
\`\`\`

Phase 34 proves the bounded reasoning/planning mechanism represented here. It does not prove AGI or unrestricted real-world planning.
