# Algorithm 4 - Adaptive World-Model Builder

## Purpose
Constructs a predictive model of an unfamiliar environment and continuously revises it when observations contradict predictions.

## API
- `predict(state, action)`
- `observe_transition(state, action, next_state)`
- `get_rules()`
- `explain_prediction(state, action)`
