# Long-Term Goal Manager

## Purpose
Maintains goals over long periods, decomposes them, and reprioritizes based on urgency and failures.

## Mechanism
- Goal Tree (DAG) with Priority Scheduling
- Bounded depth and total goals
- Precondition checking and progress updates

## API
- `add_goal`
- `decompose`
- `get_next_goal`
- `update_progress`
- `report_failure`
- `get_plan`

## Limitations
- Bounded tree depth and goal count limits long-range chaining
