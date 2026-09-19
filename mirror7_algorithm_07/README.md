# Mirror 7 Self-Debugging Algorithm (Algorithm 07)

## Purpose
Detect recurring failures in algorithmic processes, locate likely causes through statistical analysis, create candidate fixes, and evaluate them WITHOUT blindly modifying the system.

## Mechanism
- **Failure Log**: Maintains a bounded rolling window of recent failures.
- **Pattern Detection**: Clusters failures and flags clusters exceeding a threshold.
- **Root Cause Localization**: Uses information gain to rank context features.
- **Candidate Fix Generation**: Generates patches such as input filters.
- **Validation Protocol**: Tests candidates on original failing cases and passing cases.

## API
- `log_failure(record)`: Log a failure.
- `log_success(context)`: Log a passing case (for info gain).
- `detect_patterns()`: Returns list of patterns.
- `suggest_fix(pattern)`: Generates CandidatePatch.
- `validate_fix(patch, test_cases)`: Returns ValidationResult.

## Limitations
- Operates on discrete features.
- Fixes are proposed for external validation, not auto-applied.
- Simple mock sandboxed evaluation.
