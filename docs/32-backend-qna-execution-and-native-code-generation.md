# Backend Q&A completion: guarded execution and native code generation

This change completes four backend connections without loading, creating, or modifying knowledge rules. Dataset and rule ingestion remain a separate project step.

## 1. Task plans can be carried through

Normal \`POST /api/crystal/ask\` requests ask the universal pipeline to execute eligible task plans. A plan runs only when all of these are true:

- The request maps to an executable task intent and the plan is ready.
- The host application supplies a \`NativeOperationRegistry\` through a \`CapabilityExecutionEngine\`.
- The host supplies a separate \`IndependentVerificationEngine\`.
- Required independent checks are explicitly configured for each capability.

The existing execution engine applies process, time, input/output and step limits, stops on the first failure, and does not retry side-effecting operations. A successful execution is reported as verified only after the separate verifier passes all required checks and declared postconditions. If the host has not configured these components, Q&A still returns the analysis and a clear reason why it did not execute anything.

Example host configuration:

\`\`\`python
from mirror_crystal.execution import CapabilityExecutionEngine, NativeOperationRegistry
from mirror_crystal.verification import IndependentVerificationEngine, IndependentVerifierRegistry

operations = NativeOperationRegistry()
# Register only trusted, importable native functions after reviewing them.
# operations.register("capability-id", handler_function)

checks = IndependentVerifierRegistry()
# Register independent check callbacks separately.
# checks.register(IndependentCheck(...), independent=True)

service = BackendService(
    capability_execution_engine=CapabilityExecutionEngine(operations),
    independent_verification_engine=IndependentVerificationEngine(checks),
    required_verification_checks={
        # "capability-id": ("independent-check-id",)
    },
)
\`\`\`

The placeholders above are configuration points, not active built-in operations. No operation handlers or knowledge records are auto-registered by this change.

## 2. Replies are composed from supported results

The Q&A response now selects between independently supported factual claims, independently verified task outputs, a TDD-verified code artifact, a clarification request, and an honest unsupported/unknown response. A memory entry is still context, not proof. A calculation candidate or retrieved unreviewed rule is not upgraded into verified fact.

For a completed task, the answer includes its output only when the execution report and the independent verification report both pass. For generated code, the answer includes source only when at least one test ran and all generated tests passed.

## 3. A small native Python code writer is connected to Q&A

When a coding request is detected, the default backend provider can construct a few explicit beginner-sized templates without an external model:

- Print text supplied in the request.
- Ask for the missing name rather than guessing when the request says “print my name” without giving a name.
- Add, subtract or multiply two numbers.
- Square a number.

Each generated program includes specific assertions and is sent to Mirror's existing static safety check and TDD sandbox. The answer is marked as a verified task result only if every non-vacuous assertion passes.

This is an initial, intentionally limited native code writer—not a general Python developer. Unsupported requests such as arbitrary sorting libraries are declined rather than answered with fabricated code. Broader code generation depends on later capability/rule feeding and further native algorithm development.

## 4. Factual Q&A uses the supplied knowledge base conservatively

Factual answers continue to come from the mounted Crystal knowledge vault and the existing evidence/claim checks. Reliable, independently reviewed claims may be returned. Unreviewed retrieval is shown only as an unverified candidate, and missing or conflicting evidence causes honest abstention.

No internet lookup service, new factual dataset, or knowledge rule is added by this change. After the separate feeding stage, the same Q&A path can make use of the newly supplied and reviewed knowledge.

## Tests

The regression suite adds coverage for:

- Python print and arithmetic generation with non-vacuous TDD checks.
- Missing-name clarification and unsupported-code abstention.
- The HTTP code-generation route.
- A normal Q&A request carrying out a ready task through a registered native handler and a separate independent verifier.
- Unknown factual questions remaining unanswered rather than invented.

Run the backend suite and the full repository CI workflow before release.
