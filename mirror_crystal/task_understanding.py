"""Mirror-native, domain-neutral task understanding and task contracts.

This layer interprets the shape of a request and records what was and was not
specified. It does not choose capabilities, execute work, verify outcomes, or learn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Iterable, Mapping, Sequence

from .universal_representation import (
    _freeze_json,
    _thaw_json,
    canonical_json,
    stable_representation_id,
)


class TaskUnderstandingError(ValueError):
    """Raised when a task contract is malformed or internally inconsistent."""


class TaskIntent(str, Enum):
    UNKNOWN = "unknown"
    BUILD = "build"
    CREATE = "create"
    IMPLEMENT = "implement"
    WRITE = "write"
    DESIGN = "design"
    EXPLAIN = "explain"
    DESCRIBE = "describe"
    ANALYZE = "analyze"
    COMPARE = "compare"
    CALCULATE = "calculate"
    SOLVE = "solve"
    PROVE = "prove"
    DERIVE = "derive"
    VERIFY = "verify"
    TEST = "test"
    PLAN = "plan"
    OPTIMIZE = "optimize"
    DIAGNOSE = "diagnose"
    RESEARCH = "research"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    CONVERT = "convert"
    CLASSIFY = "classify"
    EXTRACT = "extract"
    IDENTIFY = "identify"
    PREDICT = "predict"
    MEASURE = "measure"
    FIND = "find"
    GENERATE = "generate"
    LIST = "list"


class TaskStatus(str, Enum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"


class ConstraintKind(str, Enum):
    MUST = "must"
    MUST_NOT = "must_not"
    LIMIT = "limit"
    FORMAT = "format"
    SCOPE = "scope"
    PRESERVE = "preserve"
    OTHER = "other"


_ACTIONS: tuple[tuple[TaskIntent, tuple[str, ...]], ...] = (
    (TaskIntent.IMPLEMENT, ("implement", "implementing")),
    (TaskIntent.BUILD, ("build", "building", "built")),
    (TaskIntent.CREATE, ("create", "creating", "make", "making")),
    (TaskIntent.WRITE, ("write", "writing", "author", "draft")),
    (TaskIntent.DESIGN, ("design", "designing", "architect")),
    (TaskIntent.EXPLAIN, ("explain", "explain", "clarify")),
    (TaskIntent.DESCRIBE, ("describe", "define")),
    (TaskIntent.ANALYZE, ("analyze", "analyse", "investigate", "inspect", "audit")),
    (TaskIntent.COMPARE, ("compare", "contrast", "differentiate")),
    (TaskIntent.CALCULATE, ("calculate", "compute", "evaluate numerically")),
    (TaskIntent.SOLVE, ("solve", "answer", "work out")),
    (TaskIntent.PROVE, ("prove", "demonstrate formally")),
    (TaskIntent.DERIVE, ("derive", "deduce")),
    (TaskIntent.VERIFY, ("verify", "validate", "confirm")),
    (TaskIntent.TEST, ("test", "check", "benchmark")),
    (TaskIntent.PLAN, ("plan", "schedule", "roadmap")),
    (TaskIntent.OPTIMIZE, ("optimize", "optimise", "improve", "reduce")),
    (TaskIntent.DIAGNOSE, ("diagnose", "debug", "troubleshoot", "fix")),
    (TaskIntent.RESEARCH, ("research", "study", "explore")),
    (TaskIntent.SUMMARIZE, ("summarize", "summarise", "recap")),
    (TaskIntent.TRANSLATE, ("translate", "localize", "localise")),
    (TaskIntent.CONVERT, ("convert", "transform", "change")),
    (TaskIntent.CLASSIFY, ("classify", "categorize", "categorise", "group")),
    (TaskIntent.EXTRACT, ("extract", "parse", "collect")),
    (TaskIntent.IDENTIFY, ("identify", "recognize", "recognise", "detect")),
    (TaskIntent.PREDICT, ("predict", "forecast", "estimate")),
    (TaskIntent.MEASURE, ("measure", "quantify")),
    (TaskIntent.FIND, ("find", "locate", "search")),
    (TaskIntent.GENERATE, ("generate", "synthesize", "synthesise")),
    (TaskIntent.LIST, ("list", "enumerate")),
)

_ACTION_LOOKUP = {
    word: intent
    for intent, words in _ACTIONS
    for word in words
}
_ACTION_PATTERN = re.compile(
    r"\b(" + "|".join(
        re.escape(word)
        for word in sorted(_ACTION_LOOKUP, key=lambda item: (-len(item), item))
    ) + r")\b",
    re.IGNORECASE,
)

_DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "programming": (
        "programming", "software", "code", "python", "javascript", "typescript",
        "function", "class", "compiler", "script", "api", "database", "algorithm",
        "calculator", "website", "web app", "application", "source code",
    ),
    "mathematics": (
        "mathematics", "math", "algebra", "equation", "arithmetic", "calculus",
        "matrix", "eigenvalue", "eigenvector", "integral", "derivative", "theorem",
        "proof", "probability", "statistics", "geometry", "number theory",
    ),
    "physics": (
        "physics", "quantum", "mechanics", "thermodynamics", "electromagnetic",
        "relativity", "velocity", "acceleration", "force", "energy", "momentum",
        "wavefunction", "particle",
    ),
    "biology": (
        "biology", "biological", "cell", "gene", "genetic", "dna", "rna", "protein",
        "organism", "ecology", "evolution", "anatomy", "physiology", "microbe",
    ),
    "chemistry": (
        "chemistry", "chemical", "molecule", "atom", "reaction", "compound",
        "element", "molar", "stoichiometry", "acid", "base", "oxidation",
    ),
    "engineering": (
        "engineering", "mechanical", "electrical", "civil", "structural",
        "control system", "circuit", "robotics", "thermofluid", "manufacturing",
    ),
    "finance": (
        "finance", "financial", "accounting", "investment", "portfolio",
        "cash flow", "revenue", "profit", "budget", "valuation",
    ),
    "writing": (
        "essay", "article", "story", "email", "memo", "report", "copywriting",
        "paragraph", "writing style",
    ),
}

_FORMAT_TERMS = (
    "json", "csv", "xml", "yaml", "markdown", "html", "python script",
    "source code", "table", "bullet list", "report", "diagram", "chart",
    "function", "class", "spreadsheet", "pdf",
)
_FORMAT_PATTERN = re.compile(
    r"\b(?:format(?:ted)?\s+as|return|output|deliver|provide|save|export)\s+"
    r"(?:in\s+)?(?:a|an|the)?\s*(" +
    "|".join(re.escape(value) for value in sorted(_FORMAT_TERMS, key=len, reverse=True)) +
    r")\b",
    re.IGNORECASE,
)

_SEQUENTIAL_SPLIT = re.compile(
    r"\s*(,\s*)?(then|and\s+then|after\s+that|afterwards|next|finally)\s+",
    re.IGNORECASE,
)
_ACTION_CONJUNCTION_SPLIT = re.compile(
    r"\s+(?:and|also)\s+(?=(?:" + "|".join(
        re.escape(word)
        for word in sorted(_ACTION_LOOKUP, key=lambda item: (-len(item), item))
    ) + r")\b)",
    re.IGNORECASE,
)

_CONSTRAINT_PATTERNS: tuple[tuple[ConstraintKind, re.Pattern[str]], ...] = (
    (ConstraintKind.MUST_NOT, re.compile(
        r"\b(?:must\s+not|mustn't|do\s+not|don't|never|avoid|without)\b", re.I
    )),
    (ConstraintKind.LIMIT, re.compile(
        r"\b(?:at\s+most|at\s+least|no\s+more\s+than|less\s+than|under|within|"
        r"exactly|maximum|minimum|limit(?:ed)?\s+to)\b", re.I
    )),
    (ConstraintKind.PRESERVE, re.compile(
        r"\b(?:preserve|keep|retain|leave\s+unchanged)\b", re.I
    )),
    (ConstraintKind.MUST, re.compile(
        r"\b(?:must|required\s+to|needs?\s+to|shall|ensure)\b", re.I
    )),
    (ConstraintKind.FORMAT, re.compile(
        r"\b(?:as\s+(?:json|csv|xml|yaml|markdown|html|a\s+table)|"
        r"return\s+(?:json|csv|xml|yaml|markdown|html))\b", re.I
    )),
    (ConstraintKind.SCOPE, re.compile(
        r"\b(?:only|exclude|excluding|include|including|specifically|within\s+the\s+scope)\b", re.I
    )),
)

_ASSUMPTION_PATTERN = re.compile(
    r"(?:^|[.;\n])\s*((?:assume|assuming)\b[^.;\n]*)", re.I
)
_ACCEPTANCE_PATTERN = re.compile(
    r"\b(?:success\s+means|done\s+when|complete\s+when|acceptance\s+criteria?\s*:|"
    r"must\s+pass\s*:?)\s*([^.;\n]+)",
    re.I,
)


def _strings(values: Iterable[Any], field_name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result):
        raise TaskUnderstandingError(f"{field_name} cannot contain empty values")
    if len(set(result)) != len(result):
        raise TaskUnderstandingError(f"{field_name} values must be unique")
    return result


def _normalize_domain(domain: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(domain).casefold()).strip("_")
    return normalized or "general"


@dataclass(frozen=True)
class TaskConstraint:
    constraint_id: str
    text: str
    kind: ConstraintKind = ConstraintKind.OTHER
    hard: bool = True
    source: str = "user_request"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ConstraintKind(self.kind))
        if not self.constraint_id.strip() or not self.text.strip():
            raise TaskUnderstandingError("constraint_id and text must be non-empty")
        if not isinstance(self.hard, bool):
            raise TaskUnderstandingError("constraint.hard must be a boolean")
        if not self.source.strip():
            raise TaskUnderstandingError("constraint.source must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "text": self.text,
            "kind": self.kind.value,
            "hard": self.hard,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskConstraint":
        return cls(**dict(value))


@dataclass(frozen=True)
class TaskSubtask:
    subtask_id: str
    goal: str
    intent: TaskIntent = TaskIntent.UNKNOWN
    sequence_index: int = 0
    depends_on: tuple[str, ...] = ()
    source_text: str = ""
    acceptance_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", TaskIntent(self.intent))
        object.__setattr__(self, "depends_on", tuple(str(value) for value in self.depends_on))
        object.__setattr__(
            self, "acceptance_criteria",
            _strings(self.acceptance_criteria, "subtask.acceptance_criteria")
        )
        if not self.subtask_id.strip() or not self.goal.strip():
            raise TaskUnderstandingError("subtask_id and goal must be non-empty")
        if isinstance(self.sequence_index, bool) or self.sequence_index < 0:
            raise TaskUnderstandingError("subtask.sequence_index must be a non-negative integer")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise TaskUnderstandingError("subtask dependencies must be unique")
        if self.subtask_id in self.depends_on:
            raise TaskUnderstandingError("a subtask cannot depend on itself")

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "goal": self.goal,
            "intent": self.intent.value,
            "sequence_index": self.sequence_index,
            "depends_on": list(self.depends_on),
            "source_text": self.source_text,
            "acceptance_criteria": list(self.acceptance_criteria),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSubtask":
        data = dict(value)
        data["depends_on"] = tuple(data.get("depends_on", ()))
        data["acceptance_criteria"] = tuple(data.get("acceptance_criteria", ()))
        return cls(**data)


@dataclass(frozen=True)
class TaskSpec:
    """Canonical description of what the user wants, not how Mirror will execute it."""

    task_id: str
    request: str
    goal: str
    intent: TaskIntent
    domain: str = "general"
    status: TaskStatus = TaskStatus.READY
    known_inputs: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    requested_outputs: tuple[str, ...] = ()
    constraints: tuple[TaskConstraint, ...] = ()
    assumptions: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    subtasks: tuple[TaskSubtask, ...] = ()
    intent_evidence: tuple[str, ...] = ()
    interpreter_version: str = "task-understanding-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", TaskIntent(self.intent))
        object.__setattr__(self, "status", TaskStatus(self.status))
        object.__setattr__(self, "domain", _normalize_domain(self.domain))
        if not self.task_id.strip() or not self.request.strip() or not self.goal.strip():
            raise TaskUnderstandingError("task_id, request, and goal must be non-empty")
        if self.schema_version != 1:
            raise TaskUnderstandingError(f"unsupported task schema version: {self.schema_version}")
        if not isinstance(self.known_inputs, Mapping) or not isinstance(self.context, Mapping):
            raise TaskUnderstandingError("known_inputs and context must be mappings")
        object.__setattr__(self, "known_inputs", _freeze_json(self.known_inputs, "known_inputs"))
        object.__setattr__(self, "context", _freeze_json(self.context, "context"))
        object.__setattr__(self, "requested_outputs", _strings(self.requested_outputs, "requested_outputs"))
        object.__setattr__(self, "assumptions", _strings(self.assumptions, "assumptions"))
        object.__setattr__(self, "unknowns", _strings(self.unknowns, "unknowns"))
        object.__setattr__(self, "acceptance_criteria", _strings(self.acceptance_criteria, "acceptance_criteria"))
        object.__setattr__(self, "intent_evidence", _strings(self.intent_evidence, "intent_evidence"))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "subtasks", tuple(self.subtasks))
        if any(not isinstance(item, TaskConstraint) for item in self.constraints):
            raise TaskUnderstandingError("constraints must contain TaskConstraint values")
        if any(not isinstance(item, TaskSubtask) for item in self.subtasks):
            raise TaskUnderstandingError("subtasks must contain TaskSubtask values")
        identifiers = [item.subtask_id for item in self.subtasks]
        if len(set(identifiers)) != len(identifiers):
            raise TaskUnderstandingError("subtask identifiers must be unique")
        self.topological_order()

    def topological_order(self) -> tuple[str, ...]:
        """Return a stable dependency-first order and reject broken/cyclic task graphs."""
        by_id = {item.subtask_id: item for item in self.subtasks}
        for item in self.subtasks:
            missing = sorted(set(item.depends_on) - set(by_id))
            if missing:
                raise TaskUnderstandingError(
                    f"{item.subtask_id} references unknown dependencies: {', '.join(missing)}"
                )
        completed: list[str] = []
        remaining = set(by_id)
        while remaining:
            ready = sorted(
                (
                    identifier for identifier in remaining
                    if set(by_id[identifier].depends_on).issubset(completed)
                ),
                key=lambda identifier: (by_id[identifier].sequence_index, identifier),
            )
            if not ready:
                raise TaskUnderstandingError("subtask dependency cycle detected")
            chosen = ready[0]
            completed.append(chosen)
            remaining.remove(chosen)
        return tuple(completed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "request": self.request,
            "goal": self.goal,
            "intent": self.intent.value,
            "domain": self.domain,
            "status": self.status.value,
            "known_inputs": _thaw_json(self.known_inputs),
            "context": _thaw_json(self.context),
            "requested_outputs": list(self.requested_outputs),
            "constraints": [item.to_dict() for item in self.constraints],
            "assumptions": list(self.assumptions),
            "unknowns": list(self.unknowns),
            "acceptance_criteria": list(self.acceptance_criteria),
            "subtasks": [item.to_dict() for item in self.subtasks],
            "intent_evidence": list(self.intent_evidence),
            "interpreter_version": self.interpreter_version,
            "schema_version": self.schema_version,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def fingerprint(self) -> str:
        import hashlib
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = dict(value)
        data["constraints"] = tuple(
            item if isinstance(item, TaskConstraint) else TaskConstraint.from_dict(item)
            for item in data.get("constraints", ())
        )
        data["subtasks"] = tuple(
            item if isinstance(item, TaskSubtask) else TaskSubtask.from_dict(item)
            for item in data.get("subtasks", ())
        )
        for name in (
            "requested_outputs", "assumptions", "unknowns",
            "acceptance_criteria", "intent_evidence",
        ):
            data[name] = tuple(data.get(name, ()))
        return cls(**data)


class TaskUnderstandingEngine:
    """Converts one request into an explicit domain-neutral TaskSpec."""

    VERSION = "task-understanding-v1"

    @staticmethod
    def _infer_intent(text: str) -> tuple[TaskIntent, tuple[str, ...]]:
        match = _ACTION_PATTERN.search(text)
        if match is None:
            return TaskIntent.UNKNOWN, ()
        verb = match.group(1).casefold()
        return _ACTION_LOOKUP[verb], (verb,)

    @staticmethod
    def _infer_domain(text: str, domain_hint: str | None) -> str:
        if domain_hint is not None:
            normalized = _normalize_domain(domain_hint)
            if not normalized:
                raise TaskUnderstandingError("domain_hint cannot be empty")
            return normalized
        lowered = text.casefold()
        scores = {
            domain: sum(
                len(re.findall(r"\b" + re.escape(term) + r"\b", lowered))
                for term in terms
            )
            for domain, terms in _DOMAIN_TERMS.items()
        }
        best_score = max(scores.values(), default=0)
        if best_score == 0:
            return "general"
        winners = sorted(domain for domain, score in scores.items() if score == best_score)
        return winners[0] if len(winners) == 1 else "general"

    @staticmethod
    def _constraints(
        text: str,
        explicit_constraints: Iterable[str],
    ) -> tuple[TaskConstraint, ...]:
        found: list[TaskConstraint] = []
        seen: set[str] = set()
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            phrase = sentence.strip(" \t\r-•")
            if not phrase:
                continue
            kinds = [kind for kind, pattern in _CONSTRAINT_PATTERNS if pattern.search(phrase)]
            if not kinds:
                continue
            normalized = " ".join(phrase.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            kind = kinds[0]
            constraint_id = stable_representation_id("constraint", {"text": normalized})
            found.append(TaskConstraint(constraint_id, phrase, kind, True, "user_request"))
        for value in explicit_constraints:
            phrase = str(value).strip()
            if not phrase:
                raise TaskUnderstandingError("explicit constraints cannot be empty")
            normalized = " ".join(phrase.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            found.append(TaskConstraint(
                stable_representation_id("constraint", {"text": normalized}),
                phrase,
                ConstraintKind.OTHER,
                True,
                "caller_context",
            ))
        return tuple(found)

    @staticmethod
    def _requested_outputs(text: str) -> tuple[str, ...]:
        results: list[str] = []
        for match in _FORMAT_PATTERN.finditer(text):
            value = match.group(1).casefold()
            if value not in results:
                results.append(value)
        return tuple(results)

    @staticmethod
    def _assumptions(text: str) -> tuple[str, ...]:
        values = []
        for match in _ASSUMPTION_PATTERN.finditer(text):
            value = match.group(1).strip()
            if value and value not in values:
                values.append(value)
        return tuple(values)

    @staticmethod
    def _acceptance_criteria(text: str, supplied: Iterable[str]) -> tuple[str, ...]:
        values = []
        for match in _ACCEPTANCE_PATTERN.finditer(text):
            value = match.group(1).strip()
            if value and value not in values:
                values.append(value)
        for item in supplied:
            value = str(item).strip()
            if not value:
                raise TaskUnderstandingError("acceptance criteria cannot be empty")
            if value not in values:
                values.append(value)
        return tuple(values)

    @staticmethod
    def _split_action_clauses(text: str) -> tuple[tuple[str, ...], tuple[bool, ...]]:
        """Return actionable clauses and whether each clause depends on its predecessor."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        bullets = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            for line in lines
            if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line)
        ]
        if len(bullets) >= 2:
            actionable = tuple(
                part for part in bullets
                if TaskUnderstandingEngine._infer_intent(part)[0] is not TaskIntent.UNKNOWN
            )
            return actionable, tuple(False for _ in actionable)

        parts: list[str] = []
        sequential: list[bool] = []
        cursor = 0
        for match in _SEQUENTIAL_SPLIT.finditer(text):
            before = text[cursor:match.start()].strip(" ,;:\t\n")
            if before:
                parts.append(before)
                sequential.append(False if not parts else bool(match.group(2)))
            cursor = match.end()
        tail = text[cursor:].strip(" ,;:\t\n")
        if tail:
            parts.append(tail)

        if len(parts) <= 1:
            conjunctive = _ACTION_CONJUNCTION_SPLIT.split(text)
            parts = [piece.strip(" ,;:\t\n") for piece in conjunctive if piece.strip(" ,;:\t\n")]
            sequential = [False] * len(parts)

        # Keep only segments that express an explicit action.
        actionable_parts = []
        actionable_sequence = []
        for index, part in enumerate(parts):
            if TaskUnderstandingEngine._infer_intent(part)[0] is TaskIntent.UNKNOWN:
                continue
            actionable_parts.append(part)
            if len(actionable_parts) == 1:
                actionable_sequence.append(False)
            else:
                actionable_sequence.append(sequential[index] if index < len(sequential) else False)
        return tuple(actionable_parts), tuple(actionable_sequence)

    @staticmethod
    def _make_subtasks(request: str, acceptance_criteria: tuple[str, ...]) -> tuple[TaskSubtask, ...]:
        parts, sequential = TaskUnderstandingEngine._split_action_clauses(request)
        if len(parts) < 2:
            return ()
        items: list[TaskSubtask] = []
        previous: str | None = None
        for index, part in enumerate(parts):
            intent, _ = TaskUnderstandingEngine._infer_intent(part)
            identifier = stable_representation_id(
                "subtask",
                {"request": request, "index": index, "text": part},
            )
            dependencies = (previous,) if (index > 0 and sequential[index]) and previous else ()
            items.append(TaskSubtask(
                subtask_id=identifier,
                goal=part,
                intent=intent,
                sequence_index=index,
                depends_on=dependencies,
                source_text=part,
                acceptance_criteria=acceptance_criteria if index == len(parts) - 1 else (),
            ))
            previous = identifier
        return tuple(items)

    def interpret(
        self,
        request: str,
        *,
        domain_hint: str | None = None,
        goal: str | None = None,
        known_inputs: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        explicit_constraints: Iterable[str] = (),
        acceptance_criteria: Iterable[str] = (),
    ) -> TaskSpec:
        if not isinstance(request, str) or not request.strip():
            raise TaskUnderstandingError("request must be a non-empty string")
        text = request.strip()
        if known_inputs is not None and not isinstance(known_inputs, Mapping):
            raise TaskUnderstandingError("known_inputs must be a mapping")
        if context is not None and not isinstance(context, Mapping):
            raise TaskUnderstandingError("context must be a mapping")
        safe_inputs = dict(known_inputs or {})
        safe_context = dict(context or {})
        intent, intent_evidence = self._infer_intent(text)
        domain = self._infer_domain(text, domain_hint)
        unknowns: list[str] = []

        if intent is TaskIntent.UNKNOWN:
            unknowns.append("No explicit action verb was recognized; clarify the intended action.")
        unresolved_reference = re.search(r"\b(?:it|this|that|these|those)\b", text, re.I)
        if unresolved_reference and not safe_context and not safe_inputs and len(text.split()) <= 8:
            unknowns.append("The request contains a reference without a supplied referent or context.")
        if domain == "general" and domain_hint is None:
            # General is retained as a valid domain; the missing label is visible, not guessed.
            unknowns.append("No domain-specific cue was strong enough; domain remains general.")

        supplied_criteria = tuple(str(value).strip() for value in acceptance_criteria)
        criteria = self._acceptance_criteria(text, supplied_criteria)
        constraints = self._constraints(text, explicit_constraints)
        outputs = self._requested_outputs(text)
        assumptions = self._assumptions(text)
        subtasks = self._make_subtasks(text, criteria)
        status = (
            TaskStatus.NEEDS_CLARIFICATION
            if intent is TaskIntent.UNKNOWN or (
                unresolved_reference is not None and not safe_context and not safe_inputs
                and len(text.split()) <= 8
            )
            else TaskStatus.READY
        )
        task_id = stable_representation_id("task", {
            "request": text,
            "domain": domain,
            "goal": (goal or text).strip(),
            "known_inputs": safe_inputs,
            "context": safe_context,
            "explicit_constraints": [str(value).strip() for value in explicit_constraints],
            "acceptance_criteria": list(criteria),
        })
        return TaskSpec(
            task_id=task_id,
            request=text,
            goal=(goal or text).strip(),
            intent=intent,
            domain=domain,
            status=status,
            known_inputs=safe_inputs,
            context=safe_context,
            requested_outputs=outputs,
            constraints=constraints,
            assumptions=assumptions,
            unknowns=tuple(dict.fromkeys(unknowns)),
            acceptance_criteria=criteria,
            subtasks=subtasks,
            intent_evidence=intent_evidence,
            interpreter_version=self.VERSION,
        )

    def interpret_dict(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self.interpret(request, **kwargs).to_dict()
