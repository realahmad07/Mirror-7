from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResponseGenerationRecord:
    sequence: int
    generated: bool
    mode: str | None
    response: Any | None


def make_generation_record(result: Any) -> ResponseGenerationRecord:
    contract = getattr(result, "realization_contract", None)
    response = getattr(result, "response", None)
    return ResponseGenerationRecord(
        sequence=int(getattr(result, "sequence", 0)),
        generated=response is not None,
        mode=getattr(contract, "mode", None),
        response=response,
    )


def record_to_mapping(record: ResponseGenerationRecord) -> Mapping[str, Any]:
    return {
        "sequence": record.sequence,
        "generated": record.generated,
        "mode": record.mode,
        "response": record.response,
    }
