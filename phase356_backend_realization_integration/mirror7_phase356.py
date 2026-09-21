from __future__ import annotations

from typing import Any

from mirror7_backend.runtime import BackendResult


def backend_realization_contract_from_result(result: BackendResult) -> Any | None:
    """Return the verified realization contract exposed by the backend result."""
    return result.realization_contract
