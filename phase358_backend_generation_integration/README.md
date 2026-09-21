# Phase 358 — Backend Generation Integration

Connects the injected response generator to `BackendService`.

Generation is optional and receives only the verified Phase 355 realization contract through the Phase 357 interface. Generator failures fail the backend step; no fallback text or invented actions are produced.
