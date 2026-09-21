# Phase 357 — Response Generation Interface

Defines the backend-facing adapter contract for response generation.

This phase does not implement or train a language model. It converts the
verified Phase 355 realization contract into a generation request and delegates
actual generation to an injected callable. The backend never invents response
content.
