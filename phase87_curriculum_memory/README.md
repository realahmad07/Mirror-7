# Phase 87 — Autonomous Curriculum Memory

Maintains a bounded queue of candidate experiments scored from novelty and uncertainty. Repeated identical candidates are down-weighted, invalid/non-finite inputs fail closed, and memory has a hard size cap.

This is a curriculum-selection mechanism, not an autonomous general scientist or AGI.
