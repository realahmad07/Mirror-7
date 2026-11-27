"""SHA-256 verified runtime dataset loader and source-disjoint splitters for Stage S11.0."""
from __future__ import annotations

from .loader import (
    DEFAULT_CACHE_DIR,
    DatasetDownloadError,
    download_or_load_verified,
    load_balanced_copa_raw,
    load_longmemeval_raw,
    load_squad_v2_raw,
)
from .splitter import (
    SourceDisjointSplitResult,
    build_connected_components,
    partition_source_clusters,
    verify_split_disjointness,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DatasetDownloadError",
    "SourceDisjointSplitResult",
    "build_connected_components",
    "download_or_load_verified",
    "load_balanced_copa_raw",
    "load_longmemeval_raw",
    "load_squad_v2_raw",
    "partition_source_clusters",
    "verify_split_disjointness",
]
