"""Deterministic source-disjoint cluster splitting with connected-component leak prevention.

GUARANTEES:
- Items sharing ANY source unit (e.g. mirrored counterfactual choice pair, Wikipedia article,
  or shared conversational session ID) are grouped into one atomic connected component before
  partitioning into ``train``, ``validation``, and ``hidden_eval``.
- Mathematical verification asserts ZERO overlap across splits on item IDs, cluster IDs,
  and all declared source resource IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SourceDisjointSplitResult:
    """Deterministic partition of cluster IDs and item IDs into train / validation / hidden_eval."""

    benchmark_id: str
    train_item_ids: tuple[str, ...]
    val_item_ids: tuple[str, ...]
    hidden_item_ids: tuple[str, ...]
    train_clusters: tuple[str, ...]
    val_clusters: tuple[str, ...]
    hidden_clusters: tuple[str, ...]
    disjointness_verified: bool
    disjointness_report: dict[str, object]


def _stable_rank_key(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}::{key}".encode("utf-8")).hexdigest()


def build_connected_components(
    item_to_resources: Mapping[str, Iterable[str]],
) -> dict[str, str]:
    """Union-Find connected components mapping each item_id to a canonical component_id.

    If two items share ANY resource (e.g. a LongMemEval session ID or Balanced COPA choice pair),
    they receive the exact same component_id and can NEVER be placed in different splits.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    resource_owner: dict[str, str] = {}
    for item_id in sorted(item_to_resources.keys()):
        parent.setdefault(item_id, item_id)
        for res in sorted(set(str(r) for r in item_to_resources[item_id])):
            if res in resource_owner:
                union(item_id, resource_owner[res])
            else:
                resource_owner[res] = item_id

    return {item_id: f"comp_{find(item_id)}" for item_id in sorted(parent.keys())}


def partition_source_clusters(
    benchmark_id: str,
    item_to_cluster: Mapping[str, str],
    *,
    train_ratio: float = 0.40,
    val_ratio: float = 0.20,
    salt: str = "s11.0-source-disjoint-v1",
    cluster_strata: Mapping[str, str] | None = None,
    fixed_hidden_clusters: Iterable[str] = (),
    item_to_resources: Mapping[str, Iterable[str]] | None = None,
) -> SourceDisjointSplitResult:
    """Deterministically partition atomic source clusters into train / validation / hidden_eval."""
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("Invalid split ratios")
    if train_ratio + val_ratio > 1.0 + 1e-9:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    fixed_hidden = set(str(c) for c in fixed_hidden_clusters)
    cluster_to_items: dict[str, list[str]] = {}
    for item_id, cid in sorted(item_to_cluster.items()):
        cluster_to_items.setdefault(str(cid), []).append(str(item_id))

    all_clusters = sorted(cluster_to_items.keys())
    assignable = [c for c in all_clusters if c not in fixed_hidden]

    train_c: list[str] = []
    val_c: list[str] = []
    hidden_c: list[str] = sorted(c for c in all_clusters if c in fixed_hidden)

    # Group assignable clusters by optional stratum for balanced representation.
    by_stratum: dict[str, list[str]] = {}
    for cid in assignable:
        strat = str(cluster_strata.get(cid, "default")) if cluster_strata else "default"
        by_stratum.setdefault(strat, []).append(cid)

    for strat in sorted(by_stratum.keys()):
        group = sorted(
            by_stratum[strat],
            key=lambda c: (_stable_rank_key(f"{salt}:{benchmark_id}:{strat}", c), c),
        )
        n = len(group)
        if fixed_hidden:
            # Partition assignable pool strictly into train vs val preserving relative proportions.
            denom = train_ratio + val_ratio
            n_train = int(round(n * (train_ratio / denom))) if denom > 0 else n
            n_train = max(1, min(n - (1 if n >= 2 and val_ratio > 0 else 0), n_train))
            train_c.extend(group[:n_train])
            val_c.extend(group[n_train:])
        else:
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            if n >= 3:
                n_train = max(1, min(n - 2, n_train))
                n_val = max(1, min(n - n_train - 1, n_val))
            elif n == 2:
                n_train, n_val = 1, 0
            elif n == 1:
                n_train, n_val = 0, 0
            train_c.extend(group[:n_train])
            val_c.extend(group[n_train : n_train + n_val])
            hidden_c.extend(group[n_train + n_val :])

    train_c_sorted = tuple(sorted(train_c))
    val_c_sorted = tuple(sorted(val_c))
    hidden_c_sorted = tuple(sorted(hidden_c))

    train_items = tuple(sorted(i for c in train_c_sorted for i in cluster_to_items[c]))
    val_items = tuple(sorted(i for c in val_c_sorted for i in cluster_to_items[c]))
    hidden_items = tuple(sorted(i for c in hidden_c_sorted for i in cluster_to_items[c]))

    ok, report = verify_split_disjointness(
        train_items=train_items,
        val_items=val_items,
        hidden_items=hidden_items,
        item_to_cluster=item_to_cluster,
        item_to_resources=item_to_resources,
    )
    if not ok:
        raise AssertionError(f"Source-disjoint split violation on {benchmark_id}: {report}")

    return SourceDisjointSplitResult(
        benchmark_id=benchmark_id,
        train_item_ids=train_items,
        val_item_ids=val_items,
        hidden_item_ids=hidden_items,
        train_clusters=train_c_sorted,
        val_clusters=val_c_sorted,
        hidden_clusters=hidden_c_sorted,
        disjointness_verified=ok,
        disjointness_report=report,
    )


def verify_split_disjointness(
    *,
    train_items: Sequence[str],
    val_items: Sequence[str],
    hidden_items: Sequence[str],
    item_to_cluster: Mapping[str, str],
    item_to_resources: Mapping[str, Iterable[str]] | None = None,
) -> tuple[bool, dict[str, object]]:
    """Mathematically verify zero overlap across train / validation / hidden_eval."""
    s_tr = set(train_items)
    s_va = set(val_items)
    s_hi = set(hidden_items)

    item_overlaps = {
        "train_val": len(s_tr & s_va),
        "train_hidden": len(s_tr & s_hi),
        "val_hidden": len(s_va & s_hi),
    }

    c_tr = {item_to_cluster[i] for i in s_tr}
    c_va = {item_to_cluster[i] for i in s_va}
    c_hi = {item_to_cluster[i] for i in s_hi}
    cluster_overlaps = {
        "train_val": len(c_tr & c_va),
        "train_hidden": len(c_tr & c_hi),
        "val_hidden": len(c_va & c_hi),
    }

    resource_overlaps = {"train_val": 0, "train_hidden": 0, "val_hidden": 0}
    if item_to_resources is not None:
        r_tr = {str(r) for i in s_tr for r in item_to_resources.get(i, ())}
        r_va = {str(r) for i in s_va for r in item_to_resources.get(i, ())}
        r_hi = {str(r) for i in s_hi for r in item_to_resources.get(i, ())}
        resource_overlaps = {
            "train_val": len(r_tr & r_va),
            "train_hidden": len(r_tr & r_hi),
            "val_hidden": len(r_va & r_hi),
        }

    ok = (
        all(v == 0 for v in item_overlaps.values())
        and all(v == 0 for v in cluster_overlaps.values())
        and all(v == 0 for v in resource_overlaps.values())
    )
    return ok, {
        "item_counts": {"train": len(s_tr), "val": len(s_va), "hidden_eval": len(s_hi)},
        "cluster_counts": {"train": len(c_tr), "val": len(c_va), "hidden_eval": len(c_hi)},
        "item_overlaps": item_overlaps,
        "cluster_overlaps": cluster_overlaps,
        "resource_overlaps": resource_overlaps,
        "zero_leakage_verified": ok,
    }
