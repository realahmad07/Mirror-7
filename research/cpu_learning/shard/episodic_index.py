"""Multi-table LSH bit-projection episodic & operator indexes over 10,240-bit BSC vectors.

STRICT ISOLATION (Stage S4):
- Pure-CPU integer bit-masking LSH + exact 10,240-bit POPCNT Hamming reranking.
- ZERO GPU, ZERO backprop, ZERO hardcoded domain literals, NEVER imports
  ``hidden_actions``, ``oracle``, or ``generators``.
- Tracks exact ``memory_accesses``, ``bucket_candidates_scanned``, ``index_bits``,
  and enforces ``.freeze()`` to prohibit test-time insertions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mirror_crystal.quotient_egraph_resonator import (
    D_UNIFIED,
    hamming_sim10240,
)
from research.cpu_learning.itl.operator_card import OperatorCard
from research.cpu_learning.ledger.episode_ledger import EpisodeLedger
from research.cpu_learning.shard.encoders import (
    encode_operator_card_hv,
    encode_sparse_delta_hv,
    encode_typed_state_hv,
)
from research.cpu_learning.simulator.state_graph import (
    SparseDelta,
    StateGraph,
    TrainingEpisode,
    compressed_bit_length,
)

SHARD_INDEX_SCHEMA_VERSION: str = "s4-shard-lsh-index-v1"


class IndexFrozenError(RuntimeError):
    """Raised when an insertion is attempted on a frozen SHARD LSH index."""


def _deterministic_lsh_bit_tables(
    num_tables: int,
    bits_per_table: int,
    seed_tag: str,
) -> tuple[tuple[int, ...], ...]:
    """Deterministically sample ``num_tables`` bit-projection tuples in ``[0, D_UNIFIED)``."""
    tables: list[tuple[int, ...]] = []
    for t_idx in range(num_tables):
        chosen: list[int] = []
        used: set[int] = set()
        nonce = 0
        while len(chosen) < bits_per_table:
            digest = hashlib.blake2b(
                f"{seed_tag}:t{t_idx}:n{nonce}".encode("utf-8"), digest_size=32
            ).digest()
            for off in range(0, len(digest), 2):
                bit_pos = int.from_bytes(digest[off : off + 2], "little") % D_UNIFIED
                if bit_pos not in used:
                    used.add(bit_pos)
                    chosen.append(bit_pos)
                    if len(chosen) == bits_per_table:
                        break
            nonce += 1
        tables.append(tuple(chosen))
    return tuple(tables)


def _lsh_bucket_key(hv: int, bit_positions: tuple[int, ...]) -> int:
    key = 0
    for idx, bpos in enumerate(bit_positions):
        if (hv >> bpos) & 1:
            key |= 1 << idx
    return key


class _MultiTableLSHCore:
    """Reusable pure-CPU multi-table bit-projection LSH index over 10,240-bit BSC vectors."""

    def __init__(
        self,
        *,
        num_tables: int = 12,
        bits_per_table: int = 4,
        seed_tag: str = "shard_lsh_v1",
    ) -> None:
        self.num_tables = max(1, int(num_tables))
        self.bits_per_table = max(1, min(16, int(bits_per_table)))
        self.seed_tag = str(seed_tag)
        self._bit_tables = _deterministic_lsh_bit_tables(
            self.num_tables, self.bits_per_table, self.seed_tag
        )
        self._buckets: list[dict[int, list[int]]] = [{} for _ in range(self.num_tables)]
        self._vectors: list[int] = []
        self._frozen: bool = False
        self.memory_accesses: int = 0
        self.bucket_candidates_scanned: int = 0
        self.queries: int = 0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def reset_telemetry(self) -> None:
        self.memory_accesses = 0
        self.bucket_candidates_scanned = 0
        self.queries = 0

    def add_vector(self, hv: int) -> int:
        if self._frozen:
            raise IndexFrozenError("SHARD LSH index is frozen; test-time insertions are forbidden")
        item_idx = len(self._vectors)
        self._vectors.append(hv)
        for t_idx, bit_positions in enumerate(self._bit_tables):
            bkey = _lsh_bucket_key(hv, bit_positions)
            self._buckets[t_idx].setdefault(bkey, []).append(item_idx)
        return item_idx

    def query_ranked_indices(
        self,
        query_hv: int,
        *,
        top_k: int | None = None,
        include_fallback: bool = True,
        tie_keys: Sequence[str] | None = None,
    ) -> list[tuple[int, int]]:
        """Retrieve candidate item indices via LSH buckets and rerank with exact POPCNT Hamming."""
        self.queries += 1
        n_total = len(self._vectors)
        if n_total == 0:
            return []

        candidate_set: set[int] = set()
        for t_idx, bit_positions in enumerate(self._bit_tables):
            bkey = _lsh_bucket_key(query_hv, bit_positions)
            bucket = self._buckets[t_idx].get(bkey)
            if bucket:
                candidate_set.update(bucket)

        # Fallback to full scan ONLY when LSH union is empty (or caller explicitly wants full order)
        if not candidate_set:
            candidate_indices = list(range(n_total))
        else:
            candidate_indices = sorted(candidate_set)

        self.bucket_candidates_scanned += len(candidate_indices)
        self.memory_accesses += len(candidate_indices)

        scored: list[tuple[int, int]] = [
            (idx, hamming_sim10240(query_hv, self._vectors[idx]))
            for idx in candidate_indices
        ]
        if tie_keys is not None:
            scored.sort(key=lambda pair: (-pair[1], tie_keys[pair[0]]))
        else:
            scored.sort(key=lambda pair: (-pair[1], pair[0]))

        if include_fallback and len(candidate_indices) < n_total:
            missing = [i for i in range(n_total) if i not in candidate_set]
            self.memory_accesses += len(missing)
            scored.extend(
                (idx, hamming_sim10240(query_hv, self._vectors[idx])) for idx in missing
            )
            if tie_keys is not None:
                scored.sort(key=lambda pair: (-pair[1], tie_keys[pair[0]]))
            else:
                scored.sort(key=lambda pair: (-pair[1], pair[0]))

        if top_k is not None:
            return scored[:top_k]
        return scored

    def raw_index_bits(self) -> int:
        # 10,240 bits per indexed hypervector + bucket pointer entries (32 bits per entry)
        vec_bits = len(self._vectors) * D_UNIFIED
        bucket_entries = sum(len(v) for table in self._buckets for v in table.values())
        return vec_bits + bucket_entries * 32


class SHARDOperatorIndex:
    """Multi-table LSH index over a verified ``OperatorCard`` library (10,240-bit BSC subspace)."""

    def __init__(
        self,
        *,
        num_tables: int = 12,
        bits_per_table: int = 4,
        seed_tag: str = "shard_op_lsh_v1",
    ) -> None:
        self._core = _MultiTableLSHCore(
            num_tables=num_tables, bits_per_table=bits_per_table, seed_tag=seed_tag
        )
        self._cards: list[OperatorCard] = []
        self._action_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}

    @property
    def is_frozen(self) -> bool:
        return self._core.is_frozen

    @property
    def memory_accesses(self) -> int:
        return self._core.memory_accesses

    @property
    def bucket_candidates_scanned(self) -> int:
        return self._core.bucket_candidates_scanned

    @property
    def queries(self) -> int:
        return self._core.queries

    def __len__(self) -> int:
        return len(self._cards)

    def freeze(self) -> None:
        self._core.freeze()

    def reset_telemetry(self) -> None:
        self._core.reset_telemetry()

    def add_operator(self, card: OperatorCard) -> int:
        if self._core.is_frozen:
            raise IndexFrozenError("SHARDOperatorIndex is frozen; test-time insertions are forbidden")
        if card.action_id in self._id_to_idx:
            raise ValueError(f"Duplicate action_id {card.action_id!r} in SHARDOperatorIndex")
        hv = encode_operator_card_hv(card)
        idx = self._core.add_vector(hv)
        self._cards.append(card)
        self._action_ids.append(card.action_id)
        self._id_to_idx[card.action_id] = idx
        return idx

    def index_library(self, cards: Mapping[str, OperatorCard]) -> None:
        for aid in sorted(cards):
            self.add_operator(cards[aid])

    def operator_hv(self, action_id: str) -> int:
        idx = self._id_to_idx[action_id]
        return self._core._vectors[idx]

    def query_by_hv(
        self,
        query_hv: int,
        *,
        top_k: int | None = None,
        include_fallback: bool = True,
    ) -> list[tuple[OperatorCard, int]]:
        ranked = self._core.query_ranked_indices(
            query_hv,
            top_k=top_k,
            include_fallback=include_fallback,
            tie_keys=self._action_ids,
        )
        return [(self._cards[idx], sim) for idx, sim in ranked]

    def query_by_delta(
        self,
        delta: SparseDelta,
        before_state: StateGraph,
        *,
        top_k: int | None = None,
        include_fallback: bool = True,
    ) -> list[tuple[OperatorCard, int]]:
        q_hv = encode_sparse_delta_hv(delta, before_state)
        return self.query_by_hv(q_hv, top_k=top_k, include_fallback=include_fallback)

    def index_bits(self) -> int:
        return self._core.raw_index_bits()

    def compressed_index_bits(self) -> int:
        if not self._cards:
            return 0
        raw_bytes = b"".join(
            hv.to_bytes(D_UNIFIED // 8, "little") for hv in self._core._vectors
        )
        return min(len(raw_bytes) * 8, compressed_bit_length(raw_bytes))

    def telemetry(self) -> dict[str, Any]:
        return {
            "schema_version": SHARD_INDEX_SCHEMA_VERSION,
            "num_operators": len(self._cards),
            "queries": self._core.queries,
            "memory_accesses": self._core.memory_accesses,
            "bucket_candidates_scanned": self._core.bucket_candidates_scanned,
            "index_bits": self.index_bits(),
            "frozen": self.is_frozen,
        }


@dataclass(frozen=True)
class IndexedEpisodeEntry:
    """Public episode entry retained inside ``SHARDEpisodicIndex``."""

    episode_id: str
    action_id: str
    roles: tuple[tuple[str, str], ...]
    sparse_delta: SparseDelta
    delta_hv: int
    state_hv: int

    def role_dict(self) -> dict[str, str]:
        return dict(self.roles)


class SHARDEpisodicIndex:
    """Multi-table LSH index over public training episodes / ``EpisodeLedger`` records."""

    def __init__(
        self,
        *,
        num_tables: int = 12,
        bits_per_table: int = 4,
        seed_tag: str = "shard_ep_lsh_v1",
    ) -> None:
        self._delta_core = _MultiTableLSHCore(
            num_tables=num_tables, bits_per_table=bits_per_table, seed_tag=f"{seed_tag}:delta"
        )
        self._entries: list[IndexedEpisodeEntry] = []
        self._entry_ids: list[str] = []

    @property
    def is_frozen(self) -> bool:
        return self._delta_core.is_frozen

    @property
    def memory_accesses(self) -> int:
        return self._delta_core.memory_accesses

    @property
    def bucket_candidates_scanned(self) -> int:
        return self._delta_core.bucket_candidates_scanned

    @property
    def queries(self) -> int:
        return self._delta_core.queries

    def __len__(self) -> int:
        return len(self._entries)

    def freeze(self) -> None:
        self._delta_core.freeze()

    def reset_telemetry(self) -> None:
        self._delta_core.reset_telemetry()

    def add_episode(self, ep: TrainingEpisode, *, effect_only: bool = True) -> int | None:
        if self._delta_core.is_frozen:
            raise IndexFrozenError("SHARDEpisodicIndex is frozen; test-time insertions are forbidden")
        if effect_only and ep.sparse_delta.is_noop:
            return None
        d_hv = encode_sparse_delta_hv(ep.sparse_delta, ep.before_state)
        s_hv = encode_typed_state_hv(ep.before_state)
        idx = self._delta_core.add_vector(d_hv)
        self._entries.append(
            IndexedEpisodeEntry(
                episode_id=ep.episode_id,
                action_id=ep.action.action_id,
                roles=ep.action.roles,
                sparse_delta=ep.sparse_delta,
                delta_hv=d_hv,
                state_hv=s_hv,
            )
        )
        self._entry_ids.append(ep.episode_id)
        return idx

    def index_episodes(
        self,
        episodes: Sequence[TrainingEpisode],
        *,
        effect_only: bool = True,
    ) -> int:
        count = 0
        for ep in episodes:
            if self.add_episode(ep, effect_only=effect_only) is not None:
                count += 1
        return count

    def index_from_ledger(
        self,
        ledger: EpisodeLedger,
        *,
        split_id: str | None = None,
        seed: int | None = None,
        effect_only: bool = True,
        caller_tag: str = "shard_episodic_index",
    ) -> int:
        """Index public records directly from an S1 ``EpisodeLedger``."""
        if self._delta_core.is_frozen:
            raise IndexFrozenError("SHARDEpisodicIndex is frozen; test-time insertions are forbidden")
        if split_id is not None:
            records = ledger.query_by_split(split_id, seed=seed, caller_tag=caller_tag)
        else:
            records = ledger.iter_records(caller_tag=caller_tag)
            if seed is not None:
                records = tuple(r for r in records if r.seed == int(seed))

        count = 0
        for rec in records:
            if not rec.selected_action_id or rec.selected_roles is None or rec.is_action_ambiguous:
                continue
            if effect_only and rec.sparse_delta.is_noop:
                continue
            before_sg = rec.before_frame.to_observed_state_graph()
            d_hv = encode_sparse_delta_hv(rec.sparse_delta, before_sg)
            s_hv = encode_typed_state_hv(before_sg)
            self._delta_core.add_vector(d_hv)
            self._entries.append(
                IndexedEpisodeEntry(
                    episode_id=rec.episode_id,
                    action_id=rec.selected_action_id,
                    roles=tuple(sorted(dict(rec.selected_roles).items())),
                    sparse_delta=rec.sparse_delta,
                    delta_hv=d_hv,
                    state_hv=s_hv,
                )
            )
            self._entry_ids.append(rec.episode_id)
            count += 1
        return count

    def query_by_delta_hv(
        self,
        delta_hv: int,
        *,
        top_k: int = 5,
        include_fallback: bool = True,
    ) -> list[tuple[IndexedEpisodeEntry, int]]:
        ranked = self._delta_core.query_ranked_indices(
            delta_hv,
            top_k=top_k,
            include_fallback=include_fallback,
            tie_keys=self._entry_ids,
        )
        return [(self._entries[idx], sim) for idx, sim in ranked]

    def query_by_delta(
        self,
        delta: SparseDelta,
        before_state: StateGraph,
        *,
        top_k: int = 5,
        include_fallback: bool = True,
    ) -> list[tuple[IndexedEpisodeEntry, int]]:
        q_hv = encode_sparse_delta_hv(delta, before_state)
        return self.query_by_delta_hv(q_hv, top_k=top_k, include_fallback=include_fallback)

    def index_bits(self) -> int:
        # Both delta_hv (indexed in LSH) and state_hv retained per episode entry
        return self._delta_core.raw_index_bits() + len(self._entries) * D_UNIFIED

    def telemetry(self) -> dict[str, Any]:
        return {
            "schema_version": SHARD_INDEX_SCHEMA_VERSION,
            "num_episodes": len(self._entries),
            "queries": self._delta_core.queries,
            "memory_accesses": self._delta_core.memory_accesses,
            "bucket_candidates_scanned": self._delta_core.bucket_candidates_scanned,
            "index_bits": self.index_bits(),
            "frozen": self.is_frozen,
        }
