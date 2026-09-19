from __future__ import annotations
import csv
import io
import urllib.request
from dataclasses import dataclass
from typing import Tuple

from phase171_180_external.protocol import ExternalEpisode, ExternalTaskPack, build_pack

UCI_IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
EXPECTED_ROWS = 150
EXPECTED_CLASSES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")

@dataclass(frozen=True)
class IrisRecord:
    row_id: int
    features: Tuple[float, float, float, float]
    label: str

def fetch_iris_records(timeout: float = 30.0) -> Tuple[IrisRecord, ...]:
    request = urllib.request.Request(UCI_IRIS_URL, headers={"User-Agent": "Mirror-7-external-evaluator/177"})
    raw = urllib.request.urlopen(request, timeout=timeout).read().decode("utf-8")
    rows = []
    for idx, row in enumerate(csv.reader(io.StringIO(raw))):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError(f"unexpected Iris row width at row {idx}: {len(row)}")
        features = tuple(float(x) for x in row[:4])
        label = row[4].strip()
        rows.append(IrisRecord(len(rows), features, label))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} Iris rows, got {len(rows)}")
    if tuple(sorted({r.label for r in rows})) != EXPECTED_CLASSES:
        raise ValueError("unexpected Iris class set")
    return tuple(rows)

def make_external_pack(records: Tuple[IrisRecord, ...], pack_id: str = "uci-iris-phase177") -> ExternalTaskPack:
    episodes = []
    for record in records:
        observation = ",".join(f"{x:.10g}" for x in record.features)
        episodes.append(ExternalEpisode(
            episode_id=f"iris-{record.row_id:03d}",
            observations=(observation,),
            actions=(),
            outcomes=(),
            goal="classify_iris_instance",
            legal_actions=EXPECTED_CLASSES,
            metadata={
                "source": "UCI Machine Learning Repository",
                "dataset": "Iris",
                "source_url": UCI_IRIS_URL,
                "row_id": record.row_id,
            },
        ))
    return build_pack(
        pack_id, "1",
        episodes,
        public_metadata={
            "source": "UCI Machine Learning Repository",
            "dataset": "Iris",
            "task": "classification",
            "instances": len(episodes),
            "source_url": UCI_IRIS_URL,
        },
        evaluator_id="phase177-independent-evaluator",
    )
