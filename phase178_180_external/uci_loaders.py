from __future__ import annotations

import csv, io, urllib.request, zipfile
from dataclasses import dataclass
from typing import Tuple

WINE_ZIP_URL = "https://archive.ics.uci.edu/static/public/109/wine.zip"
WINE_QUALITY_ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip"

@dataclass(frozen=True)
class Dataset:
    name: str
    features: Tuple[Tuple[float, ...], ...]
    labels: Tuple[int, ...]


def _download_zip(url: str, timeout: float = 30.0) -> zipfile.ZipFile:
    req = urllib.request.Request(url, headers={"User-Agent": "Mirror-7-external-evaluator/178-180"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    return zipfile.ZipFile(io.BytesIO(raw))


def load_wine(timeout: float = 30.0) -> Dataset:
    with _download_zip(WINE_ZIP_URL, timeout) as zf:
        data_name = next(n for n in zf.namelist() if n.endswith("wine.data"))
        rows = []
        for row in csv.reader(io.StringIO(zf.read(data_name).decode("utf-8"))):
            if not row: continue
            rows.append((int(row[0]), tuple(float(x) for x in row[1:])))
    if len(rows) != 178 or any(len(x) != 13 for _, x in rows):
        raise ValueError("UCI Wine integrity check failed")
    return Dataset("uci-wine", tuple(x for _, x in rows), tuple(y for y, _ in rows))


def load_wine_quality(timeout: float = 30.0, variant: str = "red") -> Dataset:
    if variant not in {"red", "white"}: raise ValueError("variant must be red or white")
    with _download_zip(WINE_QUALITY_ZIP_URL, timeout) as zf:
        name = next(n for n in zf.namelist() if n.endswith(f"winequality-{variant}.csv"))
        text = zf.read(name).decode("utf-8")
    reader = csv.reader(io.StringIO(text), delimiter=";")
    header = next(reader)
    if header[-1] != "quality" or len(header) != 12:
        raise ValueError("unexpected Wine Quality schema")
    rows = [(tuple(float(x) for x in row[:-1]), int(row[-1])) for row in reader if row]
    expected = 1599 if variant == "red" else 4898
    if len(rows) != expected or any(len(x) != 11 for x, _ in rows):
        raise ValueError("UCI Wine Quality integrity check failed")
    return Dataset(f"uci-wine-quality-{variant}", tuple(x for x, _ in rows), tuple(y for _, y in rows))
