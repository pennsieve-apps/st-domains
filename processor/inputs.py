"""
Finding the inputs: st-qc's <sample_id>.h5ad files (port `spatial`) and st-prep's
samples.csv (port `metadata`). Linked layout: INPUT_DIR/spatial/ and
INPUT_DIR/metadata/; without links the files are found by name anywhere under
INPUT_DIR (symlinks followed).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("st-domains")
SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class Sample:
    sample_id: str
    path: Path
    row: dict[str, Any] = field(default_factory=dict)


@dataclass
class Inputs:
    samples: list[Sample]
    files_seen: list[Path]
    samples_csv: Path | None
    skipped: dict[str, str]


def walk(root: Path) -> list[Path]:
    """Every file under root, following symlinks (INPUT_DIR entries are symlinks), sorted."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        found.extend(Path(dirpath) / f for f in sorted(filenames) if not f.startswith("."))
    return found


def _find(root: Path, port: str, predicate) -> list[Path]:
    """Files of a port: its own folder when it exists, else anywhere under INPUT_DIR."""
    folder = root / port
    base = folder if folder.is_dir() else root
    return [p for p in walk(base) if predicate(p)]


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def discover(input_dir: Path) -> Inputs:
    h5ads = _find(input_dir, "spatial", lambda p: p.suffix.lower() == ".h5ad")
    csvs = _find(input_dir, "metadata", lambda p: p.name.lower() == "samples.csv")
    samples_csv = csvs[0] if csvs else None
    if len(csvs) > 1:
        log.warning("%d samples.csv files; using %s", len(csvs), samples_csv)
    skipped: dict[str, str] = {}
    by_stem: dict[str, Path] = {}
    for p in h5ads:
        if p.stem in by_stem:
            skipped[str(p)] = f"two files for sample {p.stem}; using {by_stem[p.stem]}"
            continue
        by_stem[p.stem] = p

    samples: list[Sample] = []
    if samples_csv is not None:
        table = pd.read_csv(samples_csv, dtype=str, keep_default_na=False)
        if "sample_id" not in table.columns:
            raise ValueError(f"{samples_csv}: no sample_id column")
        listed: set[str] = set()
        for _, r in table.iterrows():
            row = {k: _text(v) for k, v in r.items()}
            sid = row["sample_id"]
            if not sid or sid in listed:
                continue
            listed.add(sid)
            path = by_stem.get(sid)
            if path is None:
                skipped[sid] = "listed in samples.csv but no <sample_id>.h5ad found"
                continue
            samples.append(Sample(sid, path, row))
        for stem, path in by_stem.items():
            if stem not in listed:
                skipped[stem] = "h5ad not listed in samples.csv"
    else:
        log.warning("no samples.csv: every .h5ad is a sample (sample_id = file stem)")
        for stem, path in sorted(by_stem.items()):
            if not SAMPLE_ID_RE.match(stem):
                skipped[stem] = "file stem is not a valid sample_id ([A-Za-z0-9_-]+)"
                continue
            samples.append(Sample(stem, path, {"sample_id": stem}))
    return Inputs(samples, h5ads, samples_csv, skipped)
