"""Data loading utilities for the SECOM semiconductor dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd


UCI_SECOM_ZIP_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"


@dataclass(frozen=True)
class SecomData:
    """Container for SECOM features, labels, and optional timestamps."""

    features: pd.DataFrame
    labels: pd.Series
    timestamps: pd.Series | None = None


def ensure_uci_secom_files(data_dir: Path) -> None:
    """Download and extract UCI SECOM files when they are missing."""

    data_dir.mkdir(parents=True, exist_ok=True)
    secom_file = data_dir / "secom.data"
    label_file = data_dir / "secom_labels.data"
    if secom_file.exists() and label_file.exists():
        return

    archive_path = data_dir / "secom.zip"
    urlretrieve(UCI_SECOM_ZIP_URL, archive_path)
    with ZipFile(archive_path) as archive:
        archive.extractall(data_dir)


def _find_file(data_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    available = ", ".join(p.name for p in sorted(data_dir.glob("*")))
    raise FileNotFoundError(f"Expected one of {candidates} in {data_dir}. Found: {available}")


def load_secom(data_dir: str | Path, download: bool = False) -> SecomData:
    """Load SECOM features and labels from UCI-style or Kaggle-style files."""

    data_path = Path(data_dir)
    if download:
        ensure_uci_secom_files(data_path)

    feature_file = _find_file(data_path, ("secom.data", "secom.csv", "uci-secom.csv"))
    label_file = _find_file(data_path, ("secom_labels.data", "secom_labels.csv", "secom_labels.txt"))

    features = pd.read_csv(feature_file, sep=r"\s+|,", engine="python", header=None, na_values=["NaN", "nan", ""])
    labels_raw = pd.read_csv(label_file, sep=r"\s+|,", engine="python", header=None)

    labels = labels_raw.iloc[:, 0].astype(int)
    timestamps = labels_raw.iloc[:, 1] if labels_raw.shape[1] > 1 else None
    features.columns = [f"sensor_{idx:03d}" for idx in range(features.shape[1])]
    labels.name = "fail"

    if len(features) != len(labels):
        raise ValueError(f"Feature rows ({len(features)}) and labels ({len(labels)}) do not match.")

    return SecomData(features=features, labels=labels, timestamps=timestamps)
