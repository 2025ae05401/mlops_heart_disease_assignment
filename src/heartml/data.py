"""Load UCI Cleveland Heart Disease data into memory; no CSV is saved."""
from __future__ import annotations
import io
import zipfile
from urllib.request import Request, urlopen
import pandas as pd

UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
           "oldpeak","slope","ca","thal","num"]
NUMERIC_FEATURES = ["age","trestbps","chol","thalach","oldpeak"]
CATEGORICAL_FEATURES = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"

def load_uci_cleveland(timeout: int = 60) -> pd.DataFrame:
    """Download UCI ZIP into RAM and parse processed Cleveland data without writing files."""
    request = Request(UCI_ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(f"Unable to fetch UCI archive at {UCI_ARCHIVE_URL}: {exc}") from exc
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            candidates = [n for n in archive.namelist()
                          if n.lower().endswith("processed.cleveland.data")]
            if not candidates:
                raise FileNotFoundError(
                    "processed.cleveland.data not found in UCI archive. "
                    f"Members: {archive.namelist()}"
                )
            with archive.open(candidates[0]) as raw:
                df = pd.read_csv(raw, header=None, names=COLUMNS, na_values="?")
    except zipfile.BadZipFile as exc:
        raise RuntimeError("UCI response was not a valid ZIP archive.") from exc
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = (df["num"] > 0).astype("int64")
    return df.drop(columns=["num"])

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    expected = set(FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    result = df[FEATURES + [TARGET]].copy()
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    result = result.dropna(subset=[TARGET])
    result[TARGET] = result[TARGET].astype(int)
    if not set(result[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")
    if result.empty:
        raise ValueError("Dataset contains no usable rows.")
    return result
