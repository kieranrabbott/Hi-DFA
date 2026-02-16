from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd


FractionMap = Dict[str, float]
DataPoint = Tuple[str, float, float, float, FractionMap]
Dataset = List[DataPoint]

_CANONICAL_COLUMNS = [
    "condition",
    "C",
    "tau",
    "age",
    "incomplete",
    "induced",
    "preexisting",
]


def _parse_hours(s: str) -> float:
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else np.nan


def _parse_conc_ug_per_ml(s: str) -> float:
    m = re.search(r"([\d.]+)\s*(?:µ|u)?g/?mL", str(s), flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else np.nan


def _ensure_dataframe(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(Path(source))


def _is_raw_export(df: pd.DataFrame) -> bool:
    required = {"treatment length", "concentration", "culture_age", "class", "mean"}
    return required.issubset(df.columns)


def _build_condition_label(C: float, tau: float, age: float) -> str:
    return f"C{C}_tau{tau}_age{age}"


def _from_raw_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tau"] = df["treatment length"].apply(_parse_hours)
    df["C"] = df["concentration"].apply(_parse_conc_ug_per_ml)
    df["age"] = df["culture_age"].apply(_parse_hours)

    g = df.groupby(["C", "tau", "age", "class"], as_index=False)["mean"].mean()
    wide = g.pivot_table(
        index=["C", "tau", "age"],
        columns="class",
        values="mean",
        fill_value=0.0,
    ).reset_index()

    req = [
        "Incomplete treatment survivor",
        "Antibiotic-induced survivor",
        "Pre-existing persister",
    ]
    for col in req:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["incomplete"] = wide["Incomplete treatment survivor"] / 100.0
    wide["induced"] = wide["Antibiotic-induced survivor"] / 100.0
    wide["preexisting"] = wide["Pre-existing persister"] / 100.0
    wide["condition"] = [
        _build_condition_label(c, t, a)
        for c, t, a in zip(wide["C"], wide["tau"], wide["age"], strict=False)
    ]

    return wide[_CANONICAL_COLUMNS]


def _validate_numeric(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric.")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values.")


def _validate_observation_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "condition" not in out.columns:
        if {"C", "tau", "age"}.issubset(out.columns):
            out["condition"] = [
                _build_condition_label(c, t, a)
                for c, t, a in zip(out["C"], out["tau"], out["age"], strict=False)
            ]
        else:
            raise ValueError(
                "Observation DataFrame must include 'condition' or all of ['C', 'tau', 'age']."
            )

    missing = [c for c in _CANONICAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            "Observation DataFrame missing required columns: " + ", ".join(missing)
        )

    out = out[_CANONICAL_COLUMNS].copy()
    _validate_numeric(out, ["C", "tau", "age", "incomplete", "induced", "preexisting"])

    frac_cols = ["incomplete", "induced", "preexisting"]
    for col in frac_cols:
        bad = (out[col] < 0.0) | (out[col] > 1.0)
        if bad.any():
            raise ValueError(f"Column '{col}' has values outside [0, 1].")

    frac_sum = out[frac_cols].sum(axis=1)
    if (frac_sum > 1.0 + 1e-9).any():
        raise ValueError(
            "Observed fractions must satisfy incomplete + induced + preexisting <= 1 per row."
        )

    return out


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    rows: Dataset = []
    for _, row in df.iterrows():
        rows.append(
            (
                str(row["condition"]),
                float(row["C"]),
                float(row["tau"]),
                float(row["age"]),
                {
                    "incomplete": float(row["incomplete"]),
                    "induced": float(row["induced"]),
                    "preexisting": float(row["preexisting"]),
                },
            )
        )
    return rows


def dataset_to_dataframe(data: Sequence[DataPoint]) -> pd.DataFrame:
    rows = []
    for name, C, tau, age, fr in data:
        rows.append(
            {
                "condition": name,
                "C": float(C),
                "tau": float(tau),
                "age": float(age),
                "incomplete": float(fr["incomplete"]),
                "induced": float(fr["induced"]),
                "preexisting": float(fr["preexisting"]),
            }
        )
    return _validate_observation_df(pd.DataFrame(rows))


def load_data(source: Union[str, Path, pd.DataFrame]) -> Dataset:
    """
    Load observations from either:
    - canonical DataFrame/CSV with columns:
      condition, C, tau, age, incomplete, induced, preexisting
    - raw experimental export used by the legacy notebook.
    """
    df = _ensure_dataframe(source)
    canonical = _from_raw_export(df) if _is_raw_export(df) else df
    canonical = _validate_observation_df(canonical)
    return dataframe_to_dataset(canonical)


def load_data_from_csv(path: Union[str, Path]) -> Dataset:
    """Legacy alias retained for compatibility."""
    return load_data(path)


def _validate_se_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["C", "tau", "age", "se_incomplete", "se_induced", "se_preexisting"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("SE DataFrame missing required columns: " + ", ".join(missing))

    out = df[required].copy()
    _validate_numeric(out, required)

    for col in ["se_incomplete", "se_induced", "se_preexisting"]:
        if (out[col] < 0.0).any():
            raise ValueError(f"SE column '{col}' must be non-negative.")
    return out


def _se_from_raw_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tau"] = df["treatment length"].apply(_parse_hours)
    df["C"] = df["concentration"].apply(_parse_conc_ug_per_ml)
    df["age"] = df["culture_age"].apply(_parse_hours)
    df["n_cells"] = df["total"] * df["mean"] / 100.0

    class_map = {
        "Incomplete treatment survivor": "incomplete",
        "Antibiotic-induced survivor": "induced",
        "Pre-existing persister": "preexisting",
    }

    df = df[df["class"].isin(class_map)].copy()
    df["short_class"] = df["class"].map(class_map)

    records = []
    for (C, tau, age, cls), grp in df.groupby(["C", "tau", "age", "short_class"]):
        N = float(grp["total"].sum())
        p = float(np.clip(grp["n_cells"].sum() / max(N, 1.0), 0.0, 1.0))
        se = float(np.sqrt(p * (1.0 - p) / max(N, 1.0)))
        records.append({"C": C, "tau": tau, "age": age, "class": cls, "se": se})

    se_long = pd.DataFrame(records)
    se_wide = (
        se_long.pivot_table(
            index=["C", "tau", "age"], columns="class", values="se", fill_value=0.0
        )
        .reset_index()
        .rename(
            columns={
                "incomplete": "se_incomplete",
                "induced": "se_induced",
                "preexisting": "se_preexisting",
            }
        )
    )
    se_wide.columns.name = None
    return _validate_se_df(se_wide)


def load_standard_errors(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Load standard errors from either canonical or raw export schema.
    """
    df = _ensure_dataframe(source)
    if _is_raw_export(df):
        return _se_from_raw_export(df)
    return _validate_se_df(df)


def load_se_from_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Legacy alias retained for compatibility."""
    return load_standard_errors(path)
