from __future__ import annotations

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Iterable, Dict

import numpy as np
import pandas as pd


# Common placeholder tokens that should map to NaN
_PLACEHOLDERS: Iterable[str] = (
    "--", "-", "—", "nan", "NaN", "NAN",
    "null", "NULL", "Null",
    "n/a", "N/A", "na", "NA", "",
)


def _sanitize_key(key: str) -> str:
    """
    Turn an arbitrary cache key into a filesystem-safe filename stem.
    """
    k = str(key).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    if len(k) > 160:
        h = hashlib.md5(k.encode()).hexdigest()[:10]
        k = k[:140] + "_" + h
    return k


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a best-effort to ensure a DataFrame can be written to parquet:
    - Replace common placeholder strings with NaN
    - For object/string columns, attempt numeric conversion if majority numeric
    - Use pandas' nullable StringDtype for remaining text columns
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    for col in df.columns:
        s = df[col]

        # Replace placeholders uniformly as strings where applicable
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            s = s.replace(list(_PLACEHOLDERS), np.nan)

        # Try to convert to numeric when sensible
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            # How many are convertible?
            s_num = pd.to_numeric(s, errors="coerce")
            convertible_ratio = (s_num.notna().sum() / max(1, len(s)))
            if convertible_ratio >= 0.60:
                # Mostly numeric -> adopt numeric
                df[col] = s_num
            else:
                # Keep as string (nullable string dtype is parquet-friendly)
                try:
                    df[col] = s.astype("string")
                except Exception:
                    df[col] = s.astype(object)
        else:
            # Non-string dtypes: just ensure no infinities
            if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
                s = pd.to_numeric(s, errors="coerce")
                s = s.replace([np.inf, -np.inf], np.nan)
                df[col] = s

    return df


class Cache:
    """
    Lightweight on-disk cache for DataFrames and small Python objects.

    Layout:
        <cache_dir>/<key>.parquet      (DataFrame)
        <cache_dir>/<key>.pkl          (pickle)
        <cache_dir>/<key>.json         (json)
    """

    def __init__(self, cache_dir: str | os.PathLike[str]):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Key→path helpers
    # ---------------------------

    def _path_parquet(self, key: str) -> Path:
        return self.root / f"{_sanitize_key(key)}.parquet"

    def _path_pickle(self, key: str) -> Path:
        return self.root / f"{_sanitize_key(key)}.pkl"

    def _path_json(self, key: str) -> Path:
        return self.root / f"{_sanitize_key(key)}.json"

    # ---------------------------
    # Existence
    # ---------------------------

    def exists(self, key: str, kind: Optional[str] = None) -> bool:
        """
        kind: None|'parquet'|'pkl'|'json'
        If kind is None, returns True if any representation exists.
        """
        p_pq = self._path_parquet(key)
        p_pk = self._path_pickle(key)
        p_js = self._path_json(key)
        if kind == "parquet":
            return p_pq.exists()
        if kind == "pkl":
            return p_pk.exists()
        if kind == "json":
            return p_js.exists()
        return p_pq.exists() or p_pk.exists() or p_js.exists()

    # ---------------------------
    # DataFrame I/O
    # ---------------------------

    def save_df(self, key: str, df: pd.DataFrame) -> None:
        """
        Save DataFrame to parquet. Falls back to CSV if parquet fails for any reason.
        """
        p = self._path_parquet(key)
        p.parent.mkdir(parents=True, exist_ok=True)

        try:
            df_clean = _clean_dataframe_for_parquet(df)
            df_clean.to_parquet(p, index=False)
        except Exception as e:
            # Fallback to CSV for maximum robustness
            p_csv = p.with_suffix(".csv")
            try:
                df.to_csv(p_csv, index=False)
            except Exception:
                # As a last resort, pickle
                p_pkl = self._path_pickle(key)
                with open(p_pkl, "wb") as f:
                    pickle.dump(df, f)
            # Optionally log the parquet failure (but avoid noisy prints here)
            # print(f"[Cache.save_df] parquet failed for key={key}: {e}")

    def load_df(self, key: str) -> pd.DataFrame:
        """
        Load DataFrame from parquet if present; fallback to CSV or pickle.
        Returns empty DataFrame if nothing is found.
        """
        p = self._path_parquet(key)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass

        p_csv = p.with_suffix(".csv")
        if p_csv.exists():
            try:
                return pd.read_csv(p_csv)
            except Exception:
                pass

        p_pkl = self._path_pickle(key)
        if p_pkl.exists():
            try:
                with open(p_pkl, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, pd.DataFrame):
                    return obj
            except Exception:
                pass

        # Nothing found
        return pd.DataFrame()

    # ---------------------------
    # Small object I/O
    # ---------------------------

    def save_obj(self, key: str, obj: Any) -> None:
        p = self._path_pickle(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(obj, f)

    def load_obj(self, key: str) -> Any:
        p = self._path_pickle(key)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def save_json(self, key: str, obj: Dict[str, Any]) -> None:
        p = self._path_json(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def load_json(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path_json(key)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
