from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from .config import RunConfig

warnings.filterwarnings(
    "ignore",
    message=r"No Pauling electronegativity for .*",
    category=UserWarning,
    module=r"pymatgen\.core\.periodic_table",
)

@dataclass
class PreparedData:
    dataset_name: str
    target: str
    task: str
    X: pd.DataFrame
    y: pd.Series
    numeric_cols: list[str]
    categorical_cols: list[str]
    dropped_report: pd.DataFrame
    info: dict[str, Any]


def get_matminer_dataset_functions() -> tuple[Any, Any]:
    from matminer.datasets import get_available_datasets as _get_available_datasets, load_dataset

    def _safe_get_available_datasets() -> list[str]:
        import contextlib
        import io

        try:
            return _get_available_datasets()
        except UnicodeEncodeError:
            # Some Windows consoles fail when matminer prints BOM/unicode dataset metadata.
            with contextlib.redirect_stdout(io.StringIO()):
                return _get_available_datasets()

    return _safe_get_available_datasets, load_dataset


@contextlib.contextmanager
def _silence_native_stderr(enabled: bool = True):
    if not enabled:
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    dup_fd: int | None = None
    try:
        dup_fd = os.dup(stderr_fd)
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), stderr_fd)
            yield
    except Exception:
        yield
    finally:
        if dup_fd is not None:
            try:
                os.dup2(dup_fd, stderr_fd)
            except Exception:
                pass
            try:
                os.close(dup_fd)
            except Exception:
                pass



def _featurize_dataframe_with_progress(
    df: pd.DataFrame,
    featurizer: Any,
    col_id: str,
    logger: logging.Logger,
    label: str,
    chunk_size: int = 250,
    silence_native_stderr: bool = False,
) -> pd.DataFrame:
    total = int(len(df))
    if total == 0:
        return df.copy()

    chunk_size = max(1, int(chunk_size))
    logger.info("Starting %s featurization on %d rows (chunk_size=%d).", label, total, chunk_size)
    overall_start = time.perf_counter()
    progress_row_step = max(chunk_size, max(1, total // 25))
    progress_time_step = 15.0
    last_logged_rows = 0
    last_logged_elapsed = 0.0

    if hasattr(featurizer, "set_n_jobs"):
        try:
            current_n_jobs = getattr(featurizer, "n_jobs", None)
            if current_n_jobs != 1:
                featurizer.set_n_jobs(1)
                logger.info("%s featurizer n_jobs forced to 1 (was %s).", label, current_n_jobs)
        except Exception:
            pass
    if hasattr(featurizer, "set_chunksize"):
        try:
            featurizer.set_chunksize(max(1, min(int(chunk_size), 32)))
        except Exception:
            pass

    if total <= chunk_size:
        with _silence_native_stderr(silence_native_stderr):
            out = featurizer.featurize_dataframe(df.copy(), col_id=col_id, ignore_errors=True, pbar=False)
        elapsed = time.perf_counter() - overall_start
        logger.info("Finished %s featurization: %d/%d rows | elapsed=%.1fs.", label, total, total, elapsed)
        return out

    parts: list[pd.DataFrame] = []
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk = df.iloc[start:stop].copy()
        chunk_start = time.perf_counter()
        with _silence_native_stderr(silence_native_stderr):
            chunk_out = featurizer.featurize_dataframe(chunk, col_id=col_id, ignore_errors=True, pbar=False)
        parts.append(chunk_out)
        elapsed = time.perf_counter() - overall_start
        chunk_seconds = time.perf_counter() - chunk_start
        rows_done = stop
        rows_left = max(0, total - rows_done)
        rate = rows_done / max(elapsed, 1e-9)
        eta_seconds = rows_left / max(rate, 1e-9)
        should_log_progress = (
            rows_done == total
            or last_logged_rows == 0
            or (rows_done - last_logged_rows) >= progress_row_step
            or (elapsed - last_logged_elapsed) >= progress_time_step
        )
        if should_log_progress:
            logger.info(
                "%s featurization progress: %d/%d rows | last_chunk=%.1fs | elapsed=%.1fs | eta=%.1fs.",
                label,
                rows_done,
                total,
                chunk_seconds,
                elapsed,
                eta_seconds,
            )
            last_logged_rows = rows_done
            last_logged_elapsed = elapsed

    out = pd.concat(parts, axis=0)
    out = out.loc[df.index]
    total_elapsed = time.perf_counter() - overall_start
    logger.info("Finished %s featurization: %d/%d rows | elapsed=%.1fs.", label, total, total, total_elapsed)
    return out


def _precheck_featurizer_rows(
    df: pd.DataFrame,
    featurizer: Any,
    col_id: str,
    logger: logging.Logger,
    label: str,
) -> tuple[pd.DataFrame, list[Any]]:
    if not hasattr(featurizer, "precheck_dataframe"):
        return df.copy(), []

    logger.info("Running %s precheck on %d rows.", label, int(len(df)))
    try:
        precheck_input = df[[col_id]].copy().reset_index(drop=True)
        precheck = featurizer.precheck_dataframe(precheck_input, col_id, return_frac=False, inplace=False)
    except Exception as exc:
        logger.warning("%s precheck failed; proceeding without precheck. Error: %s", label, exc)
        return df.copy(), []

    if isinstance(precheck, pd.DataFrame):
        expected_col = f"{featurizer.__class__.__name__} precheck pass"
        if expected_col in precheck.columns:
            mask = pd.Series(precheck[expected_col].values, index=df.index)
        else:
            bool_cols = [c for c in precheck.columns if pd.api.types.is_bool_dtype(precheck[c])]
            if not bool_cols:
                bool_cols = [
                    c
                    for c in precheck.columns
                    if bool(precheck[c].dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all())
                ]
            if bool_cols:
                mask = pd.Series(precheck[bool_cols[-1]].values, index=df.index)
            else:
                logger.warning(
                    "%s precheck dataframe did not contain a boolean pass/fail column; proceeding without precheck.",
                    label,
                )
                return df.copy(), []
    elif isinstance(precheck, pd.Series):
        mask = precheck.reindex(df.index)
    elif isinstance(precheck, (list, tuple, np.ndarray)) and len(precheck) == len(df):
        mask = pd.Series(precheck, index=df.index)
    elif isinstance(precheck, (bool, np.bool_)):
        mask = pd.Series([bool(precheck)] * len(df), index=df.index)
    else:
        logger.warning(
            "%s precheck returned unsupported type %s; proceeding without precheck.",
            label,
            type(precheck).__name__,
        )
        return df.copy(), []

    mask = mask.fillna(False).astype(bool)
    skipped_rows = df.index[~mask].tolist()
    supported = int(mask.sum())
    skipped = int(len(df) - supported)
    logger.info("%s precheck: %d/%d rows supported; %d skipped.", label, supported, int(len(df)), skipped)
    if skipped_rows:
        logger.warning(
            "%s precheck skipped %d row(s). First skipped row indices: %s",
            label,
            skipped,
            skipped_rows[:10],
        )
    return df.loc[mask].copy(), skipped_rows


def _merge_featurized_subset(base_df: pd.DataFrame, featurized_subset: pd.DataFrame) -> pd.DataFrame:
    if featurized_subset.empty:
        return base_df.copy()

    out = base_df.copy()
    for col in featurized_subset.columns:
        if col not in out.columns:
            out[col] = np.nan
    out.loc[featurized_subset.index, featurized_subset.columns.tolist()] = featurized_subset
    return out

def infer_task_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target) and int(target.nunique(dropna=True)) > 20:
        return "regression"
    return "classification"


def _preview_rows(df: pd.DataFrame, sample_n: int, random_state: int) -> pd.DataFrame:
    return df.sample(min(sample_n, len(df)), random_state=random_state) if len(df) > 0 else df.copy()


def _safe_scalar_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        missing = pd.isna(v)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except Exception:
        return False
    return False


def _safe_cell_to_text(v: Any) -> str:
    if _safe_scalar_missing(v):
        return "<NA>"
    return str(v)


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        # Some object values (e.g., pymatgen Structure) are unhashable for pandas nunique.
        # Fallback to string representation for robust profiling-only cardinality reporting.
        return int(series.map(_safe_cell_to_text).astype("string").nunique(dropna=True))


def make_csv_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if (
            pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s)
            or pd.api.types.is_timedelta64_dtype(s)
        ):
            continue
        out[col] = s.map(_safe_cell_to_text)
    return out


def export_csv_safe(
    df: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
    index_label: str | None = None,
) -> None:
    make_csv_safe_frame(df).to_csv(path, index=index, index_label=index_label)


def _export_full_raw_dataset(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    export_csv_safe(df, output_dir / "dataset_raw_full.csv", index=False)

    try:
        df.to_pickle(output_dir / "dataset_raw_full.pkl")
    except Exception as exc:
        logger.warning("Could not save dataset_raw_full.pkl: %s", exc)


def preview_dataset(df: pd.DataFrame, output_dir: Path, random_state: int, logger: logging.Logger) -> pd.DataFrame:
    _ = random_state  # kept for signature compatibility
    _export_full_raw_dataset(df, output_dir, logger)

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(item) for item in df.dtypes],
            "missing_pct": [float(df[col].isna().mean() * 100.0) for col in df.columns],
            "n_unique": [_safe_nunique(df[col]) for col in df.columns],
            "example": [
                str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "<NA>"
                for col in df.columns
            ],
        }
    ).sort_values(["missing_pct", "n_unique"], ascending=[False, False])
    summary.to_csv(output_dir / "dataset_column_summary.csv", index=False)

    logger.info("Saved full raw dataset and column summary.")
    return summary


def _try_parse_formula_with_pymatgen(text: str) -> dict[str, float] | None:
    try:
        from pymatgen.core import Composition, Element
    except Exception:
        return None

    try:
        comp = Composition(text)
    except Exception:
        return None

    if comp.num_atoms <= 0:
        return None

    elements = list(comp.get_el_amt_dict().items())
    total_atoms = float(sum(v for _, v in elements))
    fractions = {el: amt / total_atoms for el, amt in elements}

    z_values = []
    rows = []
    groups = []
    metal_fraction = 0.0
    for el, frac in fractions.items():
        token = str(el).strip()
        if re.fullmatch(r"[A-Z][a-z]?", token) is None:
            return None
        try:
            e = Element(token)
        except Exception:
            return None

        z_values.append(e.Z * frac)
        rows.append(float(e.row) * frac if e.row is not None else 0.0)
        groups.append(float(e.group) * frac if e.group is not None else 0.0)
        metal_fraction += frac if e.is_metal else 0.0

    return {
        "n_elements": float(len(elements)),
        "total_atoms": float(total_atoms),
        "molar_mass": float(comp.weight),
        "mean_Z": float(sum(z_values)),
        "mean_period": float(sum(rows)),
        "mean_group": float(sum(groups)),
        "metal_fraction": float(metal_fraction),
        "nonmetal_fraction": float(1.0 - metal_fraction),
    }


def _try_parse_formula_with_ase(text: str) -> dict[str, float] | None:
    try:
        from ase.formula import Formula
    except Exception:
        return None

    try:
        formula = Formula(text)
        counts = formula.count()
    except Exception:
        return None

    total_atoms = float(sum(counts.values()))
    if total_atoms <= 0:
        return None

    return {
        "ase_total_atoms": total_atoms,
        "ase_n_elements": float(len(counts)),
    }


def _detect_formula_like_column(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return False

    sample = values.sample(min(50, len(values)), random_state=0)
    attempted = 0
    ok = 0
    for value in sample:
        text = str(value).strip()
        # Allow longer alloy/composition strings (common in steel/alloy datasets).
        if text == "" or len(text) > 320:
            continue
        if re.fullmatch(r"[A-Za-z0-9().+\-]+", text) is None:
            continue

        attempted += 1
        try:
            hit = _try_parse_formula_with_pymatgen(text)
        except Exception:
            hit = None

        if hit is not None:
            ok += 1

    if attempted == 0:
        return False
    return (ok / attempted) >= 0.6


def enrich_material_descriptors(
    df: pd.DataFrame,
    target_col: str,
    logger: logging.Logger,
    memory: Memory | None,
    enabled: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": enabled,
        "applied": False,
        "formula_columns": [],
        "generated_columns": [],
    }
    if not enabled:
        return df, info

    if memory is None:
        return _enrich_material_descriptors_impl(df, target_col, logger, info)

    cached = memory.cache(_enrich_material_descriptors_impl)
    return cached(df, target_col, logger, info)


def _enrich_material_descriptors_impl(
    df: pd.DataFrame,
    target_col: str,
    logger: logging.Logger,
    info: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    candidate_cols = [col for col in out.columns if col != target_col and out[col].dtype == "object"]

    for col in candidate_cols:
        try:
            is_formula_like = _detect_formula_like_column(out[col])
        except Exception as exc:
            logger.warning("Formula-like detection failed for column %s: %s", col, exc)
            continue

        if not is_formula_like:
            continue

        info["formula_columns"].append(col)
        values = out[col].astype(str)

        feature_rows: list[dict[str, float]] = []
        element_fraction_rows: list[dict[str, float]] = []
        element_tokens: set[str] = set()
        for value in values:
            py_desc = _try_parse_formula_with_pymatgen(value) or {}
            ase_desc = _try_parse_formula_with_ase(value) or {}
            merged = {**py_desc, **ase_desc}
            feature_rows.append(merged)

            # Also export explicit elemental fractions from parsed composition so
            # users can inspect chemistry-derived features directly.
            comp = _to_composition(value)
            frac_row: dict[str, float] = {}
            if comp is not None:
                try:
                    amounts = comp.get_el_amt_dict()
                    total_atoms = float(sum(float(v) for v in amounts.values()))
                    if total_atoms > 0:
                        for el, amt in amounts.items():
                            token = str(el).strip()
                            if re.fullmatch(r"[A-Z][a-z]?", token) is None:
                                continue
                            frac = float(amt) / total_atoms
                            if np.isfinite(frac):
                                frac_row[token] = frac
                                element_tokens.add(token)
                except Exception:
                    frac_row = {}
            element_fraction_rows.append(frac_row)

        new_frames: list[pd.DataFrame] = []

        desc_df = pd.DataFrame(feature_rows, index=out.index)
        if not desc_df.empty:
            desc_df.columns = [f"{col}__{name}" for name in desc_df.columns]
            desc_df = desc_df.apply(pd.to_numeric, errors="coerce")
            new_frames.append(desc_df)

        if element_tokens:
            ordered_tokens = sorted(element_tokens)
            frac_df = pd.DataFrame(
                {
                    f"{col}__el_frac_{el}": [row.get(el, 0.0) for row in element_fraction_rows]
                    for el in ordered_tokens
                },
                index=out.index,
            )
            frac_df = frac_df.apply(pd.to_numeric, errors="coerce")
            new_frames.append(frac_df)

        if not new_frames:
            continue

        added = pd.concat(new_frames, axis=1)
        out = pd.concat([out, added], axis=1)
        info["generated_columns"].extend(added.columns.tolist())

    if info["generated_columns"]:
        info["applied"] = True
        logger.info("Generated %d material descriptor columns.", len(info["generated_columns"]))
    else:
        logger.info("No formula-like categorical columns detected for material descriptor enrichment.")

    return out, info


def apply_matminer_featurizers(
    df: pd.DataFrame,
    logger: logging.Logger,
    enabled: bool,
    memory: Memory | None,
    composition_chunk_size: int = 1000,
    structure_chunk_size: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": enabled,
        "applied": False,
        "composition_column": None,
        "structure_column": None,
        "generated_columns": [],
        "row_failures": [],
        "errors": [],
        "skipped_featurizers": [],
    }
    if not enabled:
        return df, info

    if memory is None:
        return _apply_matminer_featurizers_impl(df, logger, info, composition_chunk_size, structure_chunk_size)

    cached = memory.cache(_apply_matminer_featurizers_impl)
    return cached(df, logger, info, composition_chunk_size, structure_chunk_size)


def _apply_matminer_featurizers_impl(
    df: pd.DataFrame,
    logger: logging.Logger,
    info: dict[str, Any],
    composition_chunk_size: int,
    structure_chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    before = set(out.columns)

    composition_col = None
    structure_col = None

    for col in out.columns:
        values = out[col].dropna()
        if values.empty:
            continue
        first = values.iloc[0]
        cls = f"{type(first).__module__}.{type(first).__name__}".lower()
        if composition_col is None and "composition" in cls:
            composition_col = col
        if structure_col is None and "structure" in cls:
            structure_col = col

    if composition_col is not None:
        info["composition_column"] = composition_col
        try:
            from matminer.featurizers.composition import ElementProperty

            featurizer = ElementProperty.from_preset("magpie")
            out = _featurize_dataframe_with_progress(out, featurizer, composition_col, logger, "composition:ElementProperty", chunk_size=composition_chunk_size)
            info["applied"] = True
            logger.info("Applied ElementProperty featurizer on %s", composition_col)
        except Exception as exc:
            logger.warning("Composition featurization failed: %s", exc)

    if structure_col is not None:
        info["structure_column"] = structure_col
        try:
            from matminer.featurizers.structure import DensityFeatures

            featurizer = DensityFeatures()
            work_df, skipped_rows = _precheck_featurizer_rows(out, featurizer, structure_col, logger, "structure:DensityFeatures")
            for row_idx in skipped_rows:
                info["row_failures"].append(
                    {
                        "row_index": row_idx,
                        "stage": "structure",
                        "featurizer": "DensityFeatures",
                        "reason": "precheck_unsupported_row",
                    }
                )
            if skipped_rows and not len(work_df):
                info["skipped_featurizers"].append("structure:DensityFeatures:precheck_no_supported_rows")
                logger.warning("Skipping DensityFeatures because no rows passed precheck.")
            else:
                featurized_subset = _featurize_dataframe_with_progress(
                    work_df,
                    featurizer,
                    structure_col,
                    logger,
                    "structure:DensityFeatures",
                    chunk_size=structure_chunk_size,
                    silence_native_stderr=True,
                )
                out = _merge_featurized_subset(out, featurized_subset)
                info["applied"] = True
                logger.info("Applied DensityFeatures featurizer on %s", structure_col)
        except Exception as exc:
            logger.warning("Structure featurization failed: %s", exc)

    info["generated_columns"] = sorted(set(out.columns) - before)
    return out, info


def _coerce_numeric_if_possible(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if float(converted.notna().mean()) >= 0.95:
        return converted
    return series


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    cfg: RunConfig,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], pd.DataFrame]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    data = df.copy()
    y_raw = data[target_col]
    X = data.drop(columns=[target_col])

    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate feature columns detected before preprocessing: {dupes[:10]}")

    y = pd.to_numeric(y_raw, errors="coerce") if task == "regression" else y_raw.astype("string")
    y = y.replace("nan", pd.NA)
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    if len(y) == 0:
        raise RuntimeError("No valid target values remain after NA filtering.")
    if not y.index.is_unique:
        raise RuntimeError("Target index must be unique after filtering.")
    if len(X) != len(y):
        raise RuntimeError("Feature and target rows are misaligned after filtering.")

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = _coerce_numeric_if_possible(X[col])

    dropped: list[dict[str, str]] = []

    for col in list(X.columns):
        if col in cfg.drop_columns:
            X.drop(columns=[col], inplace=True)
            dropped.append({"column": col, "reason": "user_dropped"})

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        series = X[col]

        if pd.api.types.is_numeric_dtype(series):
            X[col] = series.replace([np.inf, -np.inf], np.nan)
            numeric_cols.append(col)
            continue

        nunique = int(series.astype("string").nunique(dropna=True))
        unique_ratio = nunique / max(1, len(series))

        if col not in cfg.include_columns and unique_ratio > 0.98:
            dropped.append({"column": col, "reason": "likely_identifier"})
            continue

        if col in cfg.include_columns or nunique <= cfg.max_categorical_cardinality:
            categorical_cols.append(col)
            cat_series = series.astype("string").replace("nan", pd.NA)
            X[col] = cat_series.fillna("__MISSING__").astype(str).replace("__MISSING__", np.nan).astype("object")
        else:
            dropped.append(
                {
                    "column": col,
                    "reason": f"categorical_cardinality>{cfg.max_categorical_cardinality}",
                }
            )

    keep_cols = numeric_cols + categorical_cols
    X = X[keep_cols]

    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate feature columns detected: {dupes[:10]}")

    all_missing = [col for col in keep_cols if X[col].isna().all() and col not in cfg.include_columns]
    if all_missing:
        X = X.drop(columns=all_missing)
        numeric_cols = [col for col in numeric_cols if col not in all_missing]
        categorical_cols = [col for col in categorical_cols if col not in all_missing]
        for col in all_missing:
            dropped.append({"column": col, "reason": "all_missing"})

    if target_col in X.columns:
        raise RuntimeError("Target leakage detected: target column still present in features.")

    if not numeric_cols and not categorical_cols:
        raise RuntimeError("No usable features left after preprocessing.")

    if not X.index.equals(y.index):
        raise RuntimeError("Feature and target indices are not aligned.")

    dropped_df = pd.DataFrame(dropped).drop_duplicates(subset=["column", "reason"]) if dropped else pd.DataFrame(columns=["column", "reason"])
    return X, y, numeric_cols, categorical_cols, dropped_df

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    imputation: str = "simple",
    scaling: str = "standard",
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_cols:
        num_steps: list[tuple[str, Any]] = []

        if imputation == "knn":
            num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
        elif imputation == "none":
            pass
        else:
            num_steps.append(("imputer", SimpleImputer(strategy="median")))

        if scaling == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif scaling == "none":
            pass
        else:
            num_steps.append(("scaler", StandardScaler()))

        num_pipe = Pipeline(steps=num_steps) if num_steps else Pipeline(steps=[("identity", "passthrough")])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _validate_split_integrity(
    X_train: pd.DataFrame,
    X_calib: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_calib: pd.Series,
    y_test: pd.Series,
    g_train: pd.Series | None = None,
    g_calib: pd.Series | None = None,
    g_test: pd.Series | None = None,
) -> None:
    if min(len(X_train), len(X_calib), len(X_test)) <= 0:
        raise RuntimeError("Split produced an empty train/calibration/test partition.")

    idx_train = set(X_train.index)
    idx_cal = set(X_calib.index)
    idx_test = set(X_test.index)

    if idx_train & idx_cal:
        raise RuntimeError("Data leakage detected: train/calibration sample overlap.")
    if idx_train & idx_test:
        raise RuntimeError("Data leakage detected: train/test sample overlap.")
    if idx_cal & idx_test:
        raise RuntimeError("Data leakage detected: calibration/test sample overlap.")

    if len(X_train) != len(y_train) or len(X_calib) != len(y_calib) or len(X_test) != len(y_test):
        raise RuntimeError("X/y split size mismatch detected.")

    if g_train is not None and g_calib is not None and g_test is not None:
        set_train = set(g_train.astype("string").tolist())
        set_cal = set(g_calib.astype("string").tolist())
        set_test = set(g_test.astype("string").tolist())
        if set_train & set_cal:
            raise RuntimeError("Group leakage detected: train/calibration groups overlap.")
        if set_train & set_test:
            raise RuntimeError("Group leakage detected: train/test groups overlap.")
        if set_cal & set_test:
            raise RuntimeError("Group leakage detected: calibration/test groups overlap.")


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    test_size: float,
    calibration_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    stratify = y if task == "classification" and y.nunique() > 1 else None

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    stratify_cal = y_train_all if task == "classification" and y_train_all.nunique() > 1 else None

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_all,
        y_train_all,
        test_size=calibration_size,
        random_state=random_state,
        stratify=stratify_cal,
    )

    _validate_split_integrity(X_train, X_calib, X_test, y_train, y_calib, y_test)
    return X_train, X_calib, X_test, y_train, y_calib, y_test

def detect_group_column(df: pd.DataFrame, target_col: str, cfg: RunConfig, logger: logging.Logger) -> str | None:
    if cfg.group_column:
        if cfg.group_column in df.columns and cfg.group_column != target_col:
            logger.info("Using user-provided group column: %s", cfg.group_column)
            return cfg.group_column
        logger.warning("Requested group column '%s' not found or equals target.", cfg.group_column)

    candidates = [
        "material_id",
        "materialid",
        "formula",
        "reduced_formula",
        "pretty_formula",
        "composition",
        "structure_id",
        "mp_id",
        "id",
        "identifier",
    ]

    for col in df.columns:
        if col == target_col:
            continue
        low = col.lower()
        if low in candidates or low.endswith("_id"):
            nunique = int(df[col].astype("string").nunique(dropna=True))
            if 2 <= nunique <= max(2, int(len(df) * 0.98)):
                logger.info("Auto-detected group column: %s", col)
                return col

    logger.info("No suitable group column detected; using standard random split/CV.")
    return None


def split_dataset_with_groups(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    task: str,
    test_size: float,
    calibration_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series | None,
    pd.Series | None,
    pd.Series | None,
]:
    if groups is None:
        X_train, X_calib, X_test, y_train, y_calib, y_test = split_dataset(
            X,
            y,
            task,
            test_size,
            calibration_size,
            random_state,
        )
        _validate_split_integrity(X_train, X_calib, X_test, y_train, y_calib, y_test)
        return X_train, X_calib, X_test, y_train, y_calib, y_test, None, None, None

    from sklearn.model_selection import GroupShuffleSplit

    grp = groups.astype("string").fillna("<NA>")

    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_all_idx, test_idx = next(first.split(X, y, groups=grp))

    X_train_all = X.iloc[train_all_idx]
    y_train_all = y.iloc[train_all_idx]
    g_train_all = grp.iloc[train_all_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    g_test = grp.iloc[test_idx]

    second = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=random_state + 1)
    train_idx, calib_idx = next(second.split(X_train_all, y_train_all, groups=g_train_all))

    X_train = X_train_all.iloc[train_idx]
    y_train = y_train_all.iloc[train_idx]
    g_train = g_train_all.iloc[train_idx]

    X_calib = X_train_all.iloc[calib_idx]
    y_calib = y_train_all.iloc[calib_idx]
    g_calib = g_train_all.iloc[calib_idx]

    _validate_split_integrity(X_train, X_calib, X_test, y_train, y_calib, y_test, g_train, g_calib, g_test)
    return X_train, X_calib, X_test, y_train, y_calib, y_test, g_train, g_calib, g_test

def _flatten_dict_columns(df: pd.DataFrame, cols: list[str], logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    generated: list[str] = []

    for col in cols:
        if col not in out.columns:
            continue
        try:
            normalized = pd.json_normalize(out[col]).add_prefix(f"{col}__")
            out = pd.concat([out.drop(columns=[col]), normalized], axis=1)
            generated.extend(normalized.columns.tolist())
            logger.info("Flattened dict column %s into %d columns.", col, len(normalized.columns))
        except Exception as exc:
            logger.warning("Could not flatten dict column %s: %s", col, exc)

    return out, generated


def _estimate_valence_electrons(element_obj: Any) -> float:
    try:
        full = getattr(element_obj, "full_electronic_structure", [])
        if not full:
            return float("nan")
        max_n = max([int(item[0]) for item in full])
        return float(sum([float(item[2]) for item in full if int(item[0]) == max_n]))
    except Exception:
        return float("nan")


def _add_element_descriptor_columns(
    df: pd.DataFrame,
    element_cols: list[str],
    logger: logging.Logger,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created: list[str] = []

    try:
        from pymatgen.core import Element
    except Exception:
        logger.warning("pymatgen not available for elemental descriptor mapping.")
        return out, created

    wanted = set([str(v).strip().lower() for v in (selected_features or []) if str(v).strip()])

    feature_map: dict[str, tuple[str, Any]] = {
        "atomic_number": ("Z", lambda e: float(e.Z)),
        "z": ("Z", lambda e: float(e.Z)),
        "electronegativity": ("X", lambda e: float(e.X) if e.X is not None else float("nan")),
        "x": ("X", lambda e: float(e.X) if e.X is not None else float("nan")),
        "atomic_radius": ("atomic_radius", lambda e: float(getattr(e, "atomic_radius", np.nan)) if getattr(e, "atomic_radius", None) is not None else float("nan")),
        "covalent_radius": ("covalent_radius", lambda e: float(getattr(e, "covalent_radius", np.nan)) if getattr(e, "covalent_radius", None) is not None else float("nan")),
        "row": ("row", lambda e: float(e.row) if e.row is not None else float("nan")),
        "group": ("group", lambda e: float(e.group) if e.group is not None else float("nan")),
        "mendeleev_no": ("mendeleev_no", lambda e: float(e.mendeleev_no) if e.mendeleev_no is not None else float("nan")),
        "atomic_mass": ("atomic_mass", lambda e: float(e.atomic_mass) if getattr(e, "atomic_mass", None) is not None else float("nan")),
        "valence_electrons": ("valence_electrons", lambda e: _estimate_valence_electrons(e)),
        "block": ("block", lambda e: {"s": 1.0, "p": 2.0, "d": 3.0, "f": 4.0}.get(str(getattr(e, "block", "")), float("nan"))),
        "electron_affinity_if_available": ("electron_affinity", lambda e: float(getattr(e, "electron_affinity", np.nan)) if getattr(e, "electron_affinity", None) is not None else float("nan")),
        "first_ionization_energy_if_available": ("first_ionization_energy", lambda e: float(getattr(e, "ionization_energy", np.nan)) if getattr(e, "ionization_energy", None) is not None else float("nan")),
    }

    default_order = ["z", "x", "atomic_radius", "row", "group", "mendeleev_no", "valence_electrons", "block"]
    keys_to_use = default_order if not wanted else [k for k in feature_map.keys() if k in wanted]

    for col in element_cols:
        if col not in out.columns:
            continue

        values = out[col].astype("string")
        mapped: dict[str, list[float]] = {}
        for key in keys_to_use:
            suffix = feature_map[key][0]
            mapped[f"{col}__{suffix}"] = []

        for item in values:
            try:
                e = Element(str(item))
                for key in keys_to_use:
                    suffix, fn = feature_map[key]
                    try:
                        mapped[f"{col}__{suffix}"].append(float(fn(e)))
                    except Exception:
                        mapped[f"{col}__{suffix}"].append(float("nan"))
            except Exception:
                for mk in mapped:
                    mapped[mk].append(float("nan"))

        for k, v in mapped.items():
            out[k] = pd.to_numeric(pd.Series(v), errors="coerce")
            created.append(k)

    def _pair_ops(c1: str, c2: str, prop: str) -> None:
        a = f"{c1}__{prop}"
        b = f"{c2}__{prop}"
        if a in out.columns and b in out.columns:
            d = f"pairdiff__{c1}__{c2}__{prop}"
            r = f"pairratio__{c1}__{c2}__{prop}"
            out[d] = out[a] - out[b]
            out[r] = out[a] / (out[b].abs() + 1e-12)
            created.extend([d, r])

    if len(element_cols) >= 2:
        pair_props = [
            p
            for p in ["Z", "X", "atomic_radius", "covalent_radius", "row", "group", "valence_electrons", "atomic_mass"]
            if any(f"__{p}" in c for c in created)
        ]
        for i in range(len(element_cols)):
            for j in range(i + 1, len(element_cols)):
                c1 = element_cols[i]
                c2 = element_cols[j]
                for prop in pair_props:
                    _pair_ops(c1, c2, prop)

    if created:
        logger.info("Generated %d site-wise elemental descriptor columns.", len(created))
    return out, created

def _to_composition(value: Any) -> Any:
    try:
        from pymatgen.core import Composition
    except Exception:
        return None

    if value is None:
        return None

    if value.__class__.__name__.lower() == "composition":
        return value

    if isinstance(value, dict):
        try:
            return Composition(value)
        except Exception:
            return None

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return None

    if text.startswith("{") and text.endswith("}"):
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return Composition(obj)
        except Exception:
            pass

    try:
        return Composition(text)
    except Exception:
        return None


def _parse_formula_column(
    df: pd.DataFrame,
    formula_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    out = df.copy()
    if formula_col not in out.columns:
        return out, None, []

    try:
        from pymatgen.core import Composition
    except Exception:
        logger.warning("pymatgen not available for formula parsing.")
        return out, None, []

    compositions: list[Any] = []
    reduced: list[str | None] = []
    systems: list[str | None] = []

    for value in out[formula_col].tolist():
        comp = _to_composition(value)
        if comp is None:
            compositions.append(None)
            reduced.append(None)
            systems.append(None)
            continue

        try:
            comp = comp if isinstance(comp, Composition) else Composition(comp)
            compositions.append(comp)
            reduced.append(comp.reduced_formula)
            systems.append("-".join(sorted([str(el) for el in comp.elements])))
        except Exception:
            compositions.append(None)
            reduced.append(None)
            systems.append(None)

    out["__composition__"] = compositions
    out["__reduced_formula__"] = pd.Series(reduced, dtype="string")
    out["__chemical_system__"] = pd.Series(systems, dtype="string")

    logger.info("Parsed formula column %s into composition/reduced formula/system columns.", formula_col)
    return out, "__composition__", ["__composition__", "__reduced_formula__", "__chemical_system__"]


def _to_structure(value: Any) -> Any:
    try:
        from pymatgen.core import Structure
    except Exception:
        return None

    if value is None:
        return None

    if value.__class__.__name__.lower() == "structure":
        return value

    if isinstance(value, dict):
        try:
            return Structure.from_dict(value)
        except Exception:
            return None

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return None

    parsed_dict: dict[str, Any] | None = None
    if text.startswith("{") and text.endswith("}"):
        try:
            candidate = json.loads(text)
            if isinstance(candidate, dict):
                parsed_dict = candidate
        except Exception:
            try:
                candidate = ast.literal_eval(text)
                if isinstance(candidate, dict):
                    parsed_dict = candidate
            except Exception:
                parsed_dict = None

    if parsed_dict is not None:
        try:
            return Structure.from_dict(parsed_dict)
        except Exception:
            pass

    try:
        return Structure.from_str(text, fmt="cif")
    except Exception:
        return None


def _parse_structure_column(
    df: pd.DataFrame,
    structure_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    out = df.copy()
    if structure_col not in out.columns:
        return out, None, []

    parsed = [_to_structure(value) for value in out[structure_col].tolist()]
    valid_count = int(sum(item is not None for item in parsed))

    if valid_count == 0:
        logger.warning("Could not parse any entries in structure column %s.", structure_col)
        return out, None, []

    out["__structure__"] = parsed
    logger.info("Parsed structure column %s into %d valid pymatgen Structure rows.", structure_col, valid_count)
    return out, "__structure__", ["__structure__"]


def _derive_composition_from_structure(
    df: pd.DataFrame,
    structure_col: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    out = df.copy()
    if structure_col not in out.columns:
        return out, None, []

    compositions: list[Any] = []
    reduced: list[str | None] = []
    systems: list[str | None] = []

    for value in out[structure_col].tolist():
        if value is None:
            compositions.append(None)
            reduced.append(None)
            systems.append(None)
            continue

        try:
            comp = value.composition
            compositions.append(comp)
            reduced.append(comp.reduced_formula)
            systems.append("-".join(sorted([str(el) for el in comp.elements])))
        except Exception:
            compositions.append(None)
            reduced.append(None)
            systems.append(None)

    generated: list[str] = []

    if "__composition__" not in out.columns:
        out["__composition__"] = compositions
        generated.append("__composition__")
    else:
        current = out["__composition__"].tolist()
        out["__composition__"] = [
            current[i] if current[i] is not None else compositions[i]
            for i in range(len(current))
        ]

    if "__reduced_formula__" not in out.columns:
        out["__reduced_formula__"] = pd.Series(reduced, dtype="string")
        generated.append("__reduced_formula__")

    if "__chemical_system__" not in out.columns:
        out["__chemical_system__"] = pd.Series(systems, dtype="string")
        generated.append("__chemical_system__")

    logger.info("Derived composition metadata from structure column %s.", structure_col)
    return out, "__composition__", generated


def _apply_feature_log_patterns(
    df: pd.DataFrame,
    target_col: str,
    patterns: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    if not patterns:
        return out, []

    lowered = [p.lower() for p in patterns]
    touched: list[str] = []

    for col in out.columns:
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        low_col = str(col).lower()
        if not any(token in low_col for token in lowered):
            continue

        numeric = pd.to_numeric(out[col], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty or float(finite.min()) <= 0:
            continue

        out[col] = np.log10(numeric)
        touched.append(str(col))

    if touched:
        logger.info("Applied log10 transform to positive condition columns: %s", touched)

    return out, touched


def _drop_redundant_room_temperature_columns(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    dropped: list[str] = []

    temp_like = [
        col
        for col in out.columns
        if any(token in str(col).lower() for token in ["temperature", "temp", "t_k", "t (k)"])
    ]

    for col in temp_like:
        vals = pd.to_numeric(out[col], errors="coerce").dropna()
        if vals.empty:
            continue
        if bool(np.all((vals >= 285.0) & (vals <= 315.0))):
            out = out.drop(columns=[col])
            dropped.append(str(col))

    if dropped:
        logger.info("Dropped room-temperature condition columns: %s", dropped)

    return out, dropped


def _build_composition_featurizer(name: str) -> Any:
    from matminer.featurizers.composition import (
        ElementFraction,
        ElementProperty,
        IonProperty,
        Miedema,
        Stoichiometry,
        ValenceOrbital,
        WenAlloys,
        YangSolidSolution,
    )

    key = str(name).strip().lower()
    if key == "elementproperty":
        return ElementProperty.from_preset("magpie")
    if key == "stoichiometry":
        return Stoichiometry()
    if key == "elementfraction":
        return ElementFraction()
    if key == "valenceorbital":
        return ValenceOrbital()
    if key == "ionproperty":
        return IonProperty(fast=True)
    if key == "miedema":
        return Miedema()
    if key == "wenalloys":
        return WenAlloys()
    if key == "yangsolidsolution":
        return YangSolidSolution()
    raise ValueError(f"Unknown composition featurizer: {name}")


def _ensure_core_composition_featurizers(names: list[str]) -> list[str]:
    core = ["ElementFraction", "ElementProperty", "Stoichiometry", "ValenceOrbital"]
    extras = [str(v) for v in names if str(v) != ""]
    merged = core + extras
    dedup: list[str] = []
    seen: set[str] = set()
    for name in merged:
        key = str(name).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(str(name).strip())
    return dedup


def _build_structure_featurizer(name: str) -> Any:
    from matminer.featurizers.structure import (
        DensityFeatures,
        GlobalSymmetryFeatures,
        JarvisCFID,
        PartialRadialDistributionFunction,
        SiteStatsFingerprint,
    )

    key = str(name).strip().lower()
    if key == "densityfeatures":
        return DensityFeatures()
    if key == "globalsymmetryfeatures":
        return GlobalSymmetryFeatures()
    if key in {"prdf", "partialradialdistributionfunction", "radialdistributionfunction"}:
        try:
            from matminer.featurizers.structure import RadialDistributionFunction

            return RadialDistributionFunction()
        except Exception:
            return PartialRadialDistributionFunction()
    if key == "jarviscfid":
        return JarvisCFID()
    if key == "sitestatsfingerprint":
        preset_candidates = [
            "OPSite",
            "CoordinationNumber_ward-prb-2017",
            "LocalPropertyDifference_ward-prb-2017",
            "CrystalNNFingerprint_ops",
        ]
        for preset in preset_candidates:
            try:
                return SiteStatsFingerprint.from_preset(preset)
            except Exception:
                continue

        # Last-resort fallback for matminer versions with differing preset registries.
        try:
            from matminer.featurizers.site import OPSiteFingerprint

            return SiteStatsFingerprint(site_featurizer=OPSiteFingerprint())
        except Exception:
            pass

        try:
            from matminer.featurizers.site import CrystalNNFingerprint

            return SiteStatsFingerprint(site_featurizer=CrystalNNFingerprint.from_preset("ops"))
        except Exception as exc:
            raise RuntimeError(
                "SiteStatsFingerprint preset resolution failed across known presets and site-featurizer fallbacks."
            ) from exc
    if key == "chemicalordering":
        from matminer.featurizers.structure import ChemicalOrdering

        return ChemicalOrdering()
    if key == "structuralheterogeneity":
        from matminer.featurizers.structure import StructuralHeterogeneity

        return StructuralHeterogeneity()
    raise ValueError(f"Unknown structure featurizer: {name}")


def _apply_selected_matminer_featurizers(
    df: pd.DataFrame,
    logger: logging.Logger,
    composition_col: str | None,
    structure_col: str | None,
    composition_featurizers: list[str],
    structure_featurizers: list[str],
    use_alloy_featurizer_precheck: bool = True,
    composition_chunk_size: int = 1000,
    structure_chunk_size: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    before = set(out.columns)
    info: dict[str, Any] = {
        "applied": False,
        "composition_column": composition_col,
        "structure_column": structure_col,
        "composition_featurizers": [],
        "structure_featurizers": [],
        "errors": [],
        "row_failures": [],
        "skipped_featurizers": [],
    }

    density_key = "densityfeatures"
    if structure_col and structure_col in out.columns:
        for name in structure_featurizers:
            if str(name).strip().lower() != density_key:
                continue
            try:
                feat = _build_structure_featurizer(name)
                before_cols = set(out.columns)
                work_df, skipped_rows = _precheck_featurizer_rows(out, feat, structure_col, logger, f"structure:{name}")
                for row_idx in skipped_rows:
                    info["row_failures"].append(
                        {
                            "row_index": row_idx,
                            "stage": "structure",
                            "featurizer": str(name),
                            "reason": "precheck_unsupported_row",
                        }
                    )
                if skipped_rows and not len(work_df):
                    info["skipped_featurizers"].append(f"structure:{name}:precheck_no_supported_rows")
                    logger.warning("Skipping structure featurizer %s because no rows passed precheck.", name)
                    continue

                featurized_subset = _featurize_dataframe_with_progress(
                    work_df,
                    feat,
                    structure_col,
                    logger,
                    f"structure:{name}",
                    chunk_size=structure_chunk_size,
                    silence_native_stderr=True,
                )
                out = _merge_featurized_subset(out, featurized_subset)
                generated_cols = sorted(set(out.columns) - before_cols)

                info["structure_featurizers"].append(name)
                info["applied"] = True

                if generated_cols:
                    fail_mask = out.loc[work_df.index, generated_cols].isna().all(axis=1)
                    if bool(fail_mask.any()):
                        for row_idx in fail_mask.index[fail_mask].tolist():
                            info["row_failures"].append(
                                {
                                    "row_index": row_idx,
                                    "stage": "structure",
                                    "featurizer": str(name),
                                    "reason": "all_generated_features_nan",
                                }
                            )
            except Exception as exc:
                info["errors"].append(f"structure:{name}:{exc}")
                logger.warning("Structure featurizer %s failed: %s", name, exc)

    if composition_col and composition_col in out.columns:
        for name in composition_featurizers:
            key = str(name).strip().lower()
            try:
                feat = _build_composition_featurizer(name)

                if use_alloy_featurizer_precheck and key in {"miedema", "wenalloys", "yangsolidsolution"} and hasattr(feat, "precheck_dataframe"):
                    try:
                        precheck = feat.precheck_dataframe(out, composition_col)
                        if isinstance(precheck, pd.Series):
                            if not bool(precheck.fillna(False).any()):
                                info["skipped_featurizers"].append(
                                    f"composition:{name}:precheck_no_supported_rows"
                                )
                                continue
                        elif precheck is False:
                            info["skipped_featurizers"].append(
                                f"composition:{name}:precheck_false"
                            )
                            continue
                    except Exception as exc:
                        info["skipped_featurizers"].append(
                            f"composition:{name}:precheck_failed:{exc}"
                        )
                        continue

                before_cols = set(out.columns)
                out = _featurize_dataframe_with_progress(out, feat, composition_col, logger, f"composition:{name}", chunk_size=composition_chunk_size)
                generated_cols = sorted(set(out.columns) - before_cols)

                info["composition_featurizers"].append(name)
                info["applied"] = True

                if generated_cols:
                    fail_mask = out[generated_cols].isna().all(axis=1)
                    if bool(fail_mask.any()):
                        for row_idx in out.index[fail_mask].tolist():
                            info["row_failures"].append(
                                {
                                    "row_index": row_idx,
                                    "stage": "composition",
                                    "featurizer": str(name),
                                    "reason": "all_generated_features_nan",
                                }
                            )
            except Exception as exc:
                info["errors"].append(f"composition:{name}:{exc}")
                logger.warning("Composition featurizer %s failed: %s", name, exc)

    if structure_col and structure_col in out.columns:
        for name in structure_featurizers:
            if str(name).strip().lower() == density_key:
                continue
            try:
                feat = _build_structure_featurizer(name)
                before_cols = set(out.columns)
                work_df, skipped_rows = _precheck_featurizer_rows(out, feat, structure_col, logger, f"structure:{name}")
                for row_idx in skipped_rows:
                    info["row_failures"].append(
                        {
                            "row_index": row_idx,
                            "stage": "structure",
                            "featurizer": str(name),
                            "reason": "precheck_unsupported_row",
                        }
                    )
                if skipped_rows and not len(work_df):
                    info["skipped_featurizers"].append(f"structure:{name}:precheck_no_supported_rows")
                    logger.warning("Skipping structure featurizer %s because no rows passed precheck.", name)
                    continue

                featurized_subset = _featurize_dataframe_with_progress(
                    work_df,
                    feat,
                    structure_col,
                    logger,
                    f"structure:{name}",
                    chunk_size=structure_chunk_size,
                    silence_native_stderr=True,
                )
                out = _merge_featurized_subset(out, featurized_subset)
                generated_cols = sorted(set(out.columns) - before_cols)

                info["structure_featurizers"].append(name)
                info["applied"] = True

                if generated_cols:
                    fail_mask = out.loc[work_df.index, generated_cols].isna().all(axis=1)
                    if bool(fail_mask.any()):
                        for row_idx in fail_mask.index[fail_mask].tolist():
                            info["row_failures"].append(
                                {
                                    "row_index": row_idx,
                                    "stage": "structure",
                                    "featurizer": str(name),
                                    "reason": "all_generated_features_nan",
                                }
                            )
            except Exception as exc:
                info["errors"].append(f"structure:{name}:{exc}")
                logger.warning("Structure featurizer %s failed: %s", name, exc)

    info["generated_columns"] = sorted(set(out.columns) - before)
    return out, info


def apply_dataset_profile_enrichment(
    df: pd.DataFrame,
    target_col: str,
    profile: dict[str, Any] | None,
    logger: logging.Logger,
    enable_matminer: bool,
    enable_material_enrichment: bool,
    force_formula_core_featurizers: bool = True,
    use_alloy_featurizer_precheck: bool = True,
    composition_chunk_size: int = 1000,
    structure_chunk_size: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if profile is None:
        matminer_info: dict[str, Any] = {}
        material_info: dict[str, Any] = {}
        out, matminer_info = apply_matminer_featurizers(df, logger, enable_matminer, memory=None, composition_chunk_size=composition_chunk_size, structure_chunk_size=structure_chunk_size)
        out, material_info = enrich_material_descriptors(out, target_col, logger, memory=None, enabled=enable_material_enrichment)
        return out, matminer_info, material_info

    out = df.copy()
    profile_info: dict[str, Any] = {"dataset_profile": profile.get("dataset_name", "unknown")}

    out, applied_filters = _apply_profile_row_filters(out, profile.get("row_filters", []) or [], logger)
    profile_info["row_filters_applied"] = applied_filters

    out, flattened_cols = _flatten_dict_columns(out, profile.get("dict_cols_to_flatten", []) or [], logger)
    profile_info["flattened_columns"] = flattened_cols

    post_flatten_drop = profile.get("post_flatten_drop_raw_cols", []) or []
    out, post_flatten_removed = _drop_named_columns(out, post_flatten_drop, target_col, logger, "post-flatten")
    profile_info["post_flatten_dropped_columns"] = post_flatten_removed

    raw_feature_cols = [str(v) for v in profile.get("raw_feature_cols", []) or []]
    raw_selected_removed: list[str] = []
    if raw_feature_cols:
        keep_cols = set(raw_feature_cols)
        keep_cols.add(str(target_col))
        keep_cols.update([str(v) for v in profile.get("categorical_cols", []) or []])
        keep_cols.update([str(v) for v in profile.get("numeric_cols", []) or []])
        keep_cols.update([str(v) for v in profile.get("special_nonfeature_cols", []) or []])
        keep_cols.update([str(v) for v in profile.get("dict_cols_to_flatten", []) or []])
        keep_cols.update([str(v) for v in profile.get("site_element_cols", []) or []])
        if profile.get("formula_col"):
            keep_cols.add(str(profile.get("formula_col")))
        if profile.get("structure_col"):
            keep_cols.add(str(profile.get("structure_col")))
        keep_cols.update([str(v) for v in profile.get("formula_cols", []) or []])
        keep_cols.update([str(v) for v in profile.get("structure_cols", []) or []])
        # Preserve formula-like helper columns for optional composition enrichment,
        # even when they are not direct modeling inputs.
        if enable_material_enrichment or force_formula_core_featurizers:
            for helper in ["formula", "composition", "pretty_formula", "reduced_formula"]:
                if helper in out.columns:
                    keep_cols.add(helper)


        for col in list(out.columns):
            if str(col) not in keep_cols:
                out = out.drop(columns=[col])
                raw_selected_removed.append(str(col))

        if raw_selected_removed:
            logger.info("Applied raw feature selection; dropped %d column(s).", int(len(raw_selected_removed)))
    profile_info["raw_feature_selection_dropped"] = sorted(set(raw_selected_removed))

    site_cols = profile.get("site_element_cols", []) or []
    site_features = profile.get("site_element_features", []) or []
    out, site_desc_cols = _add_element_descriptor_columns(out, site_cols, logger, selected_features=site_features)
    profile_info["site_descriptor_columns"] = site_desc_cols

    structure_col = profile.get("structure_col")
    if not structure_col:
        structure_candidates = profile.get("structure_cols", []) or []
        for col in structure_candidates:
            if str(col) in out.columns:
                structure_col = str(col)
                break
    parsed_structure_col = None
    generated_structure_cols: list[str] = []
    if structure_col:
        out, parsed_structure_col, generated_structure_cols = _parse_structure_column(out, str(structure_col), logger)
        profile_info["structure_generated_columns"] = generated_structure_cols

    formula_col = profile.get("formula_col")
    if not formula_col:
        formula_candidates = profile.get("formula_cols", []) or []
        for col in formula_candidates:
            if str(col) in out.columns:
                formula_col = str(col)
                break

    if not formula_col and force_formula_core_featurizers:
        for helper in ["formula", "composition", "pretty_formula", "reduced_formula"]:
            if helper in out.columns and helper != target_col:
                formula_col = helper
                break

    if not formula_col and force_formula_core_featurizers:
        for col in out.columns:
            if col == target_col or out[col].dtype != "object":
                continue
            try:
                if _detect_formula_like_column(out[col]):
                    formula_col = str(col)
                    break
            except Exception:
                continue

    parsed_comp_col = None
    generated_formula_cols: list[str] = []
    parse_formula_requested = bool(profile.get("parse_formula", False)) or bool(force_formula_core_featurizers and formula_col)
    if parse_formula_requested and formula_col:
        out, parsed_comp_col, generated_formula_cols = _parse_formula_column(out, str(formula_col), logger)
        profile_info["formula_generated_columns"] = generated_formula_cols
        if parsed_comp_col is not None and parsed_comp_col in out.columns:
            parse_fail_mask = out[str(formula_col)].notna() & out[parsed_comp_col].isna()
            profile_info["formula_parse_failed_rows"] = out.index[parse_fail_mask].tolist()
    elif parse_formula_requested and not formula_col:
        logger.warning("Formula-core featurization was requested but no formula-like column was found.")

    comp_col_for_matminer = parsed_comp_col
    if comp_col_for_matminer is None and parsed_structure_col is not None:
        out, comp_col_for_matminer, generated_from_structure = _derive_composition_from_structure(out, parsed_structure_col, logger)
        profile_info["structure_composition_generated_columns"] = generated_from_structure

    structure_col_for_matminer = parsed_structure_col

    comp_feats = [str(v) for v in (profile.get("composition_featurizers", []) or [])]
    struct_feats = [str(v) for v in (profile.get("structure_featurizers", []) or [])]
    use_comp = bool(profile.get("use_composition_featurizers", True))
    use_struct = bool(profile.get("use_structure_featurizers", False))

    if force_formula_core_featurizers and comp_col_for_matminer is not None:
        use_comp = True

    if parse_formula_requested and use_comp and comp_col_for_matminer is None:
        logger.warning("Composition featurization requested but parsed composition data was not created.")

    if use_comp and comp_col_for_matminer is not None:
        if not comp_feats:
            comp_feats = ["ElementFraction", "ElementProperty", "Stoichiometry", "ValenceOrbital"]
            logger.info("No composition featurizers were specified; using default set: %s", comp_feats)
        elif force_formula_core_featurizers:
            comp_feats = _ensure_core_composition_featurizers(comp_feats)

    if use_struct and structure_col_for_matminer is not None and not struct_feats:
        struct_feats = ["DensityFeatures", "GlobalSymmetryFeatures", "SiteStatsFingerprint"]
        logger.info("No structure featurizers were specified; using default set: %s", struct_feats)


    if enable_matminer and (use_comp or use_struct):
        out, matminer_info = _apply_selected_matminer_featurizers(
            out,
            logger,
            composition_col=comp_col_for_matminer if use_comp else None,
            structure_col=structure_col_for_matminer if use_struct else None,
            composition_featurizers=comp_feats if use_comp else [],
            structure_featurizers=struct_feats if use_struct else [],
            use_alloy_featurizer_precheck=use_alloy_featurizer_precheck,
            composition_chunk_size=composition_chunk_size,
            structure_chunk_size=structure_chunk_size,
        )
    else:
        matminer_info = {"applied": False, "generated_columns": [], "row_failures": [], "errors": [], "skipped_featurizers": []}

    out, log_touched = _apply_feature_log_patterns(
        out,
        target_col,
        profile.get("feature_log_patterns", []) or [],
        logger,
    )
    profile_info["log_transformed_columns"] = log_touched

    out, dropped_pattern_cols = _apply_drop_col_patterns(
        out,
        target_col,
        profile.get("drop_col_patterns", []) or [],
        logger,
    )
    profile_info["dropped_by_pattern"] = dropped_pattern_cols

    if bool(profile.get("room_temperature_only", False)):
        out, room_temp_dropped = _drop_redundant_room_temperature_columns(out, logger)
        profile_info["room_temperature_dropped_columns"] = room_temp_dropped

    conditional_removed: list[str] = []
    for item in profile.get("conditional_drop_cols", []) or []:
        text = str(item).strip().lower()
        if "if_constant" in text:
            for col in list(out.columns):
                if col == target_col:
                    continue
                vals = out[col].dropna()
                if len(vals) > 0 and int(vals.astype("string").nunique(dropna=True)) <= 1:
                    out = out.drop(columns=[col])
                    conditional_removed.append(str(col))
    profile_info["conditional_drop_removed"] = sorted(set(conditional_removed))

    post_feats_drop = profile.get("post_featurization_drop_raw_cols", []) or []
    out, post_feat_removed = _drop_named_columns(out, post_feats_drop, target_col, logger, "post-featurization")
    profile_info["post_featurization_dropped_columns"] = post_feat_removed

    if enable_material_enrichment:
        out, material_info = enrich_material_descriptors(out, target_col, logger, memory=None, enabled=True)
    else:
        material_info = {"enabled": False, "applied": False, "formula_columns": [], "generated_columns": []}

    auto_drop_raw_formula_cols: list[str] = []
    for raw_col in [formula_col]:
        if raw_col is None:
            continue
        col_name = str(raw_col)
        if col_name == target_col:
            continue
        if col_name in out.columns and col_name not in auto_drop_raw_formula_cols:
            auto_drop_raw_formula_cols.append(col_name)
    if auto_drop_raw_formula_cols:
        out = out.drop(columns=auto_drop_raw_formula_cols, errors="ignore")
    profile_info["auto_dropped_raw_formula_columns"] = auto_drop_raw_formula_cols

    drop_cols = set([str(c) for c in profile.get("resolved_drop_cols", []) or []])
    for col in list(drop_cols):
        if col == target_col:
            drop_cols.remove(col)

    removed_by_profile: list[str] = []
    for col in sorted(drop_cols):
        if col in out.columns:
            out = out.drop(columns=[col])
            removed_by_profile.append(col)

    profile_info["dropped_by_profile"] = removed_by_profile

    return out, {**matminer_info, **profile_info}, material_info













def _apply_profile_row_filters(df: pd.DataFrame, row_filters: list[str], logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    applied: list[str] = []
    for expr in row_filters:
        text = str(expr).strip()
        if text == "":
            continue
        try:
            parts = text.split("==")
            if len(parts) == 2:
                col = parts[0].strip()
                rhs = parts[1].strip().replace('"', "")
                rhs = rhs.replace("'", "")
                if col in out.columns:
                    if rhs.lower() in {"false", "0", "no"}:
                        out = out.loc[~out[col].astype("string").str.lower().isin(["true", "1", "yes"])].copy()
                    elif rhs.lower() in {"true", "1", "yes"}:
                        out = out.loc[out[col].astype("string").str.lower().isin(["true", "1", "yes"])].copy()
                    else:
                        out = out.loc[out[col].astype("string") == rhs].copy()
                    applied.append(text)
                    continue
            out = out.query(text).copy()
            applied.append(text)
        except Exception as exc:
            logger.warning("Could not apply profile row filter '%s': %s", text, exc)
    if applied:
        logger.info("Applied profile row filters: %s", applied)
    return out, applied


def _apply_drop_col_patterns(df: pd.DataFrame, target_col: str, patterns: list[str], logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    dropped: list[str] = []
    for pat in patterns:
        try:
            rgx = re.compile(str(pat))
        except Exception as exc:
            logger.warning("Invalid drop pattern '%s': %s", pat, exc)
            continue
        for col in list(out.columns):
            if col == target_col:
                continue
            if rgx.match(str(col)) and col in out.columns:
                out = out.drop(columns=[col])
                dropped.append(str(col))
    dropped = sorted(set(dropped))
    if dropped:
        logger.info("Dropped columns by regex patterns: %s", dropped)
    return out, dropped


def _drop_named_columns(df: pd.DataFrame, cols: list[str], target_col: str, logger: logging.Logger, label: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    removed: list[str] = []
    for col in cols:
        c = str(col)
        if c == target_col:
            continue
        if c in out.columns:
            out = out.drop(columns=[c])
            removed.append(c)
    if removed:
        logger.info("Dropped %s columns: %s", label, removed)
    return out, removed





































