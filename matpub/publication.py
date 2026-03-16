from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .config import RunConfig
from .data import (
    apply_matminer_featurizers,
    build_preprocessor,
    enrich_material_descriptors,
    prepare_features,
    split_dataset,
    split_dataset_with_groups,
)
from .models import (
    evaluate_classification_metrics,
    evaluate_regression_metrics,
    get_model_definitions,
    model_results_to_table,
    train_models,
)
from .utils import ensure_dir


def _safe_reduce_matrix(matrix: Any, random_state: int, max_dim: int = 30) -> np.ndarray:
    if sparse.issparse(matrix):
        if matrix.shape[1] <= max_dim:
            return matrix.toarray()
        n_comp = min(max_dim, max(2, matrix.shape[1] - 1))
        return TruncatedSVD(n_components=n_comp, random_state=random_state).fit_transform(matrix)

    arr = np.asarray(matrix)
    if arr.shape[1] <= max_dim:
        return arr
    return PCA(n_components=max_dim, random_state=random_state).fit_transform(arr)

def _unwrap_fitted_pipeline(model: Any) -> Pipeline:
    if hasattr(model, "regressor_"):
        return model.regressor_
    if hasattr(model, "regressor"):
        return model.regressor
    return model



def export_publication_tables(
    output_dir: Path,
    model_results_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    uncertainty_summary_df: pd.DataFrame,
) -> None:
    tables_dir = ensure_dir(output_dir / "publication_tables")

    model_results_df.to_csv(tables_dir / "table_model_results.csv", index=False)
    ranking_df.to_csv(tables_dir / "table_ranking.csv", index=False)
    stats_df.to_csv(tables_dir / "table_statistics.csv", index=False)
    uncertainty_summary_df.to_csv(tables_dir / "table_uncertainty.csv", index=False)

    try:
        model_results_df.to_latex(tables_dir / "table_model_results.tex", index=False)
        ranking_df.to_latex(tables_dir / "table_ranking.tex", index=False)
        stats_df.to_latex(tables_dir / "table_statistics.tex", index=False)
        uncertainty_summary_df.to_latex(tables_dir / "table_uncertainty.tex", index=False)
    except Exception:
        pass


def _dataset_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"

    safe = df.copy()
    for col in safe.columns:
        series = safe[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            safe[col] = series.map(lambda v: "<NA>" if pd.isna(v) else str(v))

    try:
        hashed = pd.util.hash_pandas_object(safe.fillna("<NA>"), index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()
    except Exception:
        payload = safe.astype("string").fillna("<NA>").to_csv(index=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import importlib.metadata as importlib_metadata
    except Exception:
        import importlib_metadata  # type: ignore

    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except Exception:
            versions[name] = "not_installed"
    return versions


def write_reproducibility_manifest(
    cfg: RunConfig,
    raw_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_hash_raw": _dataset_hash(raw_df),
        "dataset_hash_enriched": _dataset_hash(enriched_df),
        "n_rows_raw": int(len(raw_df)),
        "n_rows_enriched": int(len(enriched_df)),
        "n_cols_raw": int(raw_df.shape[1]),
        "n_cols_enriched": int(enriched_df.shape[1]),
        "config": cfg.__dict__,
        "package_versions": _package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "matplotlib",
                "seaborn",
                "shap",
                "matminer",
                "pymatgen",
                "ase",
                "xgboost",
                "catboost",
            ]
        ),
    }

    out_path = output_dir / "reproducibility_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out_path


def append_experiment_registry(
    cfg: RunConfig,
    output_dir: Path,
    best_model: str,
    model_results_df: pd.DataFrame,
) -> Path:
    registry_path = Path(cfg.output_root) / cfg.experiment_registry_filename

    row = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": cfg.dataset,
        "target": cfg.target,
        "task": cfg.task,
        "run_mode": cfg.run_mode,
        "best_model": best_model,
        "output_dir": str(output_dir),
    }

    if cfg.task == "regression" and "test_rmse" in model_results_df.columns:
        row["best_test_rmse"] = float(model_results_df["test_rmse"].min())
        row["best_test_r2"] = float(model_results_df["test_r2"].max()) if "test_r2" in model_results_df.columns else float("nan")
    elif cfg.task == "classification" and "test_f1_weighted" in model_results_df.columns:
        row["best_test_f1_weighted"] = float(model_results_df["test_f1_weighted"].max())

    new_row = pd.DataFrame([row])
    if registry_path.exists():
        old = pd.read_csv(registry_path)
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row

    out.to_csv(registry_path, index=False)
    return registry_path


def _prepare_external_xy(df: pd.DataFrame, target: str, task: str, pipeline: Any) -> tuple[pd.DataFrame, pd.Series] | None:
    if target not in df.columns:
        return None

    pipe = _unwrap_fitted_pipeline(pipeline)
    pre = pipe.named_steps["preprocessor"]
    expected_cols = list(getattr(pre, "feature_names_in_", []))
    if not expected_cols:
        return None

    X = df.drop(columns=[target]).copy()
    y = pd.to_numeric(df[target], errors="coerce") if task == "regression" else df[target].astype("string")
    y = y.replace("nan", pd.NA)

    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[expected_cols]

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for name, _, cols in pre.transformers_:
        if name == "num":
            numeric_cols.extend([str(c) for c in cols])
        if name == "cat":
            categorical_cols.extend([str(c) for c in cols])

    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("string")

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    if len(X) == 0:
        return None

    return X, y


def run_external_validation(
    cfg: RunConfig,
    task: str,
    target: str,
    current_dataset: str,
    best_pipeline: Pipeline,
    load_dataset: Any,
    out_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset_name in cfg.external_datasets:
        if dataset_name == current_dataset:
            continue

        try:
            raw = load_dataset(dataset_name)
            df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        except Exception as exc:
            rows.append({"dataset": dataset_name, "status": f"load_failed: {exc}"})
            continue

        parsed = _prepare_external_xy(df, target, task, best_pipeline)
        if parsed is None:
            rows.append({"dataset": dataset_name, "status": "target_or_columns_missing"})
            continue

        X_ext, y_ext = parsed

        try:
            y_pred = best_pipeline.predict(X_ext)
            if task == "regression":
                metrics = evaluate_regression_metrics(y_ext, np.asarray(y_pred))
            else:
                y_proba = best_pipeline.predict_proba(X_ext) if hasattr(best_pipeline, "predict_proba") else None
                metrics = evaluate_classification_metrics(y_ext, np.asarray(y_pred), np.asarray(y_proba) if y_proba is not None else None)

            row = {"dataset": dataset_name, "status": "ok", "n_samples": int(len(X_ext))}
            row.update({f"metric_{k}": float(v) for k, v in metrics.items()})
            rows.append(row)
        except Exception as exc:
            rows.append({"dataset": dataset_name, "status": f"predict_failed: {exc}", "n_samples": int(len(X_ext))})

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "external_validation_summary.csv", index=False)

    if not frame.empty and "status" in frame.columns:
        logger.info("External validation finished for %d dataset(s).", int((frame["status"] == "ok").sum()))
    return frame


def run_ablation_study(
    cfg: RunConfig,
    task: str,
    dataset_name: str,
    target: str,
    raw_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    settings = [
        ("full", True, True),
        ("no_matminer", False, True),
        ("no_material_enrichment", True, False),
        ("minimal", False, False),
    ]

    rows: list[dict[str, Any]] = []
    for name, use_mm, use_me in settings:
        try:
            frame, _ = apply_matminer_featurizers(raw_df, logger, use_mm, memory=None)
            frame, _ = enrich_material_descriptors(frame, target, logger, memory=None, enabled=use_me)
            X, y, num_cols, cat_cols, _ = prepare_features(frame, target, task, cfg)
            pre = build_preprocessor(num_cols, cat_cols)
            X_tr, _, X_te, y_tr, _, y_te = split_dataset(
                X,
                y,
                task,
                cfg.test_size,
                cfg.calibration_size,
                cfg.random_state,
            )

            estimator = Ridge(alpha=1.0) if task == "regression" else LogisticRegression(max_iter=2000)
            pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)

            if task == "regression":
                metrics = evaluate_regression_metrics(y_te, np.asarray(pred))
            else:
                proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
                metrics = evaluate_classification_metrics(y_te, np.asarray(pred), np.asarray(proba) if proba is not None else None)

            row = {
                "ablation": name,
                "use_matminer": int(use_mm),
                "use_material_enrichment": int(use_me),
                "n_features": int(X.shape[1]),
            }
            row.update({f"metric_{k}": float(v) for k, v in metrics.items()})
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "ablation": name,
                    "use_matminer": int(use_mm),
                    "use_material_enrichment": int(use_me),
                    "status": f"failed: {exc}",
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_study.csv", index=False)

    if not frame.empty and task == "regression" and "metric_rmse" in frame.columns:
        plt.figure(figsize=(9, 5))
        sns.barplot(data=frame, x="ablation", y="metric_rmse", color="tab:orange")
        plt.title("Ablation Study (RMSE)")
        plt.ylabel("RMSE")
        plt.xlabel("Setting")
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_study_rmse.png", dpi=300, bbox_inches="tight")
        plt.close()

    return frame


def run_robustness_tests(
    cfg: RunConfig,
    task: str,
    best_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)

    def _score(tag: str, X_eval: pd.DataFrame) -> dict[str, Any]:
        y_pred = best_pipeline.predict(X_eval)
        if task == "regression":
            m = evaluate_regression_metrics(y_test, np.asarray(y_pred))
        else:
            y_proba = best_pipeline.predict_proba(X_eval) if hasattr(best_pipeline, "predict_proba") else None
            m = evaluate_classification_metrics(y_test, np.asarray(y_pred), np.asarray(y_proba) if y_proba is not None else None)
        row = {"scenario": tag}
        row.update({f"metric_{k}": float(v) for k, v in m.items()})
        return row

    rows: list[dict[str, Any]] = []
    rows.append(_score("baseline", X_test))

    num_cols = [c for c in X_test.columns if pd.api.types.is_numeric_dtype(X_test[c])]

    for scale in [0.5, 1.0, 2.0]:
        level = cfg.robustness_noise_std_fraction * scale
        X_noisy = X_test.copy()
        for col in num_cols:
            std = float(np.nanstd(pd.to_numeric(X_train[col], errors="coerce")))
            if std <= 0:
                continue
            X_noisy[col] = pd.to_numeric(X_noisy[col], errors="coerce") + rng.normal(0.0, std * level, len(X_noisy))
        rows.append(_score(f"noise_{level:.3f}", X_noisy))

    for miss_level in cfg.robustness_missingness_levels:
        X_miss = X_test.copy()
        mask = rng.random(X_miss.shape) < float(miss_level)
        X_miss = X_miss.mask(mask)
        rows.append(_score(f"missingness_{miss_level:.3f}", X_miss))

    pipe = _unwrap_fitted_pipeline(best_pipeline)
    pre = pipe.named_steps["preprocessor"]
    train_repr = _safe_reduce_matrix(pre.transform(X_train), cfg.random_state, max_dim=30)
    test_repr = _safe_reduce_matrix(pre.transform(X_test), cfg.random_state, max_dim=30)

    kmeans = KMeans(n_clusters=max(2, cfg.ood_cluster_count), random_state=cfg.random_state, n_init=10)
    kmeans.fit(train_repr)

    train_dist = np.min(np.linalg.norm(train_repr[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2), axis=1)
    test_dist = np.min(np.linalg.norm(test_repr[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2), axis=1)

    thr = float(np.quantile(train_dist, 0.90))
    in_idx = np.where(test_dist <= thr)[0]
    out_idx = np.where(test_dist > thr)[0]

    y_pred_full = best_pipeline.predict(X_test)
    if task == "regression":
        if len(in_idx) > 0:
            m_in = evaluate_regression_metrics(y_test.iloc[in_idx], np.asarray(y_pred_full)[in_idx])
            rows.append({"scenario": "ood_in_domain", **{f"metric_{k}": float(v) for k, v in m_in.items()}, "n_samples": int(len(in_idx))})
        if len(out_idx) > 0:
            m_out = evaluate_regression_metrics(y_test.iloc[out_idx], np.asarray(y_pred_full)[out_idx])
            rows.append({"scenario": "ood_out_domain", **{f"metric_{k}": float(v) for k, v in m_out.items()}, "n_samples": int(len(out_idx))})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "robustness_tests.csv", index=False)
    return frame


def run_physics_sanity_checks(
    cfg: RunConfig,
    target: str,
    y_train: pd.Series,
    y_test_pred: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    pred = np.asarray(y_test_pred, dtype=float)
    ytr = np.asarray(pd.to_numeric(y_train, errors="coerce"), dtype=float)
    ytr = ytr[np.isfinite(ytr)]

    positive_expected = bool(cfg.expected_target_positive or (len(ytr) > 0 and np.nanmin(ytr) >= 0))

    rows: list[dict[str, Any]] = []
    if positive_expected:
        neg_count = int(np.sum(pred < 0))
        rows.append(
            {
                "check": "non_negative_target",
                "expected": "prediction >= 0",
                "violations": neg_count,
                "violation_rate": float(neg_count / max(1, len(pred))),
            }
        )

    if len(ytr) > 0:
        lo = float(np.nanmin(ytr))
        hi = float(np.nanmax(ytr))
        span = max(1e-12, hi - lo)
        lower_bound = lo - 0.2 * span
        upper_bound = hi + 0.2 * span
        out_count = int(np.sum((pred < lower_bound) | (pred > upper_bound)))
        rows.append(
            {
                "check": "train_range_guard",
                "expected": f"[{lower_bound:.6g}, {upper_bound:.6g}]",
                "violations": out_count,
                "violation_rate": float(out_count / max(1, len(pred))),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "physics_sanity_checks.csv", index=False)
    return frame




def _mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    half = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, mean - half, mean + half


def run_repeated_group_benchmark(
    cfg: RunConfig,
    task: str,
    preprocessor: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups_full: pd.Series | None,
    sample_weight_full: pd.Series | None,
    output_dir: Path,
    logger: logging.Logger,
    definitions: list[Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.enable_repeated_runs or int(cfg.repeated_runs) <= 1:
        return pd.DataFrame(), pd.DataFrame()

    if definitions is None:
        definitions = get_model_definitions(cfg, task, logger)

    run_tables: list[pd.DataFrame] = []

    for rep in range(int(cfg.repeated_runs)):
        seed = int(cfg.random_state + rep * 37)
        cfg_seed = RunConfig(**cfg.__dict__)
        cfg_seed.random_state = seed

        X_train, _, X_test, y_train, _, y_test, g_train, _, g_test = split_dataset_with_groups(
            X,
            y,
            groups_full if cfg.use_group_aware_split else None,
            task,
            cfg.test_size,
            cfg.calibration_size,
            seed,
        )

        groups_for_nested = pd.concat([g_train, g_test], axis=0) if (g_train is not None and g_test is not None) else None
        sample_weight_train = sample_weight_full.reindex(X_train.index) if sample_weight_full is not None else None

        try:
            results = train_models(
                cfg_seed,
                task,
                preprocessor,
                X_train,
                y_train,
                X_test,
                y_test,
                logger,
                groups_train=g_train if cfg.use_group_aware_cv else None,
                groups_all=groups_for_nested if cfg.use_group_aware_cv else None,
                definitions=definitions,
                sample_weight_train=sample_weight_train,
            )
            table = model_results_to_table(results, task)
            table["seed"] = seed
            run_tables.append(table)
            logger.info("Repeated benchmark run %d/%d finished.", rep + 1, int(cfg.repeated_runs))
        except Exception as exc:
            logger.warning("Repeated benchmark run %d failed: %s", rep + 1, exc)

    if not run_tables:
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.concat(run_tables, ignore_index=True)
    raw_df.to_csv(output_dir / "repeated_runs_raw.csv", index=False)

    metric_cols = [
        c
        for c in ["cv_primary", "test_r2", "test_rmse", "test_mae", "test_accuracy", "test_f1_weighted", "test_log_loss", "total_seconds"]
        if c in raw_df.columns
    ]

    summary_rows: list[dict[str, Any]] = []
    for model_name, grp in raw_df.groupby("model"):
        row: dict[str, Any] = {"model": model_name, "n_runs": int(len(grp))}
        for m in metric_cols:
            mean, std, ci_low, ci_high = _mean_ci(np.asarray(grp[m], dtype=float))
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95_low"] = ci_low
            row[f"{m}_ci95_high"] = ci_high
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("model")
    summary_df.to_csv(output_dir / "repeated_runs_summary.csv", index=False)

    if task == "regression" and {"test_rmse_mean", "test_rmse_ci95_low", "test_rmse_ci95_high"}.issubset(summary_df.columns):
        err = np.vstack(
            [
                summary_df["test_rmse_mean"] - summary_df["test_rmse_ci95_low"],
                summary_df["test_rmse_ci95_high"] - summary_df["test_rmse_mean"],
            ]
        )
        plt.figure(figsize=(10, 6))
        plt.errorbar(summary_df["model"], summary_df["test_rmse_mean"], yerr=err, fmt="o", capsize=4)
        plt.xticks(rotation=35)
        plt.ylabel("RMSE")
        plt.title("Repeated Group-aware Runs: RMSE Mean ± 95% CI")
        plt.tight_layout()
        plt.savefig(output_dir / "repeated_runs_rmse_ci95.png", dpi=300, bbox_inches="tight")
        plt.close()
    elif task == "classification" and {"test_f1_weighted_mean", "test_f1_weighted_ci95_low", "test_f1_weighted_ci95_high"}.issubset(summary_df.columns):
        err = np.vstack(
            [
                summary_df["test_f1_weighted_mean"] - summary_df["test_f1_weighted_ci95_low"],
                summary_df["test_f1_weighted_ci95_high"] - summary_df["test_f1_weighted_mean"],
            ]
        )
        plt.figure(figsize=(10, 6))
        plt.errorbar(summary_df["model"], summary_df["test_f1_weighted_mean"], yerr=err, fmt="o", capsize=4)
        plt.xticks(rotation=35)
        plt.ylabel("F1 weighted")
        plt.title("Repeated Group-aware Runs: F1 Mean ± 95% CI")
        plt.tight_layout()
        plt.savefig(output_dir / "repeated_runs_f1_ci95.png", dpi=300, bbox_inches="tight")
        plt.close()

    return raw_df, summary_df


def build_cross_dataset_generalization_table(
    task: str,
    best_model_label: str,
    model_metrics_df: pd.DataFrame,
    external_df: pd.DataFrame,
    output_dir: Path,
    best_pred_doa_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if model_metrics_df.empty:
        return pd.DataFrame()

    in_row = model_metrics_df.loc[model_metrics_df["model"] == best_model_label]
    if in_row.empty:
        in_row = model_metrics_df.iloc[[0]]
    in_row = in_row.iloc[0]

    rows: list[dict[str, Any]] = []

    if task == "regression":
        in_primary = float(in_row.get("test_rmse", np.nan))
        rows.append({"dataset": "in_domain", "domain_type": "in_domain", "primary_metric": in_primary, "status": "ok"})

        has_ood = False
        if not external_df.empty and "status" in external_df.columns:
            ok_ext = external_df.loc[external_df["status"].astype("string") == "ok"].copy()
            has_ood = not ok_ext.empty
            for r in ok_ext.itertuples():
                ext_primary = float(getattr(r, "metric_rmse", np.nan))
                rows.append(
                    {
                        "dataset": str(getattr(r, "dataset", "unknown")),
                        "domain_type": "out_of_domain",
                        "primary_metric": ext_primary,
                        "generalization_gap_vs_in_domain": float(ext_primary - in_primary),
                        "status": "ok",
                    }
                )
        if not has_ood:
            rows.append(
                {
                    "dataset": "out_of_domain (not provided)",
                    "domain_type": "out_of_domain",
                    "primary_metric": np.nan,
                    "generalization_gap_vs_in_domain": np.nan,
                    "status": "missing_external_validation",
                }
            )

        if best_pred_doa_df is not None and not best_pred_doa_df.empty:
            flag_col = None
            for candidate in [
                "doa_consensus_in_domain",
                "doa_knn_in_domain",
                "doa_mahalanobis_in_domain",
                "doa_leverage_in_domain",
            ]:
                if candidate in best_pred_doa_df.columns:
                    flag_col = candidate
                    break

            if flag_col is not None and {"y_true", "y_pred"}.issubset(best_pred_doa_df.columns):
                y_true = pd.to_numeric(best_pred_doa_df["y_true"], errors="coerce")
                y_pred = pd.to_numeric(best_pred_doa_df["y_pred"], errors="coerce")
                flags = pd.to_numeric(best_pred_doa_df[flag_col], errors="coerce").fillna(0).astype(int)
                valid = np.isfinite(y_true.to_numpy(dtype=float)) & np.isfinite(y_pred.to_numpy(dtype=float))

                in_ad = valid & (flags.to_numpy(dtype=int) == 1)
                out_ad = valid & (flags.to_numpy(dtype=int) == 0)

                if int(np.sum(in_ad)) >= 3:
                    rmse_in_ad = float(np.sqrt(np.mean((y_true.to_numpy(dtype=float)[in_ad] - y_pred.to_numpy(dtype=float)[in_ad]) ** 2)))
                    rows.append({"dataset": "in_AD", "domain_type": "applicability_domain", "primary_metric": rmse_in_ad, "status": "ok"})
                else:
                    rows.append({"dataset": "in_AD", "domain_type": "applicability_domain", "primary_metric": np.nan, "status": "insufficient_samples"})

                if int(np.sum(out_ad)) >= 3:
                    rmse_out_ad = float(np.sqrt(np.mean((y_true.to_numpy(dtype=float)[out_ad] - y_pred.to_numpy(dtype=float)[out_ad]) ** 2)))
                    rows.append({"dataset": "out_of_AD", "domain_type": "applicability_domain", "primary_metric": rmse_out_ad, "status": "ok"})
                else:
                    rows.append({"dataset": "out_of_AD", "domain_type": "applicability_domain", "primary_metric": np.nan, "status": "insufficient_samples"})
    else:
        in_primary = float(in_row.get("test_f1_weighted", np.nan))
        rows.append({"dataset": "in_domain", "domain_type": "in_domain", "primary_metric": in_primary, "status": "ok"})

        has_ood = False
        if not external_df.empty and "status" in external_df.columns:
            ok_ext = external_df.loc[external_df["status"].astype("string") == "ok"].copy()
            has_ood = not ok_ext.empty
            for r in ok_ext.itertuples():
                ext_primary = float(getattr(r, "metric_f1_weighted", np.nan))
                rows.append(
                    {
                        "dataset": str(getattr(r, "dataset", "unknown")),
                        "domain_type": "out_of_domain",
                        "primary_metric": ext_primary,
                        "generalization_gap_vs_in_domain": float(in_primary - ext_primary),
                        "status": "ok",
                    }
                )
        if not has_ood:
            rows.append(
                {
                    "dataset": "out_of_domain (not provided)",
                    "domain_type": "out_of_domain",
                    "primary_metric": np.nan,
                    "generalization_gap_vs_in_domain": np.nan,
                    "status": "missing_external_validation",
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "cross_dataset_generalization.csv", index=False)

    if not table.empty:
        plot_df = table.copy()
        plot_df["primary_metric_plot"] = pd.to_numeric(plot_df["primary_metric"], errors="coerce").fillna(0.0)
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=plot_df, x="dataset", y="primary_metric_plot", hue="domain_type")
        plt.xticks(rotation=35)
        if task == "regression":
            plt.ylabel("Primary metric (RMSE)")
        else:
            plt.ylabel("Primary metric (F1 weighted)")
        plt.title("In-domain / Out-of-domain / AD Generalization")

        if (table["status"].astype("string") == "missing_external_validation").any():
            ax.text(
                0.99,
                0.02,
                "No external out-of-domain dataset supplied (N/A shown as zero-height placeholder).",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(output_dir / "cross_dataset_generalization.png", dpi=300, bbox_inches="tight")
        plt.close()

    return table

def run_structure_graph_model_benchmark(
    cfg: RunConfig,
    dataset_profile: dict[str, Any] | None,
    task: str,
    target: str,
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_protocol = (
        f"group_aware_split={cfg.use_group_aware_split};"
        f"group_aware_cv={cfg.use_group_aware_cv};"
        f"test_size={cfg.test_size};"
        f"calibration_size={cfg.calibration_size};"
        f"random_state={cfg.random_state}"
    )

    if not cfg.enable_graph_models:
        rows.append({"model": "GraphModel", "status": "disabled_by_config", "split_protocol": split_protocol})
    elif dataset_profile is None or str(dataset_profile.get("input_mode", "")).lower() not in {"structure", "hybrid"}:
        rows.append({"model": "GraphModel", "status": "not_applicable_non_structure_dataset", "split_protocol": split_protocol})
    else:
        structure_col = dataset_profile.get("structure_col")
        if not structure_col or str(structure_col) not in df.columns:
            rows.append({"model": "GraphModel", "status": "structure_column_missing", "split_protocol": split_protocol})
        else:
            try:
                import matgl  # noqa: F401

                rows.append(
                    {
                        "model": "M3GNet/TensorNet",
                        "status": "dependency_available_training_hook_ready",
                        "note": "Structure graph benchmark hook is available; full training loop can be enabled with project-specific MatGL setup.",
                        "task": task,
                        "target": target,
                        "split_protocol": split_protocol,
                    }
                )
            except Exception as exc:
                rows.append({"model": "M3GNet/TensorNet", "status": f"dependency_missing: {exc}", "split_protocol": split_protocol})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
    logger.info("Structure graph benchmark status exported.")
    return frame


def run_subgroup_robustness_breakdown(
    cfg: RunConfig,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    intervals_file: Path,
    groups_test: pd.Series | None,
    output_dir: Path,
) -> pd.DataFrame:
    if not cfg.enable_subgroup_robustness_breakdown:
        return pd.DataFrame()

    if not intervals_file.exists():
        return pd.DataFrame()

    interval_df = pd.read_csv(intervals_file)
    if interval_df.empty:
        return pd.DataFrame()

    y_true = np.asarray(pd.to_numeric(y_test, errors="coerce"), dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)

    covered_col = None
    width_col = None
    for col in interval_df.columns:
        if col.startswith("covered_alpha_") and ("0.05" in col or "0_05" in col):
            covered_col = col
        if col.startswith("width_alpha_") and ("0.05" in col or "0_05" in col):
            width_col = col

    base = pd.DataFrame(index=X_test.index)
    base["y_true"] = y_true
    base["y_pred"] = y_hat
    base["abs_error"] = np.abs(y_true - y_hat)

    if covered_col is not None:
        base["coverage_95"] = pd.to_numeric(interval_df[covered_col], errors="coerce").to_numpy(dtype=float)
    if width_col is not None:
        base["width_95"] = pd.to_numeric(interval_df[width_col], errors="coerce").to_numpy(dtype=float)

    group_maps: dict[str, pd.Series] = {}

    qbins = pd.qcut(base["y_true"], q=min(4, max(2, int(base["y_true"].nunique()))), duplicates="drop")
    group_maps["target_quantile"] = qbins.astype("string")

    if groups_test is not None:
        group_maps["chemistry_family"] = groups_test.reindex(X_test.index).astype("string")

    synth_cols = [
        c
        for c in X_test.columns
        if any(tok in str(c).lower() for tok in ["synth", "process", "route", "anneal"])
    ]
    if synth_cols:
        col = synth_cols[0]
        group_maps["synthesis_regime"] = X_test[col].astype("string")

    rows: list[dict[str, Any]] = []
    for subgroup_type, subgroup_series in group_maps.items():
        tmp = base.copy()
        tmp["subgroup"] = subgroup_series
        for subgroup, part in tmp.groupby("subgroup"):
            if len(part) < 5:
                continue
            row = {
                "subgroup_type": subgroup_type,
                "subgroup": str(subgroup),
                "n_samples": int(len(part)),
                "rmse": float(np.sqrt(np.mean((part["y_true"] - part["y_pred"]) ** 2))),
                "mae": float(np.mean(np.abs(part["y_true"] - part["y_pred"]))),
            }
            if "coverage_95" in part.columns:
                row["coverage_95"] = float(np.nanmean(part["coverage_95"]))
            if "width_95" in part.columns:
                row["avg_width_95"] = float(np.nanmean(part["width_95"]))
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "subgroup_robustness_breakdown.csv", index=False)

    if not frame.empty and "coverage_95" in frame.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=frame, x="subgroup", y="coverage_95", hue="subgroup_type")
        plt.xticks(rotation=35)
        plt.ylabel("Coverage (95% interval)")
        plt.title("Subgroup Coverage Breakdown")
        plt.tight_layout()
        plt.savefig(output_dir / "subgroup_coverage_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()

    if not frame.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=frame, x="subgroup", y="rmse", hue="subgroup_type")
        plt.xticks(rotation=35)
        plt.ylabel("RMSE")
        plt.title("Subgroup RMSE Breakdown")
        plt.tight_layout()
        plt.savefig(output_dir / "subgroup_rmse_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()

    return frame
def generate_reporting_checklist(output_dir: Path, checks: dict[str, bool]) -> Path:
    lines = ["# Reporting Checklist", ""]
    for name, ok in checks.items():
        mark = "[x]" if ok else "[ ]"
        lines.append(f"- {mark} {name}")
    out = output_dir / "reporting_checklist.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out







def run_leave_group_protocols(
    cfg: RunConfig,
    task: str,
    preprocessor: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups_map: dict[str, pd.Series | None],
    definitions: list[Any],
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    if not cfg.enable_leave_group_protocols:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for protocol_name, group_series in groups_map.items():
        if group_series is None:
            rows.append({"protocol": protocol_name, "status": "skipped_no_groups"})
            continue

        grp = group_series.astype("string").reindex(X.index)
        n_groups = int(grp.nunique(dropna=True))
        if n_groups < int(cfg.leave_group_min_groups):
            rows.append({"protocol": protocol_name, "status": f"skipped_insufficient_groups:{n_groups}"})
            continue

        n_splits = max(2, min(int(cfg.cv_folds), n_groups))
        gkf = GroupKFold(n_splits=n_splits)

        for definition in definitions:
            fold_scores: list[float] = []
            fold_sizes: list[int] = []

            for tr_idx, te_idx in gkf.split(X, y, grp):
                X_tr = X.iloc[tr_idx]
                y_tr = y.iloc[tr_idx]
                X_te = X.iloc[te_idx]
                y_te = y.iloc[te_idx]

                pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(definition.estimator))])
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_te)

                if task == "regression":
                    score = float(np.sqrt(np.mean((np.asarray(y_te) - np.asarray(pred)) ** 2)))
                else:
                    score = float(f1_score(np.asarray(y_te), np.asarray(pred), average="weighted", zero_division=0))

                fold_scores.append(score)
                fold_sizes.append(int(len(te_idx)))

            if not fold_scores:
                rows.append({"protocol": protocol_name, "model": definition.label, "status": "failed_no_folds"})
                continue

            arr = np.asarray(fold_scores, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            half = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

            rows.append(
                {
                    "protocol": protocol_name,
                    "model": definition.label,
                    "metric": "rmse" if task == "regression" else "f1_weighted",
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "n_folds": int(len(arr)),
                    "mean_test_fold_size": float(np.mean(fold_sizes)),
                    "status": "ok",
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "leave_group_protocols.csv", index=False)

    ok = frame.loc[frame.get("status", "") == "ok"].copy() if not frame.empty else pd.DataFrame()
    if not ok.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=ok, x="model", y="mean", hue="protocol")
        plt.xticks(rotation=35)
        ylabel = "RMSE (lower is better)" if task == "regression" else "F1 weighted (higher is better)"
        plt.ylabel(ylabel)
        plt.title("Leave-Group-Out Protocol Comparison")
        plt.tight_layout()
        plt.savefig(output_dir / "leave_group_protocols.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("Leave-group protocol evaluation exported.")
    return frame


def scan_feature_leakage(
    cfg: RunConfig,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    if not cfg.enable_leakage_scan:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    rounded_cols: dict[str, pd.Series] = {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            rounded_cols[str(col)] = pd.to_numeric(s, errors="coerce").astype(float).round(int(cfg.near_duplicate_round_decimals)).fillna(-999999.0)
        else:
            rounded_cols[str(col)] = s.astype("string").fillna("<NA>")
    rounded = pd.DataFrame(rounded_cols, index=X.index)

    row_hash = pd.util.hash_pandas_object(rounded, index=False)
    dup_count = int(row_hash.duplicated(keep=False).sum())
    rows.append({"check": "exact_or_near_duplicate_rows", "value": dup_count, "details": "rows sharing rounded feature hash"})

    if len(X_train) > 0 and len(X_test) > 0:
        h_train = pd.util.hash_pandas_object(rounded.loc[X_train.index], index=False)
        h_test = pd.util.hash_pandas_object(rounded.loc[X_test.index], index=False)
        overlap = int(len(set(h_train.tolist()) & set(h_test.tolist())))
        rows.append({"check": "train_test_feature_hash_overlap", "value": overlap, "details": "possible leakage via duplicated samples"})

    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().sum() >= 5:
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            x_num = pd.to_numeric(X[col], errors="coerce")
            mask = x_num.notna() & y_num.notna()
            if int(mask.sum()) < 5:
                continue
            x_vals = x_num[mask].to_numpy(dtype=float)
            y_vals = y_num[mask].to_numpy(dtype=float)
            if float(np.nanstd(x_vals)) < 1e-12 or float(np.nanstd(y_vals)) < 1e-12:
                continue
            corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
            if np.isfinite(corr) and abs(corr) >= float(cfg.leakage_proxy_corr_threshold):
                rows.append(
                    {
                        "check": "target_proxy_correlation",
                        "feature": str(col),
                        "value": corr,
                        "details": f"abs(corr)>={cfg.leakage_proxy_corr_threshold}",
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "leakage_scan.csv", index=False)
    return out


def export_environment_lock(output_dir: Path) -> Path:
    lock_path = output_dir / "environment_lock.txt"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = proc.stdout.strip()
        if text == "":
            text = "pip freeze unavailable"
        lock_path.write_text(text + "\n", encoding="utf-8")
    except Exception as exc:
        lock_path.write_text(f"environment lock export failed: {exc}\n", encoding="utf-8")
    return lock_path


def compute_artifact_hashes(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root, _, files in os.walk(output_dir):
        for name in files:
            p = Path(root) / name
            rel = str(p.relative_to(output_dir))
            try:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                rows.append({"artifact": rel, "sha256": h, "size_bytes": int(p.stat().st_size)})
            except Exception:
                continue
    frame = pd.DataFrame(rows).sort_values("artifact") if rows else pd.DataFrame(columns=["artifact", "sha256", "size_bytes"])
    frame.to_csv(output_dir / "artifact_hashes.csv", index=False)
    return frame


def track_experiment_backend(
    cfg: RunConfig,
    output_dir: Path,
    run_id: str,
    model_results_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not cfg.enable_experiment_tracking or str(cfg.tracking_backend).lower() == "none":
        frame = pd.DataFrame([{"backend": "none", "status": "disabled"}])
        frame.to_csv(output_dir / "experiment_tracking_status.csv", index=False)
        return frame

    backend = str(cfg.tracking_backend).lower()
    status_rows: list[dict[str, Any]] = []

    best_metric = None
    if cfg.task == "regression" and "test_rmse" in model_results_df.columns:
        best_metric = float(model_results_df["test_rmse"].min())
    if cfg.task == "classification" and "test_f1_weighted" in model_results_df.columns:
        best_metric = float(model_results_df["test_f1_weighted"].max())

    if backend in {"auto", "mlflow"}:
        try:
            import mlflow

            tracking_uri = cfg.tracking_uri.strip() if cfg.tracking_uri else str((Path(cfg.output_root) / "mlruns").resolve())
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(cfg.tracking_project)
            with mlflow.start_run(run_name=run_id):
                mlflow.log_param("run_id", run_id)
                mlflow.log_param("dataset", cfg.dataset)
                mlflow.log_param("target", cfg.target)
                mlflow.log_param("task", cfg.task)
                if best_metric is not None:
                    key = "best_test_rmse" if cfg.task == "regression" else "best_test_f1_weighted"
                    mlflow.log_metric(key, best_metric)
                mlflow.log_artifacts(str(output_dir))
            status_rows.append({"backend": "mlflow", "status": "ok", "tracking_uri": tracking_uri})
            backend = "mlflow"
        except Exception as exc:
            status_rows.append({"backend": "mlflow", "status": f"failed:{exc}"})

    if backend in {"auto", "wandb"}:
        try:
            import wandb

            wb = wandb.init(
                project=cfg.tracking_project,
                id=run_id,
                name=run_id,
                resume="never",
                mode="offline",
                dir=str(output_dir),
                config={
                    "dataset": cfg.dataset,
                    "target": cfg.target,
                    "task": cfg.task,
                    "run_mode": cfg.run_mode,
                },
            )
            if best_metric is not None:
                key = "best_test_rmse" if cfg.task == "regression" else "best_test_f1_weighted"
                wandb.log({key: best_metric})
            artifact = wandb.Artifact(name=f"{run_id}_artifacts", type="results")
            artifact.add_dir(str(output_dir))
            wb.log_artifact(artifact)
            wb.finish()
            status_rows.append({"backend": "wandb", "status": "ok", "mode": "offline"})
        except Exception as exc:
            status_rows.append({"backend": "wandb", "status": f"failed:{exc}"})

    frame = pd.DataFrame(status_rows if status_rows else [{"backend": "none", "status": "no_backend_available"}])
    frame.to_csv(output_dir / "experiment_tracking_status.csv", index=False)

    logger.info("Experiment tracking status exported.")
    return frame


def run_structure_graph_model_benchmark_full(
    cfg: RunConfig,
    dataset_profile: dict[str, Any] | None,
    task: str,
    target: str,
    enriched_df: pd.DataFrame,
    y_full: pd.Series,
    train_index: pd.Index,
    test_index: pd.Index,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    split_df = pd.DataFrame(
        {
            "index": list(train_index.astype(str)) + list(test_index.astype(str)),
            "split": ["train"] * len(train_index) + ["test"] * len(test_index),
        }
    )
    split_df.to_csv(output_dir / "graph_split_indices.csv", index=False)

    if not cfg.enable_graph_models:
        rows.append({"model": "M3GNet", "status": "disabled_by_config"})
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
        return frame

    if task != "regression":
        rows.append({"model": "M3GNet", "status": "skipped_non_regression"})
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
        return frame

    if dataset_profile is None or str(dataset_profile.get("input_mode", "")).lower() not in {"structure", "hybrid"}:
        rows.append({"model": "M3GNet", "status": "not_applicable_non_structure_dataset"})
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
        return frame

    structure_candidates = []
    if "__structure__" in enriched_df.columns:
        structure_candidates.append("__structure__")
    if dataset_profile.get("structure_col"):
        structure_candidates.append(str(dataset_profile.get("structure_col")))
    structure_candidates.extend([str(v) for v in dataset_profile.get("structure_cols", []) or []])

    struct_col = None
    for c in structure_candidates:
        if c in enriched_df.columns:
            struct_col = c
            break

    if struct_col is None:
        rows.append({"model": "M3GNet", "status": "structure_column_missing"})
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
        return frame

    try:
        import matgl  # noqa: F401
        import torch  # noqa: F401
        import dgl  # noqa: F401
    except Exception as exc:
        rows.append({"model": "M3GNet", "status": f"dependency_missing:{exc}"})
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
        return frame

    df_graph = enriched_df.loc[y_full.index].copy()
    s = df_graph[struct_col]
    y_num = pd.to_numeric(y_full, errors="coerce")
    valid_mask = s.notna() & y_num.notna()
    df_graph = df_graph.loc[valid_mask]
    y_num = y_num.loc[valid_mask]

    tr_idx = [idx for idx in train_index if idx in df_graph.index]
    te_idx = [idx for idx in test_index if idx in df_graph.index]
    missing_train = int(len(train_index) - len(tr_idx))
    missing_test = int(len(test_index) - len(te_idx))

    if len(tr_idx) < 20 or len(te_idx) < 5:
        rows.append(
            {
                "model": "M3GNet",
                "status": "insufficient_valid_structure_rows",
                "n_train_available": int(len(tr_idx)),
                "n_test_available": int(len(te_idx)),
                "missing_train_from_tabular_split": missing_train,
                "missing_test_from_tabular_split": missing_test,
                "structure_col": struct_col,
                "split_indices_file": "graph_split_indices.csv",
            }
        )
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
        return frame

    try:
        from matgl.ext.pymatgen import Structure2Graph
        from matgl.graph.data import MGLDataLoader, MGLDataset
        from matgl.models import M3GNet
        from matgl.utils.training import ModelLightningModule
        import pytorch_lightning as pl

        structures_train = df_graph.loc[tr_idx, struct_col].tolist()
        structures_test = df_graph.loc[te_idx, struct_col].tolist()
        y_train = y_num.loc[tr_idx].to_numpy(dtype=float)
        y_test = y_num.loc[te_idx].to_numpy(dtype=float)

        elem_types = sorted({str(el) for st in structures_train for el in st.composition.elements})
        converter = Structure2Graph(element_types=elem_types, cutoff=5.0)

        train_dataset = MGLDataset(structures=structures_train, labels={"target": y_train}, converter=converter)
        test_dataset = MGLDataset(structures=structures_test, labels={"target": y_test}, converter=converter)

        train_loader = MGLDataLoader(train_dataset, batch_size=int(cfg.graph_batch_size), shuffle=True)
        test_loader = MGLDataLoader(test_dataset, batch_size=int(cfg.graph_batch_size), shuffle=False)

        model = M3GNet(element_types=elem_types, is_intensive=True)
        lit_model = ModelLightningModule(model=model, include_line_graph=True, loss="mse", lr=1e-3)

        trainer = pl.Trainer(
            max_epochs=int(cfg.graph_model_epochs),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(lit_model, train_loader)

        pred_batches = trainer.predict(lit_model, dataloaders=test_loader)
        preds = np.concatenate([np.asarray(p).reshape(-1) for p in pred_batches])
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        mae = float(np.mean(np.abs(preds - y_test)))
        r2 = float(1.0 - np.sum((preds - y_test) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-12))

        pred_frame = pd.DataFrame(
            {
                "index": [str(i) for i in te_idx],
                "y_true": y_test,
                "y_pred": preds,
            }
        )
        pred_frame.to_csv(output_dir / "structure_graph_predictions.csv", index=False)

        rows.append(
            {
                "model": "M3GNet",
                "status": "ok",
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "missing_train_from_tabular_split": missing_train,
                "missing_test_from_tabular_split": missing_test,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "structure_col": struct_col,
                "split_indices_file": "graph_split_indices.csv",
                "predictions_file": "structure_graph_predictions.csv",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "model": "M3GNet",
                "status": f"training_failed:{exc}",
                "structure_col": struct_col,
                "split_indices_file": "graph_split_indices.csv",
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "structure_graph_benchmark_status.csv", index=False)
    logger.info("Structure graph benchmark (full path) exported.")
    return frame






