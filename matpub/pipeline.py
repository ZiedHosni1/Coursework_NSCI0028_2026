from __future__ import annotations

import logging
import hashlib
import json
import time
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import (
    analyze_classification_model,
    analyze_regression_model,
    plot_global_uncertainty_comparison,
    run_dataset_diagnostics,
    save_plot,
)
from .config import RunConfig, args_to_config, build_parser
from .dataset_profiles import get_dataset_config, resolve_target_specific_drops
from .data import (
    apply_dataset_profile_enrichment,
    build_preprocessor,
    detect_group_column,
    get_matminer_dataset_functions,
    infer_task_type,
    prepare_features,
    preview_dataset,
    split_dataset_with_groups,
)
from .models import (
    bayesian_correlated_ttest_comparison,
    choose_best_model,
    export_tuning_spaces,
    get_model_definitions,
    model_results_to_table,
    permutation_model_comparison,
    rank_models_multi_objective,
    statistical_model_comparison,
    train_models,
)
from .publication import (
    append_experiment_registry,
    build_cross_dataset_generalization_table,
    compute_artifact_hashes,
    export_environment_lock,
    export_publication_tables,
    generate_reporting_checklist,
    run_ablation_study,
    run_external_validation,
    run_leave_group_protocols,
    run_physics_sanity_checks,
    run_repeated_group_benchmark,
    run_robustness_tests,
    run_structure_graph_model_benchmark_full,
    run_subgroup_robustness_breakdown,
    scan_feature_leakage,
    track_experiment_backend,
    write_reproducibility_manifest,
)
from .report import generate_html_report
from .utils import ensure_dir, parse_csv_arg, save_json, sanitize_filename


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("matpub")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def make_output_dir(cfg: RunConfig, dataset: str, target: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{sanitize_filename(dataset)}__{sanitize_filename(target)}__{cfg.run_mode}__{stamp}"
    return ensure_dir((Path(cfg.output_root) / name).resolve())


def prompt_choice(prompt: str, options: list[str], default_idx: int = 0) -> str:
    print(f"\n{prompt}")
    for idx, option in enumerate(options, start=1):
        print(f"[{idx}] {option}")
    while True:
        raw = input(f"Enter number (default {default_idx + 1}): ").strip()
        if raw == "":
            return options[default_idx]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice.")


def maybe_configure_interactively(cfg: RunConfig, df: pd.DataFrame) -> None:
    drop_raw = input("Columns to drop (comma separated, blank=none): ").strip()
    if drop_raw:
        cfg.drop_columns = sorted(set(cfg.drop_columns + parse_csv_arg(drop_raw)))

    keep_raw = input("Columns to force-keep as categorical (comma separated, blank=none): ").strip()
    if keep_raw:
        cfg.include_columns = sorted(set(cfg.include_columns + parse_csv_arg(keep_raw)))

    max_card = input(f"Max categorical cardinality [{cfg.max_categorical_cardinality}]: ").strip()
    if max_card:
        try:
            cfg.max_categorical_cardinality = max(2, int(max_card))
        except Exception:
            pass

    group_raw = input(f"Group column for leakage-safe split/CV [{cfg.group_column or 'auto'}]: ").strip()
    if group_raw:
        cfg.group_column = group_raw

    run_mode = prompt_choice("Run profile:", ["fast", "complete"], default_idx=0 if cfg.run_mode == "fast" else 1)
    cfg.run_mode = run_mode

    models_raw = input("Specific model keys (comma separated, blank=auto profile): ").strip()
    if models_raw:
        cfg.models = parse_csv_arg(models_raw)

    ext_raw = input("External validation datasets (comma separated, blank=none): ").strip()
    if ext_raw:
        cfg.external_datasets = parse_csv_arg(ext_raw)


def _cache_file_path(cfg: RunConfig, dataset: str, target: str, stage: str) -> Path:
    cache_root = ensure_dir(Path(cfg.cache_dir))
    return cache_root / f"{sanitize_filename(dataset)}__{sanitize_filename(target)}__{stage}.parquet"

def _profile_cache_tag(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "default"
    try:
        payload = json.dumps({"_cache_schema": 4, "profile": profile}, sort_keys=True, default=str)
    except Exception:
        payload = str(profile)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def _load_or_compute_enriched_dataframe(
    cfg: RunConfig,
    dataset_name: str,
    target: str,
    df: pd.DataFrame,
    profile: dict[str, Any] | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    matminer_info: dict[str, Any] = {}
    material_info: dict[str, Any] = {}
    cache_stage = f"enriched_{_profile_cache_tag(profile)}"

    if cfg.use_cache:
        cache_file = _cache_file_path(cfg, dataset_name, target, cache_stage)
        cache_pickle = cache_file.with_suffix(".pkl")

        if cache_file.exists():
            try:
                loaded = pd.read_parquet(cache_file)
                logger.info("Loaded enriched dataframe from cache: %s", cache_file)
                return loaded, {"cached": True, "cache_format": "parquet"}, {"cached": True}
            except Exception:
                logger.warning("Failed to read parquet cache; trying pickle fallback.")

        if cache_pickle.exists():
            try:
                loaded = pd.read_pickle(cache_pickle)
                logger.info("Loaded enriched dataframe from pickle cache: %s", cache_pickle)
                return loaded, {"cached": True, "cache_format": "pickle"}, {"cached": True}
            except Exception:
                logger.warning("Failed to read pickle cache; recomputing.")

    enriched, matminer_info, material_info = apply_dataset_profile_enrichment(
        df,
        target,
        profile,
        logger,
        enable_matminer=cfg.enable_matminer_featurizers,
        enable_material_enrichment=cfg.enable_material_descriptor_enrichment,
        force_formula_core_featurizers=cfg.force_formula_core_featurizers,
        use_alloy_featurizer_precheck=cfg.use_alloy_featurizer_precheck,
    )

    if cfg.use_cache:
        cache_file = _cache_file_path(cfg, dataset_name, target, cache_stage)
        cache_pickle = cache_file.with_suffix(".pkl")
        try:
            enriched.to_parquet(cache_file, index=False)
            logger.info("Saved enriched dataframe cache: %s", cache_file)
        except Exception as exc:
            logger.warning("Could not save parquet enriched cache: %s", exc)
            try:
                enriched.to_pickle(cache_pickle)
                logger.info("Saved enriched dataframe pickle cache fallback: %s", cache_pickle)
            except Exception as exc2:
                logger.warning("Could not save pickle enriched cache fallback: %s", exc2)

    return enriched, matminer_info, material_info

def _plot_global_performance(results_df: pd.DataFrame, task: str, out_file: Path, use_secondary_axis: bool) -> None:
    if results_df.empty:
        return

    if task == "regression" and use_secondary_axis:
        need_cols = ["test_r2", "test_rmse", "test_mae"]
        if not all(col in results_df.columns for col in need_cols):
            return

        x = np.arange(len(results_df))
        width = 0.25
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        b1 = ax1.bar(x - width, results_df["test_r2"], width=width, color="tab:blue", label="Test R2")
        b2 = ax2.bar(x, results_df["test_rmse"], width=width, color="tab:orange", alpha=0.9, label="Test RMSE")
        b3 = ax2.bar(x + width, results_df["test_mae"], width=width, color="tab:green", alpha=0.9, label="Test MAE")

        ax1.set_ylabel("R2")
        ax2.set_ylabel("RMSE / MAE")
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df["model"], rotation=35)
        ax1.set_title("Model Performance Comparison (Dual Axis)")

        handles = [b1, b2, b3]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3)
        save_plot(out_file)
        return

    if task == "classification" and use_secondary_axis:
        primary_cols = [col for col in ["test_accuracy", "test_f1_weighted"] if col in results_df.columns]
        if not primary_cols:
            return

        x = np.arange(len(results_df))
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        ax1.bar(x - width / 2, results_df[primary_cols[0]], width=width, color="tab:blue", label=primary_cols[0])
        if len(primary_cols) > 1:
            ax1.bar(x + width / 2, results_df[primary_cols[1]], width=width, color="tab:green", label=primary_cols[1])

        if "test_log_loss" in results_df.columns:
            ax2.plot(x, results_df["test_log_loss"], marker="o", color="tab:red", label="test_log_loss")
            ax2.set_ylabel("Log Loss")

        ax1.set_ylabel("Accuracy / F1")
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df["model"], rotation=35)
        ax1.set_title("Classification Performance (Dual Axis)")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3)
        save_plot(out_file)
        return

    if task == "regression":
        y_cols = [col for col in ["test_r2", "test_rmse", "test_mae"] if col in results_df.columns]
    else:
        y_cols = [col for col in ["test_accuracy", "test_f1_weighted", "test_log_loss"] if col in results_df.columns]

    if not y_cols:
        return

    plot_df = results_df[["model"] + y_cols].melt(id_vars="model", var_name="Metric", value_name="Value")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="model", y="Value", hue="Metric")
    plt.xticks(rotation=35)
    plt.title("Model Performance Comparison")
    save_plot(out_file)


def _plot_runtime(results_df: pd.DataFrame, out_file: Path) -> None:
    if "total_seconds" not in results_df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="model", y="total_seconds", color="steelblue")
    plt.xticks(rotation=35)
    plt.ylabel("Seconds")
    plt.title("Model Runtime Comparison")
    save_plot(out_file)



def _finalize_integrity_checks(
    rows: list[dict[str, Any]],
    out_file: Path,
    strict_checks: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(out_file, index=False)

    failed = frame[(frame["passed"] == 0) & (frame["critical"] == 1)] if not frame.empty else pd.DataFrame()
    if not failed.empty:
        msg = "; ".join([f"{r.check}: {r.details}" for r in failed.itertuples()])
        if strict_checks:
            raise RuntimeError(f"Critical integrity checks failed: {msg}")
        logger.warning("Critical integrity checks failed (strict mode disabled): %s", msg)

    return frame


def _build_raw_integrity_rows(df: pd.DataFrame, target_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _safe_object_to_text(v: Any) -> str:
        if v is None:
            return "<NA>"
        try:
            missing = pd.isna(v)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return "<NA>"
        except Exception:
            pass
        return str(v)

    target_exists = int(target_col in df.columns)
    rows.append({"check": "target_exists", "passed": target_exists, "critical": 1, "value": target_exists, "details": target_col})

    if target_col in df.columns:
        target_non_null = int(df[target_col].notna().sum())
        target_non_null_ratio = float(df[target_col].notna().mean()) if len(df) > 0 else 0.0
    else:
        target_non_null = 0
        target_non_null_ratio = 0.0

    rows.append(
        {
            "check": "target_non_null_ratio_gt_0",
            "passed": int(target_non_null_ratio > 0.0),
            "critical": 1,
            "value": target_non_null_ratio,
            "details": f"non_null={target_non_null}",
        }
    )

    rows.append(
        {
            "check": "n_rows_ge_10",
            "passed": int(len(df) >= 10),
            "critical": 0,
            "value": int(len(df)),
            "details": "small datasets are allowed but unstable",
        }
    )

    if len(df) > 0:
        try:
            duplicate_rows = int(df.duplicated().sum())
        except TypeError:
            safe_cols: dict[str, pd.Series] = {}
            for col in df.columns:
                s = df[col]
                if (
                    pd.api.types.is_numeric_dtype(s)
                    or pd.api.types.is_bool_dtype(s)
                    or pd.api.types.is_datetime64_any_dtype(s)
                    or pd.api.types.is_timedelta64_dtype(s)
                ):
                    safe_cols[str(col)] = s
                else:
                    safe_cols[str(col)] = s.map(_safe_object_to_text)
            duplicate_rows = int(pd.DataFrame(safe_cols, index=df.index).duplicated().sum())
    else:
        duplicate_rows = 0
    rows.append(
        {
            "check": "duplicate_rows_zero",
            "passed": int(duplicate_rows == 0),
            "critical": 0,
            "value": duplicate_rows,
            "details": "duplicate raw rows",
        }
    )

    duplicate_cols = int(df.columns.duplicated().sum())
    rows.append(
        {
            "check": "duplicate_columns_zero",
            "passed": int(duplicate_cols == 0),
            "critical": 1,
            "value": duplicate_cols,
            "details": "duplicate column labels",
        }
    )

    return rows


def _build_feature_integrity_rows(X: pd.DataFrame, y: pd.Series, target_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append({"check": "feature_rows_gt_0", "passed": int(len(X) > 0), "critical": 1, "value": int(len(X)), "details": "rows after preprocessing"})
    rows.append({"check": "feature_cols_gt_0", "passed": int(X.shape[1] > 0), "critical": 1, "value": int(X.shape[1]), "details": "columns after preprocessing"})
    rows.append({"check": "x_y_same_rows", "passed": int(len(X) == len(y)), "critical": 1, "value": int(len(X) - len(y)), "details": "row count diff"})
    rows.append({"check": "x_y_same_index", "passed": int(X.index.equals(y.index)), "critical": 1, "value": int(X.index.equals(y.index)), "details": "index alignment"})
    rows.append({"check": "target_not_in_features", "passed": int(target_col not in X.columns), "critical": 1, "value": int(target_col in X.columns), "details": target_col})
    rows.append({"check": "feature_index_unique", "passed": int(X.index.is_unique), "critical": 1, "value": int(X.index.is_unique), "details": "index uniqueness"})
    return rows


def _build_split_integrity_rows(
    X_train: pd.DataFrame,
    X_calib: pd.DataFrame,
    X_test: pd.DataFrame,
    g_train: pd.Series | None,
    g_calib: pd.Series | None,
    g_test: pd.Series | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.append({"check": "train_non_empty", "passed": int(len(X_train) > 0), "critical": 1, "value": int(len(X_train)), "details": "train rows"})
    rows.append({"check": "calib_non_empty", "passed": int(len(X_calib) > 0), "critical": 1, "value": int(len(X_calib)), "details": "calibration rows"})
    rows.append({"check": "test_non_empty", "passed": int(len(X_test) > 0), "critical": 1, "value": int(len(X_test)), "details": "test rows"})

    idx_train = set(X_train.index)
    idx_cal = set(X_calib.index)
    idx_test = set(X_test.index)

    rows.append({"check": "train_calib_no_overlap", "passed": int(len(idx_train & idx_cal) == 0), "critical": 1, "value": int(len(idx_train & idx_cal)), "details": "shared sample indices"})
    rows.append({"check": "train_test_no_overlap", "passed": int(len(idx_train & idx_test) == 0), "critical": 1, "value": int(len(idx_train & idx_test)), "details": "shared sample indices"})
    rows.append({"check": "calib_test_no_overlap", "passed": int(len(idx_cal & idx_test) == 0), "critical": 1, "value": int(len(idx_cal & idx_test)), "details": "shared sample indices"})

    if g_train is not None and g_calib is not None and g_test is not None:
        set_train = set(g_train.astype("string").tolist())
        set_cal = set(g_calib.astype("string").tolist())
        set_test = set(g_test.astype("string").tolist())

        rows.append({"check": "train_calib_group_no_overlap", "passed": int(len(set_train & set_cal) == 0), "critical": 1, "value": int(len(set_train & set_cal)), "details": "shared groups"})
        rows.append({"check": "train_test_group_no_overlap", "passed": int(len(set_train & set_test) == 0), "critical": 1, "value": int(len(set_train & set_test)), "details": "shared groups"})
        rows.append({"check": "calib_test_group_no_overlap", "passed": int(len(set_cal & set_test) == 0), "critical": 1, "value": int(len(set_cal & set_test)), "details": "shared groups"})

    return rows


def _plot_cv_primary_with_errorbars(results_df: pd.DataFrame, task: str, out_file: Path) -> None:
    if results_df.empty or "cv_primary" not in results_df.columns:
        return

    order = results_df.sort_values("cv_primary", ascending=(task == "regression"))
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        np.arange(len(order)),
        order["cv_primary"],
        yerr=order["cv_std"] if "cv_std" in order.columns else None,
        fmt="o",
        capsize=4,
        color="tab:blue",
    )
    plt.xticks(np.arange(len(order)), order["model"], rotation=35)
    plt.ylabel("CV primary metric")
    plt.title("Cross-validation Primary Metric with Error Bars")
    save_plot(out_file)


def _plot_metric_vs_runtime(results_df: pd.DataFrame, task: str, out_file: Path) -> None:
    if results_df.empty or "total_seconds" not in results_df.columns or "cv_primary" not in results_df.columns:
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=results_df, x="total_seconds", y="cv_primary", hue="model", s=90)
    plt.xlabel("Total Runtime (s)")
    plt.ylabel("CV primary metric")
    plt.title("Accuracy-Speed Tradeoff")

    for _, row in results_df.iterrows():
        plt.text(float(row["total_seconds"]), float(row["cv_primary"]), str(row["model"]), fontsize=8)

    save_plot(out_file)


def _plot_global_error_rate_vs_significance(curves: list[tuple[str, pd.DataFrame]], out_file: Path) -> None:
    if not curves:
        return

    alpha_grid = np.round(np.arange(0.05, 1.0001, 0.05), 6)

    plt.figure(figsize=(10, 7))
    plotted = False
    for label, frame in curves:
        if frame.empty or "alpha" not in frame.columns or "miscoverage" not in frame.columns:
            continue

        cur = frame[["alpha", "miscoverage"]].copy()
        cur["alpha"] = pd.to_numeric(cur["alpha"], errors="coerce")
        cur["miscoverage"] = pd.to_numeric(cur["miscoverage"], errors="coerce")
        cur = cur.dropna().sort_values("alpha")
        if cur.empty:
            continue

        cur = cur.drop_duplicates(subset=["alpha"], keep="last")
        on_grid = cur[cur["alpha"].round(6).isin(alpha_grid)]

        if len(on_grid) >= 2:
            use_alpha = on_grid["alpha"].to_numpy(dtype=float)
            use_misc = on_grid["miscoverage"].to_numpy(dtype=float)
        else:
            use_alpha = alpha_grid
            use_misc = np.interp(alpha_grid, cur["alpha"].to_numpy(dtype=float), cur["miscoverage"].to_numpy(dtype=float))

        mask = np.isfinite(use_alpha) & np.isfinite(use_misc)
        if int(np.sum(mask)) < 2:
            continue

        plt.plot(use_alpha[mask], use_misc[mask], marker="o", markersize=4, linewidth=1.8, label=label)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Significance Level (alpha)")
    plt.ylabel("Observed Error Rate (Miscoverage)")
    plt.title("Error Rate vs Significance Level Across Models")
    plt.legend(loc="best")
    save_plot(out_file)

def _plot_permutation_pvalue_heatmap(frame: pd.DataFrame, out_file: Path) -> None:
    if frame.empty:
        return
    if not {"model_a", "model_b", "permutation_p_value"}.issubset(frame.columns):
        return

    labels = sorted(set(frame["model_a"].astype(str).tolist() + frame["model_b"].astype(str).tolist()))
    mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    for r in frame.itertuples():
        mat.loc[str(r.model_a), str(r.model_b)] = float(r.permutation_p_value)
        mat.loc[str(r.model_b), str(r.model_a)] = float(r.permutation_p_value)
    np.fill_diagonal(mat.values, 0.0)

    plt.figure(figsize=(9, 7))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis_r", vmin=0.0, vmax=1.0)
    plt.title("Pairwise Permutation-Test p-values")
    save_plot(out_file)
def _resolve_optional_target_transform(current: str | None, y: pd.Series, logger: logging.Logger) -> str | None:
    key = str(current or "").strip().lower()
    if key in {"", "none", "off", "false"}:
        return None

    optional_log = {
        "optional_log1p_if_skewed",
        "log1p_if_skewed",
        "optional_log1p_if_positive_skewed",
        "optional_log1p_if_skewed_positive",
    }

    if key in optional_log:
        y_num = pd.to_numeric(y, errors="coerce")
        y_num = y_num[np.isfinite(y_num)]
        if y_num.empty or float(y_num.min()) <= -1.0:
            logger.info("Target transform decision: skipped optional log1p (non-positive support).")
            return None

        skew = float(y_num.skew())
        q10 = float(np.quantile(y_num, 0.10))
        q90 = float(np.quantile(y_num, 0.90))
        ratio = float((q90 + 1e-12) / (q10 + 1e-12)) if q10 > 0 else float("nan")
        ratio_trigger = bool(np.isfinite(ratio) and ratio > 8.0)

        if skew > 0.75 or ratio_trigger:
            logger.info("Target transform decision: applying log1p (skew=%.3f, q90/q10=%.3f).", skew, ratio)
            return "log1p"

        logger.info("Target transform decision: optional log1p not needed (skew=%.3f, q90/q10=%.3f).", skew, ratio)
        return None

    return current



def _save_preprocessed_dataset_snapshot(
    X: pd.DataFrame,
    y: pd.Series,
    target_col: str,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    snapshot = X.copy()
    snapshot[target_col] = y
    snapshot.to_csv(output_dir / "dataset_after_preprocessing.csv", index=True, index_label="row_index")
    try:
        snapshot.to_parquet(output_dir / "dataset_after_preprocessing.parquet", index=True)
    except Exception as exc:
        logger.warning("Could not save dataset_after_preprocessing.parquet: %s", exc)



def _export_featurization_failure_rows(
    output_dir: Path,
    enrichment_info: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    parse_failed = enrichment_info.get("formula_parse_failed_rows", []) if isinstance(enrichment_info, dict) else []
    for row_idx in parse_failed or []:
        rows.append(
            {
                "row_index": row_idx,
                "stage": "formula_parse",
                "featurizer": "pymatgen.Composition",
                "reason": "composition_parse_failed",
            }
        )

    for item in (enrichment_info.get("row_failures", []) if isinstance(enrichment_info, dict) else []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "row_index": item.get("row_index"),
                "stage": item.get("stage", "unknown"),
                "featurizer": item.get("featurizer", "unknown"),
                "reason": item.get("reason", "unknown"),
            }
        )

    frame = pd.DataFrame(rows, columns=["row_index", "stage", "featurizer", "reason"])
    frame.to_csv(output_dir / "featurization_failed_rows.csv", index=False)
    logger.info("Featurization failure rows exported: %d", int(len(frame)))
    return frame


def _apply_train_only_feature_filters(
    X_train: pd.DataFrame,
    X_calib: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: RunConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    x_tr = X_train.copy()
    x_ca = X_calib.copy()
    x_te = X_test.copy()

    low_var_rows: list[dict[str, Any]] = []
    high_corr_rows: list[dict[str, Any]] = []

    active_numeric = [c for c in numeric_cols if c in x_tr.columns]

    if cfg.enable_train_variance_filter and active_numeric:
        x_num = x_tr[active_numeric].apply(pd.to_numeric, errors="coerce")
        var = x_num.var(axis=0, skipna=True)
        drop_var = []
        for col in active_numeric:
            v = float(var.get(col, np.nan))
            if (not np.isfinite(v)) or (v <= float(cfg.train_variance_threshold)):
                drop_var.append(col)
                low_var_rows.append(
                    {
                        "feature": col,
                        "train_variance": v,
                        "threshold": float(cfg.train_variance_threshold),
                    }
                )

        if drop_var:
            x_tr = x_tr.drop(columns=drop_var, errors="ignore")
            x_ca = x_ca.drop(columns=drop_var, errors="ignore")
            x_te = x_te.drop(columns=drop_var, errors="ignore")

    active_numeric = [c for c in active_numeric if c in x_tr.columns]

    if cfg.enable_train_correlation_filter and len(active_numeric) >= 2:
        x_num = x_tr[active_numeric].apply(pd.to_numeric, errors="coerce")
        med = x_num.median(axis=0, skipna=True)
        x_num = x_num.fillna(med).fillna(0.0)

        corr_abs = x_num.corr().abs().fillna(0.0)
        upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

        y_num = pd.to_numeric(y_train.reindex(x_num.index), errors="coerce")
        target_corr: dict[str, float] = {}
        for col in active_numeric:
            xv = x_num[col].to_numpy(dtype=float)
            yv = y_num.to_numpy(dtype=float)
            mask = np.isfinite(xv) & np.isfinite(yv)
            if int(np.sum(mask)) < 3 or float(np.std(xv[mask])) < 1e-12 or float(np.std(yv[mask])) < 1e-12:
                target_corr[col] = float("nan")
            else:
                target_corr[col] = float(abs(np.corrcoef(xv[mask], yv[mask])[0, 1]))

        to_drop: set[str] = set()
        sorted_pairs = upper.stack().sort_values(ascending=False)
        thr = float(cfg.train_correlation_threshold)

        for (a, b), c in sorted_pairs.items():
            c_val = float(c)
            if c_val < thr:
                break
            if a in to_drop or b in to_drop:
                continue

            ca = target_corr.get(str(a), float("nan"))
            cb = target_corr.get(str(b), float("nan"))
            if np.isnan(ca) and np.isnan(cb):
                drop_col = str(max(str(a), str(b)))
                keep_col = str(min(str(a), str(b)))
            elif np.isnan(ca):
                drop_col = str(a)
                keep_col = str(b)
            elif np.isnan(cb):
                drop_col = str(b)
                keep_col = str(a)
            elif ca < cb:
                drop_col = str(a)
                keep_col = str(b)
            elif cb < ca:
                drop_col = str(b)
                keep_col = str(a)
            else:
                drop_col = str(max(str(a), str(b)))
                keep_col = str(min(str(a), str(b)))

            to_drop.add(drop_col)
            high_corr_rows.append(
                {
                    "feature_a": str(a),
                    "feature_b": str(b),
                    "abs_corr_train": c_val,
                    "corr_with_target_a": ca,
                    "corr_with_target_b": cb,
                    "dropped_feature": drop_col,
                    "kept_feature": keep_col,
                    "threshold": thr,
                }
            )

        if to_drop:
            cols = sorted(to_drop)
            x_tr = x_tr.drop(columns=cols, errors="ignore")
            x_ca = x_ca.drop(columns=cols, errors="ignore")
            x_te = x_te.drop(columns=cols, errors="ignore")

    final_numeric = [c for c in numeric_cols if c in x_tr.columns]
    final_categorical = [c for c in categorical_cols if c in x_tr.columns]

    if (len(final_numeric) + len(final_categorical)) == 0:
        raise RuntimeError("Train-only feature filtering removed all features.")

    low_var_df = pd.DataFrame(low_var_rows, columns=["feature", "train_variance", "threshold"])
    high_corr_df = pd.DataFrame(
        high_corr_rows,
        columns=[
            "feature_a",
            "feature_b",
            "abs_corr_train",
            "corr_with_target_a",
            "corr_with_target_b",
            "dropped_feature",
            "kept_feature",
            "threshold",
        ],
    )

    low_var_df.to_csv(output_dir / "low_variance_removed.csv", index=False)
    high_corr_df.to_csv(output_dir / "high_correlation_removed.csv", index=False)

    logger.info(
        "Train-only feature filtering removed %d low-variance and %d high-correlation features.",
        int(len(low_var_df)),
        int(len(high_corr_df)),
    )

    return x_tr, x_ca, x_te, final_numeric, final_categorical, low_var_df, high_corr_df


def _apply_target_outlier_cleaning(
    X: pd.DataFrame,
    y: pd.Series,
    target_col: str,
    task: str,
    cfg: RunConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Series]:
    method = str(cfg.outlier_cleaning or "none").strip().lower()
    if task != "regression" or method in {"", "none", "off", "false"}:
        return X, y

    y_num = pd.to_numeric(y, errors="coerce")
    finite_mask = pd.Series(np.isfinite(y_num.to_numpy(dtype=float)), index=y.index)
    keep = finite_mask.copy()

    if method == "target_iqr":
        finite = y_num.loc[finite_mask]
        if finite.empty:
            logger.warning("Outlier cleaning skipped: no finite numeric target values available.")
            return X, y
        q1 = float(np.quantile(finite, 0.25))
        q3 = float(np.quantile(finite, 0.75))
        iqr = float(q3 - q1)
        if (not np.isfinite(iqr)) or iqr <= 1e-12:
            logger.warning(
                "Outlier cleaning skipped for %s: IQR is ~0 (q1=%.6g, q3=%.6g).",
                target_col,
                q1,
                q3,
            )
            return X, y
        lower = float(q1 - cfg.outlier_iqr_multiplier * iqr)
        upper = float(q3 + cfg.outlier_iqr_multiplier * iqr)
        keep &= y_num.between(lower, upper, inclusive="both")
        details = f"lower={lower:.6g}, upper={upper:.6g}, iqr={iqr:.6g}"
    elif method == "target_zscore":
        finite = y_num.loc[finite_mask]
        if finite.empty:
            logger.warning("Outlier cleaning skipped: no finite numeric target values available.")
            return X, y
        mean = float(finite.mean())
        std = float(finite.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            logger.info("Outlier cleaning skipped: target std is ~0.")
            return X, y
        z = (y_num - mean).abs() / (std + 1e-12)
        keep &= (z <= float(cfg.outlier_zscore_threshold))
        details = f"mean={mean:.6g}, std={std:.6g}, z_threshold={float(cfg.outlier_zscore_threshold):.3g}"
    else:
        logger.warning("Unknown outlier-cleaning mode %s; skipping.", method)
        return X, y

    removed = (~keep) | (~finite_mask)
    removed_count = int(removed.sum())
    if removed_count <= 0:
        logger.info("Outlier cleaning mode=%s removed 0 rows (%s).", method, details)
        return X, y

    X_clean = X.loc[keep].copy()
    y_clean = y.loc[keep].copy()
    if len(X_clean) == 0:
        raise RuntimeError("Outlier cleaning removed all rows; adjust thresholds or disable outlier cleaning.")
    if task == "regression":
        y_clean_num = pd.to_numeric(y_clean, errors="coerce")
        unique_after = int(y_clean_num.nunique(dropna=True))
        if unique_after < 2:
            logger.warning(
                "Outlier cleaning mode=%s would leave a constant target (unique=%d); skipping outlier cleaning.",
                method,
                unique_after,
            )
            return X, y

    removed_frame = pd.DataFrame(
        {
            "row_index": y.index[removed].tolist(),
            target_col: y_num.loc[removed].tolist(),
            "outlier_cleaning_mode": method,
        }
    )
    removed_frame.to_csv(output_dir / "outliers_removed.csv", index=False)
    logger.info("Outlier cleaning mode=%s removed %d rows (%s).", method, removed_count, details)
    return X_clean, y_clean
def _plot_bayesian_posterior_heatmap(frame: pd.DataFrame, out_file: Path) -> None:
    if frame.empty or not {"model_a", "model_b", "p_a_better"}.issubset(frame.columns):
        return

    labels = sorted(set(frame["model_a"].astype(str).tolist() + frame["model_b"].astype(str).tolist()))
    mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    for r in frame.itertuples():
        mat.loc[str(r.model_a), str(r.model_b)] = float(r.p_a_better)
        mat.loc[str(r.model_b), str(r.model_a)] = 1.0 - float(r.p_a_better)
    np.fill_diagonal(mat.values, 0.5)

    plt.figure(figsize=(9, 7))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title("Bayesian Correlated t-test: P(model_i better than model_j)")
    save_plot(out_file)

def run_pipeline(cfg: RunConfig) -> Path:
    sns.set_theme(style="white", rc={"axes.grid": False})
    get_available_datasets, load_dataset = get_matminer_dataset_functions()

    datasets = sorted(get_available_datasets())
    if cfg.list_datasets:
        for ds in datasets:
            print(ds)
        return Path(".")

    if cfg.dataset is None:
        if cfg.non_interactive:
            raise ValueError("--dataset required with --non-interactive")
        cfg.dataset = prompt_choice("Select dataset:", datasets, 0)

    if cfg.dataset not in datasets:
        raise ValueError(f"Dataset {cfg.dataset} not found")

    df = None
    load_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            df = load_dataset(cfg.dataset)
            load_exc = None
            break
        except Exception as exc:
            load_exc = exc
            if attempt < 3:
                time.sleep(float(attempt))

    if load_exc is not None:
        msg = str(load_exc)
        low = msg.lower()
        if any(tok in low for tok in ["failed to resolve", "name resolution", "getaddrinfo", "max retries exceeded", "httpsconnectionpool", "connectionerror"]):
            raise RuntimeError(
                "Dataset download failed after 3 attempts due to a network/DNS issue while accessing matminer's Figshare host. "
                f"Dataset={cfg.dataset!r}. Ensure internet/DNS/proxy access, or pre-download the dataset into the matminer datasets folder and rerun. "
                f"Original error: {msg}"
            ) from load_exc
        raise RuntimeError(f"Failed to load dataset {cfg.dataset!r}: {msg}") from load_exc

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    profile_file = cfg.dataset_config_file
    if not profile_file:
        default_profile_file = Path.home() / "Downloads" / "datasets.json"
        if default_profile_file.exists():
            profile_file = str(default_profile_file)

    dataset_profile = get_dataset_config(cfg.dataset, external_config_file=profile_file)

    if dataset_profile is not None:
        profile_target = dataset_profile.get("target_col")
        if cfg.target is None and profile_target in df.columns:
            cfg.target = str(profile_target)

        if cfg.task is None and dataset_profile.get("task_type") in {"regression", "classification"}:
            cfg.task = str(dataset_profile.get("task_type"))

        if cfg.group_column is None and dataset_profile.get("group_split_col"):
            cfg.group_column = str(dataset_profile.get("group_split_col"))

        if not cfg.models and dataset_profile.get("model_keys"):
            cfg.models = [str(item) for item in dataset_profile.get("model_keys", [])]

        if str(cfg.target_transform or "").strip() == "":
            cfg.target_transform = dataset_profile.get("target_transform")
        profile_outlier_mode = str(dataset_profile.get("outlier_cleaning", "") or "").strip().lower()
        if str(cfg.outlier_cleaning or "none").strip().lower() in {"", "none"} and profile_outlier_mode:
            cfg.outlier_cleaning = profile_outlier_mode
        if float(cfg.outlier_iqr_multiplier) == 1.5 and dataset_profile.get("outlier_iqr_multiplier") is not None:
            cfg.outlier_iqr_multiplier = float(dataset_profile.get("outlier_iqr_multiplier"))
        if float(cfg.outlier_zscore_threshold) == 4.0 and dataset_profile.get("outlier_zscore_threshold") is not None:
            cfg.outlier_zscore_threshold = float(dataset_profile.get("outlier_zscore_threshold"))
        cfg.imputation = str(dataset_profile.get("imputation", cfg.imputation))
        cfg.scaling = str(dataset_profile.get("scaling", cfg.scaling))

        profile_cat = [str(item) for item in dataset_profile.get("categorical_cols", []) or []]
        if profile_cat:
            cfg.include_columns = sorted(set(cfg.include_columns + profile_cat))

        profile_nonfeature = [str(item) for item in dataset_profile.get("special_nonfeature_cols", []) or []]
        if profile_nonfeature:
            cfg.drop_columns = sorted(set(cfg.drop_columns + profile_nonfeature))

    if cfg.target is None:
        if cfg.non_interactive:
            raise ValueError("--target required with --non-interactive")
        cfg.target = prompt_choice("Select target:", df.columns.tolist(), 0)

    if cfg.target not in df.columns:
        raise ValueError(f"Target {cfg.target} not in columns")

    if dataset_profile is not None:
        resolved_drop = resolve_target_specific_drops(dataset_profile, cfg.target)
        if resolved_drop:
            cfg.drop_columns = sorted(set(cfg.drop_columns + resolved_drop))
        dataset_profile["resolved_drop_cols"] = resolved_drop

    if cfg.task is None:
        inferred = infer_task_type(df[cfg.target])
        cfg.task = inferred if cfg.non_interactive else prompt_choice(
            f"Detected task: {inferred}", ["regression", "classification"], 0 if inferred == "regression" else 1
        )

    if not cfg.non_interactive:
        maybe_configure_interactively(cfg, df)

    output_dir = make_output_dir(cfg, cfg.dataset, cfg.target)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    models_dir = ensure_dir(output_dir / "models")
    logger = configure_logger(output_dir)

    logger.info("Running pipeline | dataset=%s target=%s task=%s mode=%s", cfg.dataset, cfg.target, cfg.task, cfg.run_mode)
    if profile_file and dataset_profile is not None:
        logger.info("Using dataset profile from external config file: %s", profile_file)
    logger.info("Target selected: %s", cfg.target)
    logger.info("Raw dataset shape: rows=%d cols=%d", int(len(df)), int(df.shape[1]))
    logger.info("User/profile drop columns: %s", cfg.drop_columns if cfg.drop_columns else "<none>")
    logger.info("Target transform setting (pre-auto): %s", cfg.target_transform or "none")
    logger.info("Outlier cleaning mode: %s", cfg.outlier_cleaning)
    save_json({"config": cfg.__dict__}, output_dir / "run_config.json")

    preview_dataset(df, output_dir, cfg.random_state, logger)
    run_dataset_diagnostics(df, cfg.target, cfg.task, cfg, output_dir, logger)

    raw_checks = _build_raw_integrity_rows(df, cfg.target)
    _finalize_integrity_checks(raw_checks, output_dir / "data_integrity_raw.csv", cfg.strict_checks, logger)

    enriched_df, matminer_info, material_info = _load_or_compute_enriched_dataframe(
        cfg,
        cfg.dataset,
        cfg.target,
        df,
        dataset_profile,
        logger,
    )

    featurization_info: dict[str, Any] = {}
    if isinstance(matminer_info, dict):
        featurization_info.update(matminer_info)
    if isinstance(material_info, dict):
        for k, v in material_info.items():
            featurization_info.setdefault(k, v)
    _export_featurization_failure_rows(output_dir, featurization_info, logger)


    X, y, numeric_cols, categorical_cols, dropped = prepare_features(enriched_df, cfg.target, cfg.task, cfg)
    X, y = _apply_target_outlier_cleaning(X, y, cfg.target, cfg.task, cfg, output_dir, logger)
    cfg.target_transform = _resolve_optional_target_transform(cfg.target_transform, y, logger)
    dropped.to_csv(output_dir / "dropped_features.csv", index=False)
    _save_preprocessed_dataset_snapshot(X, y, cfg.target, output_dir, logger)

    try:
        prepared_diag = X.copy()
        prepared_diag[cfg.target] = y
        run_dataset_diagnostics(prepared_diag, cfg.target, cfg.task, cfg, output_dir, logger)
        logger.info("Saved prepared-feature EDA diagnostics.")
    except Exception as exc:
        logger.warning("Prepared-feature EDA diagnostics failed: %s", exc)

    rows_removed = int(len(enriched_df) - len(X))
    logger.info("Rows removed during target cleaning/filtering: %d", rows_removed)
    logger.info("Features used: numeric=%d categorical=%d total=%d", int(len(numeric_cols)), int(len(categorical_cols)), int(X.shape[1]))
    logger.info("Columns dropped during feature preparation: %d", int(len(dropped)))

    removed_index = enriched_df.index.difference(y.index)
    if len(removed_index) > 0:
        logger.info("Removed row indices (first 30): %s", removed_index[:30].tolist())

    if not dropped.empty:
        logger.info("Dropped columns detail (first 50):\n%s", dropped.head(50).to_string(index=False))

    used_features_pre = pd.DataFrame(
        {
            "feature": numeric_cols + categorical_cols,
            "feature_type": ["numeric"] * len(numeric_cols) + ["categorical"] * len(categorical_cols),
            "stage": "prepared_before_split",
        }
    )
    used_features_pre.to_csv(output_dir / "used_features_pre_split.csv", index=False)
    if not used_features_pre.empty:
        logger.info("Prepared features before train-only filtering (first 60): %s", used_features_pre["feature"].head(60).tolist())

    generated_cols = []
    for info in [matminer_info, material_info]:
        if isinstance(info, dict):
            generated_cols.extend([str(c) for c in info.get("generated_columns", []) or []])
    generated_cols = sorted(set(generated_cols))
    pd.DataFrame({"generated_feature": generated_cols}).to_csv(output_dir / "generated_features.csv", index=False)
    logger.info("Generated features from enrichment: %d", int(len(generated_cols)))
    if generated_cols:
        logger.info("Generated feature names (first 60): %s", generated_cols[:60])

    feature_checks = _build_feature_integrity_rows(X, y, cfg.target)
    _finalize_integrity_checks(feature_checks, output_dir / "data_integrity_features.csv", cfg.strict_checks, logger)

    sample_weight_full: pd.Series | None = None
    if dataset_profile is not None:
        sw_col = dataset_profile.get("sample_weight_col")
        if sw_col and str(sw_col) in enriched_df.columns:
            sw_raw = pd.to_numeric(enriched_df.loc[y.index, str(sw_col)], errors="coerce")
            mode = str(dataset_profile.get("sample_weight_mode") or "direct").strip().lower()
            if mode == "inverse":
                sw_series = 1.0 / (sw_raw.abs() + 1e-12)
            else:
                sw_series = sw_raw

            finite = sw_series[np.isfinite(sw_series) & (sw_series > 0)]
            if not finite.empty:
                fill = float(np.median(finite))
                sample_weight_full = sw_series.where(np.isfinite(sw_series) & (sw_series > 0), fill).astype(float)
                sample_weight_full = sample_weight_full / (float(sample_weight_full.mean()) + 1e-12)
                logger.info("Using sample weights from column %s (mode=%s).", sw_col, mode)
            else:
                logger.warning("Sample-weight column %s did not contain usable positive values.", sw_col)

    group_col = None
    groups_full = None
    if cfg.use_group_aware_split or cfg.use_group_aware_cv:
        preferred_group_col = cfg.group_column

        if preferred_group_col is None and dataset_profile is not None:
            strategy = str(dataset_profile.get("group_split_strategy") or "").strip().lower()
            if strategy == "reduced_formula" and "__reduced_formula__" in enriched_df.columns:
                preferred_group_col = "__reduced_formula__"
            elif strategy in {"chemical_system", "binary_system"} and "__chemical_system__" in enriched_df.columns:
                preferred_group_col = "__chemical_system__"
            elif strategy == "chem_family_ab":
                for a_col, b_col in [("atom a", "atom b"), ("atom_a", "atom_b")]:
                    if a_col in enriched_df.columns and b_col in enriched_df.columns:
                        fam_col = "__chem_family_ab__"
                        enriched_df[fam_col] = (
                            enriched_df[a_col].astype("string").fillna("<NA>")
                            + "-"
                            + enriched_df[b_col].astype("string").fillna("<NA>")
                        )
                        preferred_group_col = fam_col
                        break

        if preferred_group_col is not None and preferred_group_col in enriched_df.columns:
            group_col = preferred_group_col
        else:
            group_col = detect_group_column(enriched_df.loc[y.index], cfg.target, cfg, logger)

        if group_col and group_col in enriched_df.columns:
            groups_full = enriched_df.loc[y.index, group_col].astype("string").fillna("<NA>")

    X_train, X_calib, X_test, y_train, y_calib, y_test, g_train, g_calib, g_test = split_dataset_with_groups(
        X,
        y,
        groups_full if cfg.use_group_aware_split else None,
        cfg.task,
        cfg.test_size,
        cfg.calibration_size,
        cfg.random_state,
    )

    X_train, X_calib, X_test, numeric_cols, categorical_cols, low_var_df, high_corr_df = _apply_train_only_feature_filters(
        X_train,
        X_calib,
        X_test,
        y_train,
        numeric_cols,
        categorical_cols,
        cfg,
        output_dir,
        logger,
    )

    split_checks = _build_split_integrity_rows(X_train, X_calib, X_test, g_train, g_calib, g_test)
    _finalize_integrity_checks(split_checks, output_dir / "data_integrity_splits.csv", cfg.strict_checks, logger)

    feature_space_shapes = pd.DataFrame(
        [
            {"split": "train", "rows": int(len(X_train)), "cols": int(X_train.shape[1])},
            {"split": "calibration", "rows": int(len(X_calib)), "cols": int(X_calib.shape[1])},
            {"split": "test", "rows": int(len(X_test)), "cols": int(X_test.shape[1])},
        ]
    )
    feature_space_shapes.to_csv(output_dir / "feature_space_shapes.csv", index=False)
    logger.info(
        "Final feature space after train-only filtering: train=%d x %d, calib=%d x %d, test=%d x %d",
        int(len(X_train)),
        int(X_train.shape[1]),
        int(len(X_calib)),
        int(X_calib.shape[1]),
        int(len(X_test)),
        int(X_test.shape[1]),
    )

    used_features = pd.DataFrame(
        {
            "feature": numeric_cols + categorical_cols,
            "feature_type": ["numeric"] * len(numeric_cols) + ["categorical"] * len(categorical_cols),
            "stage": "final_after_train_filters",
        }
    )
    used_features.to_csv(output_dir / "used_features.csv", index=False)
    if not used_features.empty:
        logger.info("Final used features (first 60): %s", used_features["feature"].head(60).tolist())

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, imputation=cfg.imputation, scaling=cfg.scaling)

    try:
        leakage_cols = [c for c in (numeric_cols + categorical_cols) if c in X.columns]
        leakage_X = X[leakage_cols].copy() if leakage_cols else X.copy()
        leakage_df = scan_feature_leakage(cfg, leakage_X, y, X_train, X_test, output_dir)
        if not leakage_df.empty:
            logger.info("Leakage scan findings exported: %d row(s).", int(len(leakage_df)))
    except Exception as exc:
        logger.warning("Leakage scan failed: %s", exc)

    sample_weight_train = sample_weight_full.reindex(X_train.index) if sample_weight_full is not None else None

    definitions = get_model_definitions(cfg, cfg.task, logger)
    export_tuning_spaces(definitions).to_csv(output_dir / "hyperparameter_ranges.csv", index=False)

    groups_for_nested = None
    if g_train is not None and g_test is not None:
        groups_for_nested = pd.concat([g_train, g_test], axis=0)

    model_results = train_models(
        cfg,
        cfg.task,
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

    if not model_results:
        extra = ""
        if cfg.task == "regression":
            y_train_num = pd.to_numeric(y_train, errors="coerce")
            unique_train = int(y_train_num.nunique(dropna=True))
            extra = (
                f" Regression target unique values in train={unique_train}. "
                f"Current outlier_cleaning={cfg.outlier_cleaning!r}. "
                "Try --outlier-cleaning none, relax thresholds, or include baseline in --models."
            )
        raise RuntimeError(f"No model results were produced; aborting run.{extra}")
    table = model_results_to_table(model_results, cfg.task)
    if table.empty:
        raise RuntimeError("Model metrics table is empty; aborting run.")
    table.to_csv(output_dir / "model_metrics.csv", index=False)

    _plot_global_performance(table, cfg.task, output_dir / "model_performance_comparison.png", cfg.performance_plot_secondary_axis)
    _plot_runtime(table, output_dir / "model_runtime_comparison.png")
    _plot_cv_primary_with_errorbars(table, cfg.task, output_dir / "model_cv_primary_with_errorbars.png")
    _plot_metric_vs_runtime(table, cfg.task, output_dir / "model_metric_vs_runtime.png")

    uncertainty_rows: list[pd.DataFrame] = []
    uncertainty_quality_map: dict[str, float] = {}
    model_summaries: list[dict[str, Any]] = []
    coverage_curves: list[tuple[str, pd.DataFrame]] = []

    for result in model_results:
        if cfg.task == "regression":
            conf_df, doa_df, summary = analyze_regression_model(
                cfg,
                result,
                X_train,
                y_train,
                X_calib,
                y_calib,
                X_test,
                y_test,
                models_dir,
                logger,
            )
            model_dir = models_dir / result.label
            uncertainty_boot = pd.read_csv(model_dir / "uncertainty_bootstrap_summary.csv")
            uncertainty_rows.append(uncertainty_boot)
            uncertainty_quality_map[result.key] = float(summary.get("ace", np.nan))
            coverage_curves.append((result.label, conf_df[["alpha", "miscoverage"]].copy()))

            pred_df = pd.DataFrame(
                {
                    "y_true": y_test,
                    "y_pred": result.y_test_pred,
                },
                index=y_test.index,
            )
            pred_df = pred_df.join(doa_df, how="left")
            pred_df.to_csv(model_dir / "test_predictions_with_doa.csv", index=True)
        else:
            summary = analyze_classification_model(cfg, result, X_test, y_test, models_dir, logger)
            uncertainty_quality_map[result.key] = float(summary.get("mean_entropy", np.nan))

        model_summaries.append({"model": result.label, **summary})

    uncertainty_summary_df = pd.concat(uncertainty_rows, ignore_index=True) if uncertainty_rows else pd.DataFrame()
    if not uncertainty_summary_df.empty:
        uncertainty_summary_df.to_csv(output_dir / "uncertainty_summary.csv", index=False)
        plot_global_uncertainty_comparison(uncertainty_summary_df, output_dir / "uncertainty_comparison.png")
    _plot_global_error_rate_vs_significance(coverage_curves, output_dir / "error_rate_vs_significance_models.png")

    ranking_df = rank_models_multi_objective(model_results, cfg.task, uncertainty_quality_map)
    ranking_df.to_csv(output_dir / "multi_objective_ranking.csv", index=False)

    stats_df = statistical_model_comparison(cfg.task, model_results, y_test, cfg.bootstrap_repeats, cfg.random_state)
    stats_df.to_csv(output_dir / "model_statistical_comparison.csv", index=False)

    perm_df = permutation_model_comparison(cfg.task, model_results, y_test, cfg.permutation_test_repeats, cfg.random_state)
    perm_df.to_csv(output_dir / "model_permutation_comparison.csv", index=False)
    _plot_permutation_pvalue_heatmap(perm_df, output_dir / "model_permutation_pvalue_heatmap.png")

    if cfg.enable_bayesian_model_comparison:
        try:
            bayes_df = bayesian_correlated_ttest_comparison(cfg.task, model_results, rope=cfg.bayesian_rope)
            bayes_df.to_csv(output_dir / "model_bayesian_comparison.csv", index=False)
            _plot_bayesian_posterior_heatmap(bayes_df, output_dir / "model_bayesian_posterior_heatmap.png")
        except Exception as exc:
            logger.warning("Bayesian model comparison failed: %s", exc)

    summary_df = pd.DataFrame(model_summaries)
    summary_df.to_csv(output_dir / "model_analysis_summary.csv", index=False)

    repeated_raw_df = pd.DataFrame()
    repeated_summary_df = pd.DataFrame()
    if cfg.run_mode == "complete":
        try:
            repeated_raw_df, repeated_summary_df = run_repeated_group_benchmark(
                cfg,
                cfg.task,
                preprocessor,
                X,
                y,
                groups_full if cfg.use_group_aware_split else None,
                sample_weight_full,
                output_dir,
                logger,
                definitions=definitions,
            )
        except Exception as exc:
            logger.warning("Repeated group-aware benchmark failed: %s", exc)

    if cfg.run_mode == "complete":
        try:
            groups_map: dict[str, pd.Series | None] = {}
            if "__reduced_formula__" in enriched_df.columns:
                groups_map["leave_chemistry_family_out"] = enriched_df.loc[y.index, "__reduced_formula__"].astype("string")
            elif groups_full is not None:
                groups_map["leave_chemistry_family_out"] = groups_full.astype("string")
            else:
                groups_map["leave_chemistry_family_out"] = None

            if "__chemical_system__" in enriched_df.columns:
                groups_map["leave_system_out"] = enriched_df.loc[y.index, "__chemical_system__"].astype("string")
            elif "system" in enriched_df.columns:
                groups_map["leave_system_out"] = enriched_df.loc[y.index, "system"].astype("string")
            else:
                groups_map["leave_system_out"] = None

            _ = run_leave_group_protocols(
                cfg,
                cfg.task,
                preprocessor,
                X,
                y,
                groups_map,
                definitions,
                output_dir,
                logger,
            )
        except Exception as exc:
            logger.warning("Leave-group protocols failed: %s", exc)

    best_model = choose_best_model(model_results, cfg.task)
    from joblib import dump

    dump(best_model.pipeline, output_dir / "best_model.joblib")

    if cfg.enable_ablation_study and cfg.run_mode == "complete":
        try:
            run_ablation_study(cfg, cfg.task, cfg.dataset, cfg.target, df, output_dir, logger)
        except Exception as exc:
            logger.warning("Ablation study failed: %s", exc)

    external_validation_df = pd.DataFrame()
    if cfg.enable_external_validation and cfg.external_datasets:
        try:
            external_validation_df = run_external_validation(
                cfg,
                cfg.task,
                cfg.target,
                cfg.dataset,
                best_model.pipeline,
                load_dataset,
                output_dir,
                logger,
            )
        except Exception as exc:
            logger.warning("External validation failed: %s", exc)

    best_pred_doa_df = pd.DataFrame()
    best_doa_file = models_dir / best_model.label / "test_predictions_with_doa.csv"
    if best_doa_file.exists():
        try:
            best_pred_doa_df = pd.read_csv(best_doa_file)
        except Exception as exc:
            logger.warning("Could not read DoA prediction file for cross-generalization table: %s", exc)

    try:
        _ = build_cross_dataset_generalization_table(
            cfg.task,
            best_model.label,
            table,
            external_validation_df,
            output_dir,
            best_pred_doa_df=best_pred_doa_df,
        )
    except Exception as exc:
        logger.warning("Cross-dataset generalization table failed: %s", exc)

    try:
        run_structure_graph_model_benchmark_full(
            cfg,
            dataset_profile,
            cfg.task,
            cfg.target,
            enriched_df,
            y,
            X_train.index,
            X_test.index,
            output_dir,
            logger,
        )
    except Exception as exc:
        logger.warning("Structure graph benchmark failed: %s", exc)

    if cfg.task == "regression":
        try:
            intervals_file = models_dir / best_model.label / "uncertainty_intervals_samples.csv"
            run_subgroup_robustness_breakdown(
                cfg,
                X_test,
                y_test,
                best_model.y_test_pred,
                intervals_file,
                g_test,
                output_dir,
            )
        except Exception as exc:
            logger.warning("Subgroup robustness breakdown failed: %s", exc)
    if cfg.enable_robustness_tests and cfg.task == "regression":
        try:
            run_robustness_tests(cfg, cfg.task, best_model.pipeline, X_train, X_test, y_test, output_dir)
        except Exception as exc:
            logger.warning("Robustness tests failed: %s", exc)

    if cfg.enable_physics_sanity_checks and cfg.task == "regression":
        try:
            run_physics_sanity_checks(cfg, cfg.target, y_train, best_model.y_test_pred, output_dir)
        except Exception as exc:
            logger.warning("Physics sanity checks failed: %s", exc)

    if cfg.enable_reproducibility_manifest:
        try:
            write_reproducibility_manifest(cfg, df, enriched_df, output_dir)
        except Exception as exc:
            logger.warning("Reproducibility manifest failed: %s", exc)

    if cfg.enable_experiment_registry:
        try:
            append_experiment_registry(cfg, output_dir, best_model.label, table)
        except Exception as exc:
            logger.warning("Experiment registry update failed: %s", exc)

    if cfg.publication_export_latex:
        try:
            export_publication_tables(
                output_dir,
                table,
                ranking_df,
                stats_df,
                uncertainty_summary_df if not uncertainty_summary_df.empty else pd.DataFrame(),
            )
        except Exception as exc:
            logger.warning("Publication tables export failed: %s", exc)

    checklist = {
        "Featurization failure rows exported": (output_dir / "featurization_failed_rows.csv").exists(),
        "Train-only low-variance report exported": (output_dir / "low_variance_removed.csv").exists(),
        "Train-only high-correlation report exported": (output_dir / "high_correlation_removed.csv").exists(),
        "Feature-space shape report exported": (output_dir / "feature_space_shapes.csv").exists(),
        "Raw full dataset exported": (output_dir / "dataset_raw_full.csv").exists(),
        "Preprocessed dataset exported": (output_dir / "dataset_after_preprocessing.csv").exists(),
        "Hyperparameter ranges exported": (output_dir / "hyperparameter_ranges.csv").exists(),
        "Model metrics exported": (output_dir / "model_metrics.csv").exists(),
        "Statistical comparison exported": (output_dir / "model_statistical_comparison.csv").exists(),
        "Uncertainty analysis exported": (output_dir / "uncertainty_summary.csv").exists() or cfg.task != "regression",
        "Reproducibility manifest exported": (output_dir / "reproducibility_manifest.json").exists() or not cfg.enable_reproducibility_manifest,
        "Ablation study exported": (output_dir / "ablation_study.csv").exists() or not (cfg.enable_ablation_study and cfg.run_mode == "complete"),
        "Robustness tests exported": (output_dir / "robustness_tests.csv").exists() or not (cfg.enable_robustness_tests and cfg.task == "regression"),
        "External validation exported": (output_dir / "external_validation_summary.csv").exists() or not (cfg.enable_external_validation and bool(cfg.external_datasets)),
        "Leakage scan exported": (output_dir / "leakage_scan.csv").exists() or not cfg.enable_leakage_scan,
        "Bayesian comparison exported": (output_dir / "model_bayesian_comparison.csv").exists() or not cfg.enable_bayesian_model_comparison,
        "Leave-group protocol exported": (output_dir / "leave_group_protocols.csv").exists() or not (cfg.enable_leave_group_protocols and cfg.run_mode == "complete"),
        "Environment lock exported": (output_dir / "environment_lock.txt").exists(),
        "Experiment tracking status exported": (output_dir / "experiment_tracking_status.csv").exists() or not cfg.enable_experiment_tracking,
    }
    generate_reporting_checklist(output_dir, checklist)

    save_json(
        {
            "run_id": run_id,
            "dataset": cfg.dataset,
            "target": cfg.target,
            "task": cfg.task,
            "best_model": best_model.label,
            "group_column": group_col,
            "profile_file": profile_file,
            "matminer_info": matminer_info,
            "material_info": material_info,
        },
        output_dir / "run_summary.json",
    )

    report_path = generate_html_report(
        output_dir,
        table,
        ranking_df,
        stats_df,
        uncertainty_summary_df if not uncertainty_summary_df.empty else pd.DataFrame(),
    )
    logger.info("HTML report saved to %s", report_path)

    try:
        export_environment_lock(output_dir)
        _ = compute_artifact_hashes(output_dir)
    except Exception as exc:
        logger.warning("Environment lock / artifact hashes export failed: %s", exc)

    if cfg.enable_reproducibility_manifest:
        try:
            write_reproducibility_manifest(cfg, df, enriched_df, output_dir)
        except Exception as exc:
            logger.warning("Final reproducibility manifest refresh failed: %s", exc)

    try:
        _ = track_experiment_backend(cfg, output_dir, run_id, table, logger)
    except Exception as exc:
        logger.warning("Experiment tracking backend failed: %s", exc)

    return output_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    out_dir = run_pipeline(cfg)
    print(f"Artifacts saved to: {out_dir.resolve()}")






































































