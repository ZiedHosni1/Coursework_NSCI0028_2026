from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import linregress, norm, probplot, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from .config import RunConfig
from .models import ModelResult, evaluate_regression_metrics
from .utils import ensure_dir


def save_plot(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _unwrap_pipeline(model: Any) -> Pipeline:
    if hasattr(model, "regressor_"):
        return model.regressor_  # fitted TransformedTargetRegressor
    if hasattr(model, "regressor"):
        return model.regressor
    return model


def get_transformed_feature_names(model: Any, fallback: list[str]) -> np.ndarray:
    pipe = _unwrap_pipeline(model)
    pre = pipe.named_steps["preprocessor"]
    try:
        return pre.get_feature_names_out()
    except Exception:
        return np.asarray(fallback, dtype=object)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if len(scores) == 0:
        return float("nan")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    rank = max(1, min(rank, len(scores)))
    sorted_scores = np.sort(scores)
    return float(sorted_scores[rank - 1])


def _as_2d_dense(matrix: Any) -> np.ndarray:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return dense


def reduce_matrix(matrix: Any, random_state: int, max_dim: int = 40) -> np.ndarray:
    if sparse.issparse(matrix):
        n_samples, n_features = matrix.shape
        if n_features <= max_dim:
            return matrix.toarray()
        max_allowed = min(n_samples, n_features)
        if max_allowed <= 1:
            return matrix.toarray()
        n_comp = min(max_dim, max_allowed)
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        return svd.fit_transform(matrix)

    dense = _as_2d_dense(matrix)
    if dense.shape[1] <= max_dim:
        return dense

    max_allowed = min(dense.shape[0], dense.shape[1])
    if max_allowed <= 1:
        return dense
    n_comp = min(max_dim, max_allowed)
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(dense)


def reduce_matrix_pair(
    train_matrix: Any,
    test_matrix: Any,
    random_state: int,
    max_dim: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(train_matrix) or sparse.issparse(test_matrix):
        train_sparse = train_matrix if sparse.issparse(train_matrix) else sparse.csr_matrix(train_matrix)
        n_samples, n_features = train_sparse.shape
        max_allowed = min(n_samples, n_features)
        if max_allowed <= 1 or n_features <= max_dim:
            return _as_2d_dense(train_matrix), _as_2d_dense(test_matrix)
        n_comp = min(max_dim, max_allowed)
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        return svd.fit_transform(train_sparse), svd.transform(test_matrix)

    train_dense = _as_2d_dense(train_matrix)
    test_dense = _as_2d_dense(test_matrix)
    max_allowed = min(train_dense.shape[0], train_dense.shape[1])
    if max_allowed <= 1 or train_dense.shape[1] <= max_dim:
        return train_dense, test_dense
    n_comp = min(max_dim, max_allowed)
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(train_dense), pca.transform(test_dense)


def _encode_mixed_frame_for_embedding(frame: pd.DataFrame) -> pd.DataFrame:
    encoded: dict[str, pd.Series] = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            num = pd.to_numeric(series, errors="coerce")
            fill = float(num.median()) if num.notna().any() else 0.0
            encoded[str(col)] = num.fillna(fill)
        else:
            fac, _ = pd.factorize(series.astype("string").fillna("<NA>"), sort=True)
            encoded[str(col)] = pd.Series(fac.astype(float), index=frame.index)
    return pd.DataFrame(encoded, index=frame.index)


def run_dataset_diagnostics(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    cfg: RunConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    if target_col not in df.columns:
        return

    frame = df.copy()
    y = pd.to_numeric(frame[target_col], errors="coerce") if task == "regression" else frame[target_col].astype("string")
    y = y.replace("nan", pd.NA)
    mask = y.notna()
    frame = frame.loc[mask].copy()
    y = y.loc[mask].copy()

    if len(frame) == 0:
        return

    sample_n = min(cfg.eda_sample_size, len(frame))
    sampled = frame.sample(sample_n, random_state=cfg.random_state) if len(frame) > sample_n else frame
    y_sampled = y.loc[sampled.index]

    try:
        plt.figure(figsize=(8, 5))
        if task == "regression":
            sns.histplot(pd.to_numeric(y_sampled, errors="coerce"), bins=30, kde=True, color="tab:blue")
            plt.xlabel(target_col)
            plt.ylabel("Count")
            plt.title("Target Distribution")
        else:
            counts = y_sampled.astype("string").value_counts().sort_values(ascending=False)
            sns.barplot(x=counts.index.astype(str), y=counts.values, color="tab:blue")
            plt.xticks(rotation=35)
            plt.xlabel(target_col)
            plt.ylabel("Count")
            plt.title("Target Class Distribution")
        save_plot(out_dir / "dataset_target_distribution.png")
    except Exception as exc:
        logger.warning("Could not plot dataset target distribution: %s", exc)

    try:
        miss = sampled.isna().mean().sort_values(ascending=False)
        top_miss_cols = miss.head(40).index.tolist()
        if len(top_miss_cols) >= 2:
            heat = sampled[top_miss_cols].isna().astype(int)
            if len(heat) > 300:
                heat = heat.sample(300, random_state=cfg.random_state)
            plt.figure(figsize=(12, 6))
            sns.heatmap(heat.T, cbar=True)
            plt.title("Missingness Heatmap (Top Columns)")
            plt.xlabel("Samples")
            plt.ylabel("Features")
            save_plot(out_dir / "dataset_missingness_heatmap.png")
    except Exception as exc:
        logger.warning("Could not plot missingness heatmap: %s", exc)

    try:
        numeric = sampled.select_dtypes(include=[np.number]).copy()
        if target_col in sampled.columns and task == "regression":
            numeric[target_col] = pd.to_numeric(y_sampled, errors="coerce")
        if numeric.shape[1] >= 2:
            var_order = numeric.var(numeric_only=True).sort_values(ascending=False).index.tolist()
            use_cols = var_order[: min(20, len(var_order))]
            corr = numeric[use_cols].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm", center=0)
            plt.title("Numeric Correlation Heatmap")
            save_plot(out_dir / "dataset_numeric_correlation_heatmap.png")
    except Exception as exc:
        logger.warning("Could not plot numeric correlation heatmap: %s", exc)
    try:
        numeric_cols = [c for c in sampled.columns if c != target_col and pd.api.types.is_numeric_dtype(sampled[c])]
        if len(numeric_cols) >= 2:
            top_k = min(5, max(2, int(cfg.pairplot_features)))

            if task == "regression":
                corr_ref = pd.concat([sampled[numeric_cols], pd.to_numeric(y_sampled, errors="coerce").rename(target_col)], axis=1)
                corr_vals = corr_ref.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore").abs().sort_values(ascending=False)
                pick = corr_vals.head(top_k).index.tolist()
                vars_pair = [target_col] + pick
                pair_df = sampled[pick].copy()
                pair_df[target_col] = pd.to_numeric(y_sampled, errors="coerce")

                hue_col = None
                cat_candidates = [
                    c
                    for c in sampled.columns
                    if c != target_col and c not in vars_pair and not pd.api.types.is_numeric_dtype(sampled[c])
                ]
                ranked: list[tuple[int, float, str]] = []
                for c in cat_candidates:
                    s = sampled[c].astype("string")
                    nun = int(s.nunique(dropna=True))
                    if 2 <= nun <= 6:
                        keep_ratio = float(s.notna().mean())
                        pref = 0 if c in cfg.include_columns else 1
                        ranked.append((pref, -keep_ratio, str(c)))
                if ranked:
                    ranked.sort()
                    hue_col = ranked[0][2]
                    pair_df[hue_col] = sampled[hue_col].astype("string").fillna("<NA>")
            else:
                var_order = sampled[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
                pick = var_order.head(top_k).index.tolist()
                vars_pair = pick
                pair_df = sampled[pick].copy()
                pair_df[target_col] = y_sampled.astype("string")
                hue_col = target_col if pair_df[target_col].nunique() <= 8 else None

            pair_n = min(len(pair_df), 500)
            if len(pair_df) > pair_n:
                pair_df = pair_df.sample(pair_n, random_state=cfg.random_state)

            if task == "regression":
                if hue_col is not None:
                    groups_arr = pair_df[hue_col].astype("string").to_numpy()
                    group_levels = list(pd.Series(groups_arr).dropna().astype("string").unique())[:3]
                    pal = sns.color_palette("tab10", n_colors=max(1, len(group_levels)))
                    group_color = {g: pal[i] for i, g in enumerate(group_levels)}
                    grid = sns.PairGrid(pair_df, vars=vars_pair, hue=hue_col, diag_sharey=False, palette="tab10")
                    grid.map_lower(sns.scatterplot, s=18, alpha=0.65)
                    grid.map_diag(sns.histplot, bins=25, kde=False, alpha=0.45, element="step", common_norm=False)
                else:
                    groups_arr = None
                    group_levels = []
                    group_color = {}
                    grid = sns.PairGrid(pair_df[vars_pair], vars=vars_pair, diag_sharey=False)
                    grid.map_lower(sns.scatterplot, s=18, alpha=0.65)
                    grid.map_diag(sns.histplot, bins=25, kde=False)

                def _annot_corr(x: np.ndarray, y_: np.ndarray, **kwargs: Any) -> None:
                    del kwargs
                    ax = plt.gca()
                    x_arr = np.asarray(x, dtype=float)
                    y_arr = np.asarray(y_, dtype=float)
                    m = np.isfinite(x_arr) & np.isfinite(y_arr)

                    lines: list[str] = []
                    colors: list[Any] = []
                    if int(np.sum(m)) < 3:
                        lines.append("all: r=nan ρ=nan")
                        colors.append("black")
                    else:
                        r_all = float(np.corrcoef(x_arr[m], y_arr[m])[0, 1])
                        rho_all = float(spearmanr(x_arr[m], y_arr[m])[0])
                        lines.append(f"all: r={r_all:.2f} ρ={rho_all:.2f}")
                        colors.append("black")

                    if groups_arr is not None and len(groups_arr) == len(x_arr):
                        for grp in group_levels:
                            gmask = m & (groups_arr == grp)
                            if int(np.sum(gmask)) < 3:
                                lines.append(f"{grp}: r=nan ρ=nan")
                            else:
                                r_g = float(np.corrcoef(x_arr[gmask], y_arr[gmask])[0, 1])
                                rho_g = float(spearmanr(x_arr[gmask], y_arr[gmask])[0])
                                lines.append(f"{grp}: r={r_g:.2f} ρ={rho_g:.2f}")
                            colors.append(group_color.get(grp, "tab:gray"))

                    ax.set_axis_off()
                    y0 = 0.84
                    for line, color in zip(lines, colors):
                        ax.text(0.03, y0, line, transform=ax.transAxes, ha="left", va="center", fontsize=9, fontweight="bold", color=color)
                        y0 -= 0.17

                grid.map_upper(_annot_corr)
                if hue_col is not None:
                    grid.add_legend(title=hue_col)
                grid.figure.suptitle("EDA Pairplot", y=1.02, fontsize=16, fontweight="bold")
                grid.figure.savefig(out_dir / "dataset_pairplot.png", dpi=300, bbox_inches="tight")
                plt.close(grid.figure)
            else:
                grid = sns.pairplot(pair_df, vars=pick, hue=hue_col, diag_kind="hist", corner=True)
                grid.figure.suptitle("EDA Pairplot", y=1.02, fontsize=16, fontweight="bold")
                grid.figure.savefig(out_dir / "dataset_pairplot.png", dpi=300, bbox_inches="tight")
                plt.close(grid.figure)
    except Exception as exc:
        logger.warning("Could not plot pairplot: %s", exc)
    try:
        feat = sampled.drop(columns=[target_col], errors="ignore")
        embed = _encode_mixed_frame_for_embedding(feat)
        if embed.shape[1] >= 2 and len(embed) >= 10:
            n_embed = min(len(embed), cfg.eda_sample_size)
            emb_sample = embed.sample(n_embed, random_state=cfg.random_state) if len(embed) > n_embed else embed
            y_emb = y_sampled.loc[emb_sample.index]

            pca = PCA(n_components=2, random_state=cfg.random_state)
            p = pca.fit_transform(emb_sample)
            plt.figure(figsize=(8, 6))
            if task == "regression":
                sc = plt.scatter(p[:, 0], p[:, 1], c=pd.to_numeric(y_emb, errors="coerce"), cmap="viridis", s=25, alpha=0.8)
                plt.colorbar(sc, label=target_col)
            else:
                for cls in y_emb.astype("string").unique():
                    m = y_emb.astype("string") == cls
                    plt.scatter(p[m, 0], p[m, 1], s=25, alpha=0.7, label=str(cls))
                plt.legend(loc="best")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Feature Space PCA")
            save_plot(out_dir / "dataset_feature_space_pca.png")

            n_tsne = min(len(emb_sample), cfg.tsne_sample_size)
            if n_tsne >= 40:
                tsne_sample = emb_sample.sample(n_tsne, random_state=cfg.random_state) if len(emb_sample) > n_tsne else emb_sample
                y_tsne = y_emb.loc[tsne_sample.index]
                perplexity = min(30.0, max(5.0, float((len(tsne_sample) - 1) / 3.0)))
                tsne = TSNE(n_components=2, random_state=cfg.random_state, init="pca", learning_rate="auto", perplexity=perplexity)
                t = tsne.fit_transform(tsne_sample)
                plt.figure(figsize=(8, 6))
                if task == "regression":
                    sc2 = plt.scatter(t[:, 0], t[:, 1], c=pd.to_numeric(y_tsne, errors="coerce"), cmap="viridis", s=25, alpha=0.8)
                    plt.colorbar(sc2, label=target_col)
                else:
                    for cls in y_tsne.astype("string").unique():
                        m = y_tsne.astype("string") == cls
                        plt.scatter(t[m, 0], t[m, 1], s=25, alpha=0.7, label=str(cls))
                    plt.legend(loc="best")
                plt.xlabel("t-SNE 1")
                plt.ylabel("t-SNE 2")
                plt.title("Feature Space t-SNE")
                save_plot(out_dir / "dataset_feature_space_tsne.png")
    except Exception as exc:
        logger.warning("Could not plot feature-space embeddings: %s", exc)


def plot_cv_score_distribution(model_label: str, cv_scores: np.ndarray | None, task: str, out_file: Path) -> None:
    if cv_scores is None:
        return
    arr = np.asarray(cv_scores, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return

    plt.figure(figsize=(7, 5))
    sns.histplot(arr[np.isfinite(arr)], bins=max(5, min(20, arr.size)), kde=True, color="tab:blue")
    plt.xlabel("CV score" if task == "classification" else "CV score (neg RMSE)")
    plt.ylabel("Count")
    plt.title(f"CV Score Distribution: {model_label}")
    save_plot(out_file)


def plot_residual_diagnostics(
    model_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_residual_scatter: Path,
    out_residual_qq: Path,
) -> dict[str, float]:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residual = y_t - y_p

    plt.figure(figsize=(8, 6))
    plt.scatter(y_p, residual, alpha=0.65, s=30)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residuals vs Predicted: {model_label}")
    save_plot(out_residual_scatter)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.histplot(residual, bins=30, kde=True, ax=axes[0], color="tab:orange")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual")

    (osm, osr), (slope, intercept, r) = probplot(residual, dist="norm")
    axes[1].scatter(osm, osr, s=18, alpha=0.7)
    axes[1].plot(osm, slope * np.asarray(osm) + intercept, "r--", linewidth=1.2)
    axes[1].set_title("QQ Plot of Residuals")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Ordered Residuals")
    fig.suptitle(f"Residual Diagnostics: {model_label}")
    fig.savefig(out_residual_qq, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_abs_mean": float(np.mean(np.abs(residual))),
    }


def plot_interval_width_vs_error(
    model_label: str,
    y_true: np.ndarray,
    intervals: dict[float, dict[str, np.ndarray]],
    out_file: Path,
) -> dict[str, float]:
    if not intervals:
        return {"width_error_spearman": float("nan")}

    alphas = sorted(intervals.keys())
    alpha = float(alphas[np.argmin(np.abs(np.asarray(alphas) - 0.05))])
    bundle = intervals[alpha]

    y_t = np.asarray(y_true, dtype=float)
    pred = np.asarray(bundle["pred"], dtype=float)
    lower = np.asarray(bundle["lower"], dtype=float)
    upper = np.asarray(bundle["upper"], dtype=float)

    width = upper - lower
    abs_err = np.abs(y_t - pred)

    corr, _ = spearmanr(width, abs_err)

    plt.figure(figsize=(8, 6))
    plt.scatter(width, abs_err, alpha=0.65, s=28)
    plt.xlabel("Interval Width")
    plt.ylabel("Absolute Error")
    plt.title(f"Interval Width vs Absolute Error ({int((1-alpha)*100)}% CI): {model_label}")
    plt.text(
        0.02,
        0.98,
        f"Spearman={corr:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    save_plot(out_file)

    return {"width_error_spearman": float(corr)}
def plot_learning_curve(lc_df: pd.DataFrame, model_label: str, metric_label: str, out_file: Path) -> None:
    if lc_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(lc_df["train_size"], lc_df["train_mean"], marker="o", color="blue", label=f"Train {metric_label}")
    plt.plot(lc_df["train_size"], lc_df["cv_mean"], marker="o", color="orange", label=f"CV {metric_label}")

    plt.fill_between(
        lc_df["train_size"],
        lc_df["cv_mean"] - lc_df["cv_std"],
        lc_df["cv_mean"] + lc_df["cv_std"],
        alpha=0.2,
        color="orange",
    )

    plt.xlabel("Number of Training Samples")
    plt.ylabel(metric_label)
    plt.title(f"Learning Curve: {model_label}")
    plt.legend()
    save_plot(out_file)


def plot_actual_vs_predicted_with_marginals(
    model_label: str,
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    out_file: Path,
) -> dict[str, float]:
    train_metrics = evaluate_regression_metrics(y_train, y_train_pred)
    test_metrics = evaluate_regression_metrics(y_test, y_test_pred)

    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(4, 4)
    ax_histx = fig.add_subplot(grid[0, :3])
    ax_scatter = fig.add_subplot(grid[1:, :3])
    ax_histy = fig.add_subplot(grid[1:, 3])

    ax_scatter.scatter(y_train, y_train_pred, c="blue", alpha=0.75, edgecolors="white", linewidth=0.5, label="Train")
    ax_scatter.scatter(y_test, y_test_pred, c="red", alpha=0.7, edgecolors="white", linewidth=0.5, label="Test")

    low = float(min(np.min(y_train), np.min(y_test), np.min(y_train_pred), np.min(y_test_pred)))
    high = float(max(np.max(y_train), np.max(y_test), np.max(y_train_pred), np.max(y_test_pred)))
    ax_scatter.plot([low, high], [low, high], "k--", linewidth=1.5)

    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.legend(loc="upper left")

    metrics_text = (
        f"Train R2={train_metrics['r2']:.2f}, RMSE={train_metrics['rmse']:.2f}\n"
        f"Test R2={test_metrics['r2']:.2f}, RMSE={test_metrics['rmse']:.2f}\n"
        f"Train N={len(y_train)}, Test N={len(y_test)}"
    )
    ax_scatter.text(
        0.98,
        0.03,
        metrics_text,
        transform=ax_scatter.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    sns.histplot(x=np.asarray(y_train, dtype=float), bins=20, kde=True, ax=ax_histx, color="blue", alpha=0.35)
    sns.histplot(x=np.asarray(y_test, dtype=float), bins=20, kde=True, ax=ax_histx, color="red", alpha=0.35)
    ax_histx.set_ylabel("Count")
    ax_histx.tick_params(axis="x", labelbottom=False)

    sns.histplot(y=np.asarray(y_train_pred, dtype=float), bins=20, kde=True, ax=ax_histy, color="blue", alpha=0.35)
    sns.histplot(y=np.asarray(y_test_pred, dtype=float), bins=20, kde=True, ax=ax_histy, color="red", alpha=0.35)
    ax_histy.set_xlabel("Count")
    ax_histy.tick_params(axis="y", labelleft=False)

    fig.suptitle(f"Actual vs Predicted - {model_label}", y=0.98)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "train_r2": float(train_metrics["r2"]),
        "train_rmse": float(train_metrics["rmse"]),
        "test_r2": float(test_metrics["r2"]),
        "test_rmse": float(test_metrics["rmse"]),
    }


def compute_regression_calibration(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 12) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)

    if int(np.sum(finite_mask)) == 0:
        return {
            "bin_pred": np.array([], dtype=float),
            "bin_true": np.array([], dtype=float),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "fit_available": False,
            "status": "no_finite_pairs",
        }

    y_true_clean = y_true_arr[finite_mask]
    y_pred_clean = y_pred_arr[finite_mask]

    order = np.argsort(y_pred_clean)
    y_true_sorted = y_true_clean[order]
    y_pred_sorted = y_pred_clean[order]

    splits = np.array_split(np.arange(len(y_true_sorted)), min(bins, len(y_true_sorted)))
    bin_pred = np.array([np.mean(y_pred_sorted[idx]) for idx in splits if len(idx) > 0], dtype=float)
    bin_true = np.array([np.mean(y_true_sorted[idx]) for idx in splits if len(idx) > 0], dtype=float)

    if len(bin_pred) < 2 or len(np.unique(np.round(bin_pred, 12))) < 2:
        return {
            "bin_pred": bin_pred,
            "bin_true": bin_true,
            "slope": float("nan"),
            "intercept": float(np.nanmean(bin_true)) if len(bin_true) > 0 else float("nan"),
            "r2": float("nan"),
            "fit_available": False,
            "status": "constant_or_insufficient_predictions",
        }

    try:
        slope, intercept, r_value, _, _ = linregress(bin_pred, bin_true)
        return {
            "bin_pred": bin_pred,
            "bin_true": bin_true,
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r_value**2),
            "fit_available": True,
            "status": "ok",
        }
    except Exception:
        return {
            "bin_pred": bin_pred,
            "bin_true": bin_true,
            "slope": float("nan"),
            "intercept": float(np.nanmean(bin_true)) if len(bin_true) > 0 else float("nan"),
            "r2": float("nan"),
            "fit_available": False,
            "status": "linregress_failed",
        }


def plot_regression_calibration(model_label: str, calib: dict[str, Any], out_file: Path) -> None:
    bin_pred = np.asarray(calib["bin_pred"])
    bin_true = np.asarray(calib["bin_true"])

    if len(bin_pred) == 0:
        return

    low = min(float(np.min(bin_pred)), float(np.min(bin_true)))
    high = max(float(np.max(bin_pred)), float(np.max(bin_true)))

    plt.figure(figsize=(8, 6))
    plt.plot(bin_pred, bin_true, "o-", color="orange", label="Calibration")
    plt.plot([low, high], [low, high], "--", color="gray", label="Ideal")

    fit_available = bool(calib.get("fit_available", False))
    if fit_available and np.isfinite(calib.get("slope", np.nan)) and np.isfinite(calib.get("intercept", np.nan)):
        fit_line = calib["slope"] * bin_pred + calib["intercept"]
        plt.plot(bin_pred, fit_line, "--", color="red", label=f"Fit: y={calib['slope']:.2f}x+{calib['intercept']:.2f}, R2={calib['r2']:.2f}")
        txt = f"Slope={calib['slope']:.2f}\nIntercept={calib['intercept']:.2f}\nR2={calib['r2']:.2f}"
    else:
        txt = f"Fit unavailable\nReason: {calib.get('status', 'unknown')}"

    plt.xlabel("Predicted")
    plt.ylabel("Average True")
    plt.title(f"Calibration: {model_label}")
    plt.legend(loc="lower right")
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    save_plot(out_file)



def assess_regression_calibration_need(
    y_calib: np.ndarray,
    y_calib_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    improvement_threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    raw_metrics = evaluate_regression_metrics(pd.Series(y_test), np.asarray(y_test_pred))

    finite_mask = np.isfinite(y_calib) & np.isfinite(y_calib_pred)
    y_cal = np.asarray(y_calib)[finite_mask]
    y_cal_pred_clean = np.asarray(y_calib_pred)[finite_mask]

    calibrated_pred = np.asarray(y_test_pred, dtype=float)
    status = "unavailable"

    if len(y_cal) >= 3 and len(np.unique(np.round(y_cal_pred_clean, 12))) >= 2:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_cal_pred_clean, y_cal)
            calibrated_pred = iso.transform(np.asarray(y_test_pred, dtype=float))
            status = "ok"
        except Exception:
            status = "fit_failed"

    iso_metrics = evaluate_regression_metrics(pd.Series(y_test), calibrated_pred)

    cal_raw = compute_regression_calibration(np.asarray(y_test), np.asarray(y_test_pred), bins=12)
    cal_iso = compute_regression_calibration(np.asarray(y_test), np.asarray(calibrated_pred), bins=12)

    rmse_before = float(raw_metrics.get("rmse", np.nan))
    rmse_after = float(iso_metrics.get("rmse", np.nan))
    mae_before = float(raw_metrics.get("mae", np.nan))
    mae_after = float(iso_metrics.get("mae", np.nan))

    rmse_gain = float((rmse_before - rmse_after) / (abs(rmse_before) + 1e-12))
    mae_gain = float((mae_before - mae_after) / (abs(mae_before) + 1e-12))

    slope_before = float(cal_raw.get("slope", np.nan))
    slope_after = float(cal_iso.get("slope", np.nan))
    slope_err_before = abs(1.0 - slope_before) if np.isfinite(slope_before) else float("nan")
    slope_err_after = abs(1.0 - slope_after) if np.isfinite(slope_after) else float("nan")

    slope_improved = bool(
        np.isfinite(slope_err_before)
        and np.isfinite(slope_err_after)
        and (slope_err_after + 0.02 < slope_err_before)
    )

    calibration_needed = bool(
        status == "ok"
        and (
            rmse_gain >= improvement_threshold
            or mae_gain >= improvement_threshold
            or slope_improved
        )
    )

    decision = {
        "status": status,
        "calibration_needed": int(calibration_needed),
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "rmse_relative_gain": rmse_gain,
        "mae_before": mae_before,
        "mae_after": mae_after,
        "mae_relative_gain": mae_gain,
        "r2_before": float(raw_metrics.get("r2", np.nan)),
        "r2_after": float(iso_metrics.get("r2", np.nan)),
        "slope_before": slope_before,
        "slope_after": slope_after,
        "slope_error_before": slope_err_before,
        "slope_error_after": slope_err_after,
        "improvement_threshold": float(improvement_threshold),
    }
    return decision, calibrated_pred


def plot_isotonic_calibration_check(
    model_label: str,
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_iso: np.ndarray,
    decision: dict[str, Any],
    out_file: Path,
) -> None:
    plt.figure(figsize=(8, 7))
    low = float(min(np.min(y_true), np.min(y_pred_raw), np.min(y_pred_iso)))
    high = float(max(np.max(y_true), np.max(y_pred_raw), np.max(y_pred_iso)))

    plt.scatter(y_true, y_pred_raw, alpha=0.55, s=30, label="Raw prediction")
    plt.scatter(y_true, y_pred_iso, alpha=0.55, s=30, label="Isotonic calibrated")
    plt.plot([low, high], [low, high], "k--", linewidth=1.4, label="Ideal")

    needed = bool(decision.get("calibration_needed", 0))
    txt = (
        f"Need calibration: {needed}\n"
        f"RMSE gain: {decision.get('rmse_relative_gain', np.nan):.3f}\n"
        f"MAE gain: {decision.get('mae_relative_gain', np.nan):.3f}\n"
        f"R2 before/after: {decision.get('r2_before', np.nan):.3f} / {decision.get('r2_after', np.nan):.3f}"
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Isotonic Calibration Need Check: {model_label}")
    plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.legend(loc="lower right")
    save_plot(out_file)


def compute_uncertainty_reliability_by_target_bins(
    y_true: np.ndarray,
    intervals: dict[float, dict[str, np.ndarray]],
    reference_alpha: float = 0.05,
    n_bins: int = 4,
) -> pd.DataFrame:
    if not intervals:
        return pd.DataFrame()

    available = np.asarray(sorted(intervals.keys()), dtype=float)
    alpha = float(available[np.argmin(np.abs(available - reference_alpha))])
    bundle = intervals[alpha]

    y_arr = np.asarray(y_true, dtype=float)
    lower = np.asarray(bundle["lower"], dtype=float)
    upper = np.asarray(bundle["upper"], dtype=float)
    covered = np.asarray(bundle["covered"], dtype=bool)

    q_count = min(max(2, int(n_bins)), max(2, min(10, int(np.unique(y_arr).shape[0]))))
    bins = pd.qcut(y_arr, q=q_count, duplicates="drop")
    bin_series = pd.Series(bins).astype("string")
    width = upper - lower

    rows: list[dict[str, Any]] = []
    for name in bin_series.dropna().unique():
        idx = np.where(bin_series.to_numpy() == name)[0]
        if len(idx) == 0:
            continue
        cov = float(np.mean(covered[idx]))
        avg_width = float(np.mean(width[idx]))
        rows.append(
            {
                "target_bin": str(name),
                "alpha": alpha,
                "nominal_coverage": float(1.0 - alpha),
                "observed_coverage": cov,
                "miscoverage": float(1.0 - cov),
                "avg_width": avg_width,
                "efficiency": float(1.0 / (1.0 + avg_width)),
                "n_samples": int(len(idx)),
            }
        )

    return pd.DataFrame(rows)


def plot_uncertainty_reliability_by_bin(model_label: str, frame: pd.DataFrame, out_file: Path) -> None:
    if frame.empty:
        return

    x = np.arange(len(frame))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bars = ax1.bar(x - 0.15, frame["observed_coverage"], width=0.3, color="tab:blue", label="Observed coverage")
    ax1.plot(x, frame["nominal_coverage"], "k--", linewidth=1.4, label="Nominal coverage")
    ax2.bar(x + 0.15, frame["avg_width"], width=0.3, color="tab:orange", alpha=0.8, label="Avg interval width")

    ax1.set_xticks(x)
    ax1.set_xticklabels(frame["target_bin"].tolist())
    ax1.set_ylabel("Coverage")
    ax2.set_ylabel("Interval Width")
    ax1.set_title(f"Uncertainty Reliability by Target Bin: {model_label}")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    save_plot(out_file)


def run_permutation_stability(
    model_label: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cfg: RunConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    scorer = "neg_root_mean_squared_error" if task == "regression" else "f1_weighted"

    if len(X) < max(30, cfg.importance_stability_folds * 5):
        return pd.DataFrame()

    repeat_vectors: list[np.ndarray] = []

    for rep in range(cfg.importance_stability_repeats):
        cv = KFold(n_splits=max(2, cfg.importance_stability_folds), shuffle=True, random_state=cfg.random_state + rep * 31)
        fold_vectors: list[np.ndarray] = []

        for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            y_va = y.iloc[va_idx]

            try:
                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)
                perm = permutation_importance(
                    fold_model,
                    X_va,
                    y_va,
                    scoring=scorer,
                    n_repeats=5,
                    random_state=cfg.random_state + rep * 101 + fold_id,
                    n_jobs=1,
                )
                fold_vectors.append(np.asarray(perm.importances_mean, dtype=float))
            except Exception as exc:
                logger.warning("Permutation stability fold failed for %s (rep %d fold %d): %s", model_label, rep + 1, fold_id, exc)

        if fold_vectors:
            repeat_vectors.append(np.mean(np.vstack(fold_vectors), axis=0))

    if len(repeat_vectors) < 2:
        return pd.DataFrame()

    arr = np.vstack(repeat_vectors)
    mean_imp = np.mean(arr, axis=0)
    std_imp = np.std(arr, axis=0)

    feature_names = X.columns.astype(str)
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": mean_imp,
            "importance_std": std_imp,
        }
    ).sort_values("importance_mean", ascending=False)

    corrs: list[float] = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            corr, _ = spearmanr(arr[i], arr[j])
            if np.isfinite(corr):
                corrs.append(float(corr))

    summary = pd.DataFrame(
        [
            {
                "model": model_label,
                "repeats": int(len(arr)),
                "mean_pairwise_spearman": float(np.mean(corrs)) if corrs else float("nan"),
                "std_pairwise_spearman": float(np.std(corrs)) if corrs else float("nan"),
            }
        ]
    )

    frame.to_csv(out_dir / "permutation_stability.csv", index=False)
    summary.to_csv(out_dir / "permutation_stability_summary.csv", index=False)

    top = frame.head(20).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="tab:purple", alpha=0.8)
    plt.xlabel("Permutation Importance (mean ± std)")
    plt.title(f"Permutation Importance Stability: {model_label}")
    save_plot(out_dir / "permutation_stability_top20.png")

    return frame

def _default_coverage_alpha_grid() -> list[float]:
    return [float(np.round(v, 2)) for v in np.arange(0.05, 1.0001, 0.05)]


def _merge_alpha_grid(alphas: list[float]) -> list[float]:
    merged: list[float] = []
    for raw in list(alphas) + _default_coverage_alpha_grid():
        try:
            a = float(raw)
        except Exception:
            continue
        if not (0.0 < a <= 1.0):
            continue
        merged.append(float(np.round(a, 6)))
    uniq = sorted(set(merged))
    return uniq if uniq else _default_coverage_alpha_grid()


def _compute_local_conformal_scales(
    model: Pipeline,
    X_calib: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        pipe = _unwrap_pipeline(model)
        if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
            pre = pipe.named_steps["preprocessor"]
            calib_repr, test_repr = reduce_matrix_pair(
                pre.transform(X_calib),
                pre.transform(X_test),
                random_state=42,
                max_dim=30,
            )
        else:
            calib_repr = _encode_mixed_frame_for_embedding(X_calib).to_numpy(dtype=float)
            test_repr = _encode_mixed_frame_for_embedding(X_test).to_numpy(dtype=float)

        n_cal = int(len(calib_repr))
        if n_cal < 4:
            return None, None

        from sklearn.neighbors import NearestNeighbors

        n_neighbors = min(8, n_cal)
        if n_neighbors < 2:
            return None, None

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(calib_repr)

        d_calib, _ = knn.kneighbors(calib_repr, n_neighbors=n_neighbors)
        calib_scale = np.mean(d_calib[:, 1:], axis=1) if n_neighbors > 1 else np.ravel(d_calib)

        d_test, _ = knn.kneighbors(test_repr, n_neighbors=min(n_neighbors, n_cal))
        test_scale = np.mean(d_test, axis=1)

        valid = np.isfinite(calib_scale) & (calib_scale > 0)
        if not np.any(valid):
            return None, None

        base = float(np.nanmedian(calib_scale[valid]))
        if not np.isfinite(base) or base <= 0:
            base = 1.0

        floor = max(base * 0.1, 1e-6)
        calib_scale = np.where(np.isfinite(calib_scale) & (calib_scale > 0), calib_scale, base)
        test_scale = np.where(np.isfinite(test_scale) & (test_scale > 0), test_scale, base)

        calib_scale = np.maximum(calib_scale, floor)
        test_scale = np.maximum(test_scale, floor)

        return calib_scale.astype(float), test_scale.astype(float)
    except Exception:
        return None, None


def compute_conformal_intervals(
    model: Pipeline,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alphas: list[float],
) -> tuple[pd.DataFrame, dict[float, dict[str, np.ndarray]]]:
    if len(X_calib) == 0 or len(X_test) == 0:
        raise RuntimeError("Calibration and test partitions must be non-empty for conformal intervals.")
    if len(X_calib) != len(y_calib) or len(X_test) != len(y_test):
        raise RuntimeError("X/y mismatch detected before conformal interval computation.")

    y_cal_pred = model.predict(X_calib)
    y_test_pred = model.predict(X_test)
    if len(y_cal_pred) != len(y_calib) or len(y_test_pred) != len(y_test):
        raise RuntimeError("Prediction length mismatch during conformal interval computation.")

    y_cal_arr = np.asarray(y_calib, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    y_cal_pred_arr = np.asarray(y_cal_pred, dtype=float)
    y_test_pred_arr = np.asarray(y_test_pred, dtype=float)

    residuals_raw = np.abs(y_cal_arr - y_cal_pred_arr)
    if np.isfinite(residuals_raw).sum() == 0:
        raise RuntimeError("No finite calibration residuals available for conformal interval computation.")

    if not np.isfinite(y_test_arr).all():
        raise RuntimeError("Test target contains non-finite values for conformal interval computation.")
    y_range = float(np.max(y_test_arr) - np.min(y_test_arr) + 1e-12)

    calib_scale, test_scale = _compute_local_conformal_scales(model, X_calib, X_test)
    use_adaptive = (
        calib_scale is not None
        and test_scale is not None
        and len(calib_scale) == len(residuals_raw)
        and len(test_scale) == len(y_test_pred_arr)
        and float(np.nanstd(test_scale)) > 1e-12
    )

    if use_adaptive:
        valid = np.isfinite(residuals_raw) & np.isfinite(calib_scale) & (calib_scale > 0)
        scores = residuals_raw[valid] / calib_scale[valid]
        interval_mode = "adaptive_local_conformal"
    else:
        valid = np.isfinite(residuals_raw)
        scores = residuals_raw[valid]
        interval_mode = "split_conformal"

    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise RuntimeError("No finite conformal scores available for uncertainty interval computation.")

    rows: list[dict[str, float | str]] = []
    intervals: dict[float, dict[str, np.ndarray]] = {}

    for alpha in _merge_alpha_grid(alphas):
        qhat = conformal_quantile(scores, alpha)

        if use_adaptive:
            radius = np.asarray(test_scale, dtype=float) * float(qhat)
            lower = y_test_pred_arr - radius
            upper = y_test_pred_arr + radius
        else:
            lower = y_test_pred_arr - float(qhat)
            upper = y_test_pred_arr + float(qhat)

        covered = (y_test_arr >= lower) & (y_test_arr <= upper)
        coverage = float(np.mean(covered))
        width = upper - lower
        avg_width = float(np.mean(width))
        pinaw = float(avg_width / y_range)
        efficiency = float(1.0 / (1.0 + pinaw))

        rows.append(
            {
                "alpha": float(alpha),
                "nominal_coverage": float(1.0 - alpha),
                "observed_coverage": float(coverage),
                "miscoverage": float(1.0 - coverage),
                "qhat": float(qhat),
                "avg_width": avg_width,
                "pinaw": pinaw,
                "efficiency": efficiency,
                "interval_mode": interval_mode,
            }
        )

        intervals[float(alpha)] = {
            "pred": np.asarray(y_test_pred_arr),
            "lower": np.asarray(lower),
            "upper": np.asarray(upper),
            "covered": np.asarray(covered),
        }

    frame = pd.DataFrame(rows).sort_values("alpha")
    return frame, intervals
def plot_conformal_coverage_curve(model_label: str, conf_df: pd.DataFrame, out_file: Path) -> dict[str, float]:
    if conf_df.empty:
        return {"ace": float("nan"), "overconfidence": float("nan"), "underconfidence": float("nan")}

    alpha = conf_df["alpha"].to_numpy()
    observed_misc = conf_df["miscoverage"].to_numpy()
    ideal = alpha

    ace = float(np.mean(np.abs(observed_misc - ideal)))
    over = float(np.mean(np.maximum(observed_misc - ideal, 0)))
    under = float(np.mean(np.maximum(ideal - observed_misc, 0)))

    plt.figure(figsize=(8, 6))
    plt.plot(alpha, observed_misc, marker="o", label="Observed")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal (miscov = alpha)")
    plt.xlabel("Significance Level (alpha)")
    plt.ylabel("Miscoverage (1 - coverage)")
    plt.title(f"Coverage Curve: {model_label}")
    plt.legend()
    save_plot(out_file)

    return {"ace": ace, "overconfidence": over, "underconfidence": under}


def plot_ci_scatter(
    model_label: str,
    alpha: float,
    y_true: np.ndarray,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    out_file: Path,
) -> None:
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = upper - lower
    pinaw = float(np.mean(width) / (np.max(y_true) - np.min(y_true) + 1e-12))
    efficiency = float(1.0 / (1.0 + pinaw))

    order = np.argsort(y_true)
    y_ord = y_true[order]
    p_ord = pred[order]
    l_ord = lower[order]
    u_ord = upper[order]

    plt.figure(figsize=(9, 7))
    plt.plot([float(np.min(y_true)), float(np.max(y_true))], [float(np.min(y_true)), float(np.max(y_true))], "k--", label="Ideal")
    plt.scatter(y_ord, y_ord, c="red", s=35, alpha=0.7, label="True Values")
    plt.errorbar(y_ord, p_ord, yerr=[p_ord - l_ord, u_ord - p_ord], fmt="o", color="green", ecolor="gray", alpha=0.6, label=f"{int((1-alpha)*100)}% CI")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{int((1-alpha)*100)}% CI: {model_label}\nCoverage={coverage:.2f}, Efficiency={efficiency:.2f}")
    plt.legend()
    save_plot(out_file)




def plot_multi_ci_scatter(
    model_label: str,
    y_true: np.ndarray,
    intervals: dict[float, dict[str, np.ndarray]],
    out_file: Path,
    max_levels: int = 3,
) -> None:
    if not intervals:
        return

    sorted_alpha = sorted(intervals.keys())[: max(1, int(max_levels))]
    y_arr = np.asarray(y_true, dtype=float)

    plt.figure(figsize=(9, 7))
    plt.plot([float(np.min(y_arr)), float(np.max(y_arr))], [float(np.min(y_arr)), float(np.max(y_arr))], "k--", label="Ideal")
    plt.scatter(y_arr, y_arr, c="black", s=18, alpha=0.5, label="True")

    palette = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
    for idx_alpha, alpha in enumerate(sorted_alpha):
        bundle = intervals[alpha]
        pred = np.asarray(bundle["pred"], dtype=float)
        lower = np.asarray(bundle["lower"], dtype=float)
        upper = np.asarray(bundle["upper"], dtype=float)
        color = palette[idx_alpha % len(palette)]
        label = f"{int((1-alpha)*100)}% CI"
        plt.errorbar(
            y_arr,
            pred,
            yerr=[pred - lower, upper - pred],
            fmt="o",
            color=color,
            ecolor=color,
            alpha=0.25 + 0.2 * idx_alpha,
            markersize=4,
            label=label,
        )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Multi-level Confidence Intervals: {model_label}")
    plt.legend(loc="best")
    save_plot(out_file)
def plot_feature_correlation_heatmap(
    X: pd.DataFrame,
    y: pd.Series,
    model_label: str,
    out_file: Path,
    top_n: int = 20,
) -> None:
    numeric = X.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        return

    y_num = pd.to_numeric(y, errors="coerce")
    corr_to_target = numeric.corrwith(y_num).abs().sort_values(ascending=False)
    selected = corr_to_target.head(min(top_n, len(corr_to_target))).index.tolist()
    data = numeric[selected].copy()
    data["target"] = y_num

    corr = data.corr().fillna(0)

    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title(f"{model_label} Top-{min(top_n, len(selected))} Feature Correlation (incl. target)")
    save_plot(out_file)


def compute_mahalanobis(train_repr: np.ndarray, test_repr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    train_repr = _as_2d_dense(train_repr)
    test_repr = _as_2d_dense(test_repr)
    mu = np.mean(train_repr, axis=0)
    cov = np.atleast_2d(np.cov(train_repr, rowvar=False))
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(cov.shape[0]) * 1e-8
    inv_cov = np.linalg.pinv(cov)

    def distances(arr: np.ndarray) -> np.ndarray:
        diff = arr - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))

    train_dist = distances(train_repr)
    test_dist = distances(test_repr)
    threshold = float(np.quantile(train_dist, 0.95))
    return train_dist, test_dist, threshold


def plot_mahalanobis_distribution(model_label: str, distances: np.ndarray, threshold: float, out_file: Path) -> None:
    plt.figure(figsize=(10, 7))
    plt.hist(distances, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"95th percentile = {threshold:.2f}")
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Frequency")
    plt.title(f"Mahalanobis Distance Distribution: {model_label}")
    plt.legend()
    save_plot(out_file)


def plot_williams(
    model_label: str,
    leverage: np.ndarray,
    standardized_residuals: np.ndarray,
    h_star: float,
    out_file: Path,
) -> np.ndarray:
    outlier_mask = (np.abs(standardized_residuals) > 3.0) | (leverage > h_star)

    plt.figure(figsize=(10, 8))
    plt.scatter(leverage, standardized_residuals, alpha=0.6, s=70, label="Points", color="tab:blue")
    plt.scatter(
        leverage[outlier_mask],
        standardized_residuals[outlier_mask],
        facecolors="none",
        edgecolors="red",
        s=200,
        linewidth=1.8,
        label="Outliers/High leverage",
    )

    plt.axhline(3.0, linestyle="--", color="gray")
    plt.axhline(-3.0, linestyle="--", color="gray")
    plt.axvline(h_star, linestyle="--", color="gray", label=f"h* = {h_star:.3f}")
    plt.xlabel("Leverage")
    plt.ylabel("Standardized Residuals")
    plt.title(f"Williams Plot: {model_label}")
    plt.legend()
    save_plot(out_file)

    return outlier_mask


def run_outlier_analysis(
    model_label: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    random_state: int,
    out_dir: Path,
) -> pd.DataFrame:
    pipe = _unwrap_pipeline(model)
    pre: ColumnTransformer = pipe.named_steps["preprocessor"]
    train_repr, test_repr = reduce_matrix_pair(
        pre.transform(X_train),
        pre.transform(X_test),
        random_state=random_state,
        max_dim=30,
    )

    train_dist, test_dist, threshold = compute_mahalanobis(train_repr, test_repr)
    plot_mahalanobis_distribution(model_label, test_dist, threshold, out_dir / "mahalanobis_distribution.png")

    x_train = np.column_stack([np.ones(len(train_repr)), train_repr])
    x_test = np.column_stack([np.ones(len(test_repr)), test_repr])
    xtx_inv = np.linalg.pinv(x_train.T @ x_train)

    leverage_test = np.einsum("ij,jk,ik->i", x_test, xtx_inv, x_test)
    p = x_train.shape[1] - 1
    h_star = float(3.0 * (p + 1) / len(x_train))

    train_resid_std = np.std(np.asarray(y_train) - np.asarray(y_train_pred)) + 1e-12
    standardized_resid = (np.asarray(y_test) - np.asarray(y_test_pred)) / train_resid_std

    outlier_mask = plot_williams(
        model_label,
        leverage_test,
        standardized_resid,
        h_star,
        out_dir / "williams_plot.png",
    )

    frame = pd.DataFrame(
        {
            "mahalanobis_distance": test_dist,
            "mahalanobis_outlier": (test_dist > threshold).astype(int),
            "leverage": leverage_test,
            "standardized_residual": standardized_resid,
            "williams_outlier": outlier_mask.astype(int),
        },
        index=X_test.index,
    )
    frame.to_csv(out_dir / "outlier_analysis.csv", index=True)
    return frame


def run_feature_importance_and_shap(
    model_label: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    cfg: RunConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    out_dir = ensure_dir(Path(out_dir).resolve())
    scorer = "neg_root_mean_squared_error" if task == "regression" else "f1_weighted"

    try:
        perm = permutation_importance(model, X_test, y_test, scoring=scorer, n_repeats=10, random_state=cfg.random_state, n_jobs=max(1, int(cfg.n_jobs)))
        perm_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(out_dir / "permutation_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        top = perm_df.head(20).iloc[::-1]
        sns.barplot(data=top, x="importance_mean", y="feature", orient="h")
        plt.title(f"Permutation Importance: {model_label}")
        save_plot(out_dir / "permutation_importance.png")
    except Exception as exc:
        logger.warning("Permutation importance failed for %s: %s", model_label, exc)

    try:
        import shap
    except Exception:
        return

    pipe = _unwrap_pipeline(model)
    pre = pipe.named_steps["preprocessor"]
    estimator = pipe.named_steps["model"]

    if estimator.__class__.__name__.lower().startswith("dummy"):
        logger.info("Skipping SHAP for %s (dummy baseline model).", model_label)
        return

    sample_n = min(cfg.shap_sample_size, len(X_test))

    sample = X_test.sample(sample_n, random_state=cfg.random_state)
    transformed = pre.transform(sample)
    if sparse.issparse(transformed):
        density_shape = transformed.shape
        if density_shape[1] > cfg.max_shap_features:
            logger.info("Skipping SHAP for %s due to feature dimensionality guard.", model_label)
            return
        transformed = transformed.toarray()
    else:
        if transformed.shape[1] > cfg.max_shap_features:
            logger.info("Skipping SHAP for %s due to feature dimensionality guard.", model_label)
            return

    feature_names = get_transformed_feature_names(model, sample.columns.tolist())
    explain_df = pd.DataFrame(transformed, columns=feature_names)

    try:
        explainer = shap.Explainer(estimator, transformed)
        shap_values = explainer(transformed)
    except Exception:
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(transformed)
        except Exception as exc:
            logger.warning("SHAP explainer failed for %s: %s", model_label, exc)
            return

    try:
        shap.summary_plot(shap_values, explain_df, show=False, max_display=20)
        save_plot(out_dir / "shap_beeswarm.png")
        shap.summary_plot(shap_values, explain_df, show=False, plot_type="bar", max_display=20)
        save_plot(out_dir / "shap_bar.png")
    except Exception as exc:
        logger.warning("SHAP plotting failed for %s: %s", model_label, exc)


def build_uncertainty_summary_for_model(
    model_key: str,
    model_label: str,
    test_metrics: dict[str, float],
    conf_df: pd.DataFrame,
    bootstrap_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    row = {
        "model_key": model_key,
        "model": model_label,
        "test_r2": float(test_metrics.get("r2", np.nan)),
        "test_rmse": float(test_metrics.get("rmse", np.nan)),
    }

    conf_local = conf_df.copy()
    conf_local["alpha"] = pd.to_numeric(conf_local.get("alpha"), errors="coerce")
    conf_local = conf_local.dropna(subset=["alpha"])
    alpha95 = conf_local.loc[np.isclose(conf_local["alpha"], 0.05)]

    if alpha95.empty and not conf_local.empty:
        nearest_idx = (conf_local["alpha"] - 0.05).abs().idxmin()
        alpha95 = conf_local.loc[[nearest_idx]].copy()

    if not alpha95.empty:
        row["coverage_95"] = float(pd.to_numeric(alpha95["observed_coverage"], errors="coerce").iloc[0])
        row["efficiency_95"] = float(pd.to_numeric(alpha95["efficiency"], errors="coerce").iloc[0])
        row["interval_width_95"] = float(pd.to_numeric(alpha95["avg_width"], errors="coerce").iloc[0]) if "avg_width" in alpha95.columns else float("nan")
        row["pinaw_95"] = float(pd.to_numeric(alpha95["pinaw"], errors="coerce").iloc[0]) if "pinaw" in alpha95.columns else float("nan")
    else:
        row["coverage_95"] = float("nan")
        row["efficiency_95"] = float("nan")
        row["interval_width_95"] = float("nan")
        row["pinaw_95"] = float("nan")

    row_df = pd.DataFrame([row])

    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(bootstrap_repeats):
        sampled = row_df.sample(n=1, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        stats.append(sampled.iloc[0].to_dict())
    boot = pd.DataFrame(stats)

    summary = pd.DataFrame(
        [
            {
                "model_key": model_key,
                "model": model_label,
                "r2_mean": float(boot["test_r2"].mean()),
                "r2_std": float(boot["test_r2"].std()),
                "coverage_mean": float(boot["coverage_95"].mean()),
                "coverage_std": float(boot["coverage_95"].std()),
                "rmse_mean": float(boot["test_rmse"].mean()),
                "rmse_std": float(boot["test_rmse"].std()),
                "efficiency_mean": float(boot["efficiency_95"].mean()),
                "efficiency_std": float(boot["efficiency_95"].std()),
                "interval_width_mean": float(boot["interval_width_95"].mean()),
                "interval_width_std": float(boot["interval_width_95"].std()),
                "pinaw_mean": float(boot["pinaw_95"].mean()),
                "pinaw_std": float(boot["pinaw_95"].std()),
            }
        ]
    )
    return summary


def plot_global_uncertainty_comparison(summary_df: pd.DataFrame, out_file: Path) -> None:
    if summary_df.empty:
        return

    labels = summary_df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Prefer 95% interval width (target units) so it is directly comparable to RMSE.
    # Fall back to dimensionless efficiency only for backward compatibility with older outputs.
    if {"interval_width_mean", "interval_width_std"}.issubset(summary_df.columns):
        eff_plot = pd.to_numeric(summary_df["interval_width_mean"], errors="coerce")
        eff_std_plot = pd.to_numeric(summary_df["interval_width_std"], errors="coerce")
        eff_label = "Interval width @95% (target units)"
        right_ylabel = "RMSE / Interval width (target units)"
    else:
        eff_plot = pd.to_numeric(summary_df["efficiency_mean"], errors="coerce")
        eff_std_plot = pd.to_numeric(summary_df["efficiency_std"], errors="coerce")
        eff_label = "Efficiency (dimensionless)"
        right_ylabel = "RMSE / Efficiency"

    b1 = ax1.bar(x - 1.5 * width, summary_df["r2_mean"], width, yerr=summary_df["r2_std"], label="R2 (bootstrap mean)")
    b2 = ax1.bar(x - 0.5 * width, summary_df["coverage_mean"], width, yerr=summary_df["coverage_std"], label="Coverage (bootstrap mean)")
    b3 = ax2.bar(x + 0.5 * width, summary_df["rmse_mean"], width, yerr=summary_df["rmse_std"], label="RMSE (bootstrap mean)", color="green")
    b4 = ax2.bar(x + 1.5 * width, eff_plot, width, yerr=eff_std_plot, label=eff_label, color="red")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35)
    ax1.set_ylabel("R2 / Coverage")
    ax2.set_ylabel(right_ylabel)

    handles = [b1, b2, b3, b4]
    labels_leg = [h.get_label() for h in handles]
    ax1.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4)
    save_plot(out_file)










def run_doa_suite(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    random_state: int,
    out_dir: Path,
) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    pipe = _unwrap_pipeline(model)
    pre: ColumnTransformer = pipe.named_steps["preprocessor"]
    train_repr, test_repr = reduce_matrix_pair(
        pre.transform(X_train),
        pre.transform(X_test),
        random_state=random_state,
        max_dim=30,
    )

    train_residual = np.abs(np.asarray(y_train) - np.asarray(y_train_pred))
    test_residual = np.abs(np.asarray(y_test) - np.asarray(y_test_pred))

    frame = pd.DataFrame(index=X_test.index)

    n_neighbors = min(6, len(train_repr))
    if n_neighbors >= 2:
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(train_repr)
        train_dist, _ = nn.kneighbors(train_repr)
        test_dist, test_idx = nn.kneighbors(test_repr)

        train_score = train_dist[:, 1:].mean(axis=1)
        test_score = test_dist[:, 1:].mean(axis=1) if test_dist.shape[1] > 1 else test_dist[:, 0]
        threshold = float(np.quantile(train_score, 0.95))

        frame["doa_knn_distance"] = test_score
        frame["doa_knn_in_domain"] = (test_score <= threshold).astype(int)

        plt.figure(figsize=(8, 5))
        sns.histplot(train_score, bins=30, alpha=0.5, label="Train")
        sns.histplot(test_score, bins=30, alpha=0.5, label="Test")
        plt.axvline(threshold, color="red", linestyle="--", label="95th percentile")
        plt.title("kNN Distance DoA")
        plt.legend()
        save_plot(out_dir / "doa_knn_distance_hist.png")

        local_q: list[float] = []
        for neigh in test_idx:
            neigh_idx = neigh[1:] if len(neigh) > 1 else neigh
            q = float(np.quantile(train_residual[neigh_idx], 0.95))
            local_q.append(q)

        local_q_arr = np.asarray(local_q, dtype=float)
        frame["doa_local_conformal_q95"] = local_q_arr
        frame["doa_local_conformal_in_domain"] = (test_residual <= local_q_arr).astype(int)

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=local_q_arr, y=test_residual, alpha=0.7, s=35)
        plt.xlabel("Local Conformal Threshold (q95)")
        plt.ylabel("Observed Absolute Residual")
        plt.title("Local Conformal Error Model")
        save_plot(out_dir / "doa_local_conformal_scatter.png")
    else:
        frame["doa_knn_distance"] = np.nan
        frame["doa_knn_in_domain"] = 1
        frame["doa_local_conformal_q95"] = np.nan
        frame["doa_local_conformal_in_domain"] = 1

    x_train = np.column_stack([np.ones(len(train_repr)), train_repr])
    x_test = np.column_stack([np.ones(len(test_repr)), test_repr])
    xtx_inv = np.linalg.pinv(x_train.T @ x_train)
    leverage_test = np.einsum("ij,jk,ik->i", x_test, xtx_inv, x_test)
    h_star = float(3.0 * x_train.shape[1] / len(x_train))

    frame["doa_leverage"] = leverage_test
    frame["doa_leverage_in_domain"] = (leverage_test <= h_star).astype(int)

    iso = IsolationForest(contamination=0.05, random_state=random_state)
    iso.fit(train_repr)
    train_iso = iso.score_samples(train_repr)
    test_iso = iso.score_samples(test_repr)
    iso_thr = float(np.quantile(train_iso, 0.05))

    frame["doa_isolation_score"] = test_iso
    frame["doa_isolation_in_domain"] = (test_iso >= iso_thr).astype(int)

    _, test_maha, maha_thr = compute_mahalanobis(train_repr, test_repr)
    frame["doa_mahalanobis"] = test_maha
    frame["doa_mahalanobis_in_domain"] = (test_maha <= maha_thr).astype(int)

    frame["doa_consensus_in_domain"] = (
        (frame["doa_knn_in_domain"] == 1)
        & (frame["doa_leverage_in_domain"] == 1)
        & (frame["doa_isolation_in_domain"] == 1)
        & (frame["doa_mahalanobis_in_domain"] == 1)
    ).astype(int)
    frame["abs_residual"] = test_residual

    frame.to_csv(out_dir / "doa_suite_metrics.csv", index=True)
    return frame


def analyze_regression_model(
    cfg: RunConfig,
    model_result: ModelResult,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_root: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model_dir = ensure_dir((out_root / model_result.label).resolve())

    scatter_metrics = plot_actual_vs_predicted_with_marginals(
        model_result.label,
        y_train,
        np.asarray(model_result.y_train_pred),
        y_test,
        np.asarray(model_result.y_test_pred),
        model_dir / "actual_vs_predicted_with_marginals.png",
    )

    residual_metrics = plot_residual_diagnostics(
        model_result.label,
        np.asarray(y_test),
        np.asarray(model_result.y_test_pred),
        model_dir / "residuals_vs_predicted.png",
        model_dir / "residual_distribution_qq.png",
    )

    plot_cv_score_distribution(
        model_result.label,
        model_result.cv_scores,
        "regression",
        model_dir / "cv_score_distribution.png",
    )
    plot_learning_curve(
        model_result.learning_curve_df,
        model_result.label,
        "RMSE",
        model_dir / "learning_curve.png",
    )

    calibration = compute_regression_calibration(np.asarray(y_test), np.asarray(model_result.y_test_pred), bins=12)
    plot_regression_calibration(model_result.label, calibration, model_dir / "calibration_curve.png")

    calibration_need: dict[str, Any] = {
        "status": "disabled",
        "calibration_needed": 0,
        "rmse_relative_gain": float("nan"),
        "mae_relative_gain": float("nan"),
    }
    calibrated_test_pred = np.asarray(model_result.y_test_pred)

    if cfg.enable_calibration_need_check and len(X_calib) > 0 and len(y_calib) > 0:
        y_calib_pred = model_result.pipeline.predict(X_calib)
        calibration_need, calibrated_test_pred = assess_regression_calibration_need(
            np.asarray(y_calib),
            np.asarray(y_calib_pred),
            np.asarray(y_test),
            np.asarray(model_result.y_test_pred),
            cfg.calibration_improvement_threshold,
        )

        pd.DataFrame([calibration_need]).to_csv(model_dir / "calibration_need_assessment.csv", index=False)
        pd.DataFrame(
            {
                "y_true": np.asarray(y_test),
                "y_pred_raw": np.asarray(model_result.y_test_pred),
                "y_pred_isotonic": calibrated_test_pred,
            }
        ).to_csv(model_dir / "isotonic_calibrated_predictions.csv", index=False)

        plot_isotonic_calibration_check(
            model_result.label,
            np.asarray(y_test),
            np.asarray(model_result.y_test_pred),
            calibrated_test_pred,
            calibration_need,
            model_dir / "isotonic_calibration_check.png",
        )

    curve_alpha_grid = _default_coverage_alpha_grid()
    eval_alphas = sorted(set([float(a) for a in list(cfg.uncertainty_alphas) + curve_alpha_grid if 0.0 < float(a) <= 1.0]))
    conf_df, intervals = compute_conformal_intervals(
        model_result.pipeline,
        X_calib,
        y_calib,
        X_test,
        y_test,
        eval_alphas,
    )
    if conf_df.empty:
        raise RuntimeError(f"Uncertainty metrics are empty for model {model_result.label}.")
    conf_df.to_csv(model_dir / "uncertainty_metrics.csv", index=False)

    reference_alpha = min(eval_alphas, key=lambda a: abs(float(a) - 0.05)) if eval_alphas else 0.05
    sharp_df, ence_df = compute_uncertainty_extra_metrics(
        np.asarray(y_test),
        intervals,
        reference_alpha=float(reference_alpha),
        n_bins=10,
    )
    if not sharp_df.empty:
        sharp_df.to_csv(model_dir / "uncertainty_sharpness_curve.csv", index=False)
        plot_uncertainty_sharpness_curve(model_result.label, sharp_df, model_dir / "uncertainty_sharpness_curve.png")
    if not ence_df.empty:
        ence_df.to_csv(model_dir / "uncertainty_ence_bins.csv", index=False)
        plot_ence_reliability(model_result.label, ence_df, model_dir / "uncertainty_ence_reliability.png")

    width_diag = plot_interval_width_vs_error(
        model_result.label,
        np.asarray(y_test),
        intervals,
        model_dir / "interval_width_vs_error.png",
    )

    subgroup_unc = compute_uncertainty_reliability_by_target_bins(np.asarray(y_test), intervals, reference_alpha=0.05, n_bins=4)
    subgroup_unc.to_csv(model_dir / "uncertainty_reliability_by_target_bins.csv", index=False)
    plot_uncertainty_reliability_by_bin(model_result.label, subgroup_unc, model_dir / "uncertainty_reliability_by_target_bins.png")

    curve_df = conf_df[conf_df["alpha"].round(6).isin(np.round(np.asarray(curve_alpha_grid, dtype=float), 6))].copy()
    if curve_df.empty:
        curve_df = conf_df.copy()
    cal_err = plot_conformal_coverage_curve(model_result.label, curve_df, model_dir / "coverage_curve.png")

    display_alphas: list[float] = []
    for alpha in cfg.uncertainty_alphas:
        a = float(alpha)
        if a in intervals:
            display_alphas.append(a)
    if not display_alphas:
        display_alphas = sorted(intervals.keys())[:3]

    for alpha in display_alphas:
        bundle = intervals[alpha]
        pct = int((1 - alpha) * 100)
        plot_ci_scatter(
            model_result.label,
            alpha,
            np.asarray(y_test),
            bundle["pred"],
            bundle["lower"],
            bundle["upper"],
            model_dir / f"ci_scatter_{pct}.png",
        )

    multi_levels = {float(a): intervals[float(a)] for a in display_alphas if float(a) in intervals}
    if not multi_levels:
        multi_levels = {float(a): intervals[float(a)] for a in sorted(intervals.keys())[:3]}
    plot_multi_ci_scatter(
        model_result.label,
        np.asarray(y_test),
        multi_levels,
        model_dir / "ci_scatter_multi_alpha.png",
        max_levels=3,
    )

    interval_cols: dict[str, np.ndarray] = {"y_true": np.asarray(y_test)}
    for alpha, bundle in intervals.items():
        alpha_tag = f"{alpha:.3f}".replace(".", "_")
        pred_arr = np.asarray(bundle["pred"])
        lower_arr = np.asarray(bundle["lower"])
        upper_arr = np.asarray(bundle["upper"])
        interval_cols[f"pred_alpha_{alpha_tag}"] = pred_arr
        interval_cols[f"lower_alpha_{alpha_tag}"] = lower_arr
        interval_cols[f"upper_alpha_{alpha_tag}"] = upper_arr
        interval_cols[f"covered_alpha_{alpha_tag}"] = np.asarray(bundle["covered"]).astype(int)
        interval_cols[f"width_alpha_{alpha_tag}"] = upper_arr - lower_arr
    interval_rows = pd.DataFrame(interval_cols, index=X_test.index)
    interval_rows.to_csv(model_dir / "uncertainty_intervals_samples.csv", index=True)

    try:
        plot_feature_correlation_heatmap(
            X_test,
            y_test,
            model_result.label,
            model_dir / "feature_correlation_heatmap.png",
            top_n=20,
        )
    except Exception as exc:
        logger.warning("Feature correlation heatmap failed for %s: %s", model_result.label, exc)

    outlier_df = run_outlier_analysis(
        model_result.label,
        model_result.pipeline,
        X_train,
        y_train,
        model_result.y_train_pred,
        X_test,
        y_test,
        model_result.y_test_pred,
        cfg.random_state,
        model_dir,
    )

    try:
        doa_df = run_doa_suite(
            model_result.pipeline,
            X_train,
            y_train,
            np.asarray(model_result.y_train_pred),
            X_test,
            y_test,
            np.asarray(model_result.y_test_pred),
            cfg.random_state,
            model_dir,
        )
    except Exception as exc:
        logger.warning("DoA suite failed for %s: %s", model_result.label, exc)
        doa_df = pd.DataFrame(index=X_test.index)

    try:
        run_feature_importance_and_shap(
            model_result.label,
            model_result.pipeline,
            X_test,
            y_test,
            "regression",
            cfg,
            model_dir,
            logger,
        )
    except Exception as exc:
        logger.warning("Feature importance/SHAP failed for %s: %s", model_result.label, exc)

    uncertainty_boot = build_uncertainty_summary_for_model(
        model_result.key,
        model_result.label,
        model_result.test_metrics,
        conf_df,
        cfg.bootstrap_repeats,
        cfg.random_state,
    )
    ensure_dir(model_dir)
    uncertainty_boot.to_csv(model_dir / "uncertainty_bootstrap_summary.csv", index=False)

    alpha95 = conf_df.iloc[np.argmin(np.abs(conf_df["alpha"].to_numpy(dtype=float) - 0.05))]

    summary = {
        "model_key": model_result.key,
        "model": model_result.label,
        **scatter_metrics,
        **residual_metrics,
        **width_diag,
        **cal_err,
        "coverage_95": float(alpha95["observed_coverage"]),
        "efficiency_95": float(alpha95["efficiency"]),
        "pinaw_95": float(alpha95["pinaw"]),
        "avg_width_95": float(alpha95["avg_width"]),
        "calibration_needed": int(calibration_need.get("calibration_needed", 0)),
        "calibration_status": str(calibration_need.get("status", "disabled")),
        "uncertainty_interval_mode": str(conf_df["interval_mode"].iloc[0]) if "interval_mode" in conf_df.columns else "split_conformal",
        "uncertainty_bin_mean_coverage": float(subgroup_unc["observed_coverage"].mean()) if not subgroup_unc.empty else float("nan"),
        "uncertainty_bin_coverage_std": float(subgroup_unc["observed_coverage"].std()) if not subgroup_unc.empty else float("nan"),
        "uncertainty_ence": float(ence_df["ence"].iloc[0]) if (not ence_df.empty and "ence" in ence_df.columns) else float("nan"),
        "uncertainty_pinaw_mean": float(sharp_df["pinaw"].mean()) if not sharp_df.empty else float("nan"),
        "uncertainty_avg_width": float(sharp_df["avg_width"].mean()) if not sharp_df.empty else float("nan"),
        "uncertainty_coverage_gap_mean": float(np.mean(np.abs(sharp_df["empirical_coverage"] - sharp_df["nominal_coverage"]))) if not sharp_df.empty else float("nan"),
        "outlier_count": int(outlier_df["williams_outlier"].sum()) if (not outlier_df.empty and "williams_outlier" in outlier_df.columns) else 0,
    }

    return conf_df, doa_df, summary


def analyze_classification_model(
    cfg: RunConfig,
    model_result: ModelResult,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_root: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    model_dir = ensure_dir((out_root / model_result.label).resolve())
    summary: dict[str, float] = {}

    plot_cv_score_distribution(
        model_result.label,
        model_result.cv_scores,
        "classification",
        model_dir / "cv_score_distribution.png",
    )

    y_pred = np.asarray(model_result.y_test_pred)
    proba = None if model_result.y_test_proba is None else np.asarray(model_result.y_test_proba, dtype=float)

    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(model_dir / "classification_predictions.csv", index=False)

    if proba is not None and proba.ndim == 2 and proba.shape[0] == len(y_test):
        max_prob = np.max(proba, axis=1)
        entropy = -np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1) / np.log(max(2, proba.shape[1]))

        plt.figure(figsize=(8, 5))
        sns.histplot(max_prob, bins=30, kde=True)
        plt.title(f"Prediction Confidence: {model_result.label}")
        save_plot(model_dir / "confidence_histogram.png")

        plt.figure(figsize=(8, 5))
        sns.histplot(entropy, bins=30, kde=True)
        plt.title(f"Prediction Entropy: {model_result.label}")
        save_plot(model_dir / "entropy_histogram.png")

        summary["mean_max_probability"] = float(np.mean(max_prob))
        summary["mean_entropy"] = float(np.mean(entropy))

    return summary


def compute_uncertainty_extra_metrics(
    y_true: np.ndarray,
    intervals: dict[float, dict[str, np.ndarray]],
    reference_alpha: float = 0.32,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not intervals:
        return pd.DataFrame(), pd.DataFrame()

    y_arr = np.asarray(y_true, dtype=float)
    y_range = float(np.max(y_arr) - np.min(y_arr) + 1e-12)

    sharp_rows: list[dict[str, float]] = []
    for alpha in sorted(intervals.keys()):
        bundle = intervals[alpha]
        lower = np.asarray(bundle["lower"], dtype=float)
        upper = np.asarray(bundle["upper"], dtype=float)
        covered = (y_arr >= lower) & (y_arr <= upper)
        width = upper - lower
        sharp_rows.append(
            {
                "alpha": float(alpha),
                "nominal_coverage": float(1.0 - alpha),
                "empirical_coverage": float(np.mean(covered)),
                "avg_width": float(np.mean(width)),
                "pinaw": float(np.mean(width) / y_range),
                "sharpness": float(1.0 / (1.0 + np.mean(width))),
            }
        )

    sharp_df = pd.DataFrame(sharp_rows).sort_values("alpha")

    available = np.asarray(sorted(intervals.keys()), dtype=float)
    idx = int(np.argmin(np.abs(available - float(reference_alpha))))
    alpha_ref = float(available[idx])
    bundle = intervals[alpha_ref]
    pred = np.asarray(bundle["pred"], dtype=float)
    lower = np.asarray(bundle["lower"], dtype=float)
    upper = np.asarray(bundle["upper"], dtype=float)

    z = float(norm.ppf(1.0 - alpha_ref / 2.0)) if 0 < alpha_ref < 1 else 1.0
    z = max(z, 1e-6)
    sigma_hat = np.maximum((upper - lower) / (2.0 * z), 1e-12)
    abs_err = np.abs(y_arr - pred)

    order = np.argsort(sigma_hat)
    sigma_ord = sigma_hat[order]
    err_ord = abs_err[order]

    n = int(len(sigma_ord))
    n_bins = max(2, min(int(n_bins), n))
    edges = np.linspace(0, n, n_bins + 1, dtype=int)

    ence_rows: list[dict[str, float]] = []
    contributions: list[float] = []
    for i in range(n_bins):
        a = int(edges[i])
        b = int(edges[i + 1])
        if b <= a:
            continue
        s_bin = sigma_ord[a:b]
        e_bin = err_ord[a:b]
        rmse_bin = float(np.sqrt(np.mean(e_bin ** 2)))
        sigma_bin = float(np.mean(s_bin))
        rel = float(abs(rmse_bin - sigma_bin) / (rmse_bin + 1e-12))
        contributions.append(rel)
        ence_rows.append(
            {
                "bin": i,
                "n_samples": int(b - a),
                "rmse_bin": rmse_bin,
                "sigma_bin": sigma_bin,
                "relative_gap": rel,
                "alpha_reference": alpha_ref,
            }
        )

    ence = float(np.mean(contributions)) if contributions else float("nan")
    ence_df = pd.DataFrame(ence_rows)
    if not ence_df.empty:
        ence_df["ence"] = ence

    return sharp_df, ence_df


def plot_uncertainty_sharpness_curve(model_label: str, sharp_df: pd.DataFrame, out_file: Path) -> None:
    if sharp_df.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(sharp_df["nominal_coverage"], sharp_df["avg_width"], marker="o", label="Avg width")
    plt.xlabel("Nominal Coverage")
    plt.ylabel("Average Interval Width")
    plt.title(f"Interval Sharpness Curve: {model_label}")
    plt.legend()
    save_plot(out_file)


def plot_ence_reliability(model_label: str, ence_df: pd.DataFrame, out_file: Path) -> None:
    if ence_df.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(ence_df["sigma_bin"], ence_df["rmse_bin"], s=40, alpha=0.8)
    lo = float(min(ence_df["sigma_bin"].min(), ence_df["rmse_bin"].min()))
    hi = float(max(ence_df["sigma_bin"].max(), ence_df["rmse_bin"].max()))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    ence_val = float(ence_df["ence"].iloc[0]) if "ence" in ence_df.columns else float("nan")
    plt.title(f"Uncertainty Reliability (ENCE={ence_val:.3f}): {model_label}")
    plt.xlabel("Mean predicted sigma per bin")
    plt.ylabel("Observed RMSE per bin")
    save_plot(out_file)















