from __future__ import annotations

from pathlib import Path

import pandas as pd


def _image_tag(path: Path, title: str) -> str:
    if not path.exists():
        return ""
    rel = path.name
    return f"<h4>{title}</h4><img src='{rel}' style='max-width:100%; border:1px solid #ccc;'/>"


def _table_block(path: Path, title: str, max_rows: int = 400) -> str:
    if not path.exists():
        return ""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return ""
    if frame.empty:
        return f"<h3>{title}</h3><p>No rows.</p>"
    if len(frame) > max_rows:
        frame = frame.head(max_rows)
    return f"<h3>{title}</h3>{frame.to_html(index=False)}"


def generate_html_report(
    output_dir: Path,
    model_results_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    statistical_df: pd.DataFrame,
    uncertainty_summary_df: pd.DataFrame,
) -> Path:
    lines: list[str] = []
    lines.append("<html><head><meta charset='utf-8'><title>Materials ML Report</title>")
    lines.append("<style>body{font-family:Arial,sans-serif;margin:24px;} h1,h2{margin-top:30px;} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px;} img{margin:10px 0;} .model{border-top:2px solid #444;padding-top:20px;margin-top:20px;}</style>")
    lines.append("</head><body>")
    lines.append("<h1>Materials ML Pipeline Report</h1>")

    lines.append("<h2>Model Metrics</h2>")
    lines.append(model_results_df.to_html(index=False))

    lines.append("<h2>Multi-objective Ranking</h2>")
    lines.append(ranking_df.to_html(index=False))

    lines.append("<h2>Statistical Comparison</h2>")
    lines.append(statistical_df.to_html(index=False))

    lines.append("<h2>Uncertainty Summary</h2>")
    lines.append(uncertainty_summary_df.to_html(index=False))

    lines.append("<h2>Data Integrity Checks</h2>")
    for csv_name, title in [
        ("data_integrity_raw.csv", "Raw Data Integrity"),
        ("data_integrity_features.csv", "Feature Matrix Integrity"),
        ("data_integrity_splits.csv", "Split Integrity"),
        ("leakage_scan.csv", "Leakage Scan"),
        ("repeated_runs_summary.csv", "Repeated Runs Summary"),
        ("cross_dataset_generalization.csv", "Cross-dataset Generalization"),
        ("model_permutation_comparison.csv", "Permutation Model Comparison"),
        ("model_bayesian_comparison.csv", "Bayesian Model Comparison"),
        ("leave_group_protocols.csv", "Leave-Group-Out Protocols"),
        ("subgroup_robustness_breakdown.csv", "Subgroup Robustness Breakdown"),
        ("structure_graph_benchmark_status.csv", "Structure Graph Benchmark Status"),
        ("structure_graph_predictions.csv", "Structure Graph Predictions"),
        ("experiment_tracking_status.csv", "Experiment Tracking Status"),
        ("artifact_hashes.csv", "Artifact Hashes"),
    ]:
        block = _table_block(output_dir / csv_name, title)
        if block:
            lines.append(block)

    lines.append("<h2>Dataset Diagnostics</h2>")
    for image_name, title in [
        ("dataset_target_distribution.png", "Target Distribution"),
        ("dataset_missingness_heatmap.png", "Missingness Heatmap"),
        ("dataset_numeric_correlation_heatmap.png", "Numeric Correlation Heatmap"),
        ("dataset_pairplot.png", "Feature Pairplot"),
        ("dataset_feature_space_pca.png", "Feature-Space PCA"),
        ("dataset_feature_space_tsne.png", "Feature-Space t-SNE"),
    ]:
        image_path = output_dir / image_name
        if image_path.exists():
            lines.append(_image_tag(image_path, title))

    lines.append("<h2>Global Comparison Plots</h2>")
    for image_name, title in [
        ("model_runtime_comparison.png", "Runtime Comparison"),
        ("model_performance_comparison.png", "Performance Comparison"),
        ("model_cv_primary_with_errorbars.png", "CV Primary Metric with Error Bars"),
        ("model_metric_vs_runtime.png", "Metric vs Runtime Tradeoff"),
        ("uncertainty_comparison.png", "Uncertainty Comparison"),
        ("error_rate_vs_significance_models.png", "Error Rate vs Significance Across Models"),
        ("model_permutation_pvalue_heatmap.png", "Permutation p-value Heatmap"),
        ("model_bayesian_posterior_heatmap.png", "Bayesian Posterior Heatmap"),
        ("leave_group_protocols.png", "Leave-Group-Out Protocol Comparison"),
        ("repeated_runs_rmse_ci95.png", "Repeated Runs RMSE Mean ± 95% CI"),
        ("repeated_runs_f1_ci95.png", "Repeated Runs F1 Mean ± 95% CI"),
        ("cross_dataset_generalization.png", "Cross-dataset Generalization"),
        ("subgroup_coverage_breakdown.png", "Subgroup Coverage Breakdown"),
        ("subgroup_rmse_breakdown.png", "Subgroup RMSE Breakdown"),
    ]:
        image_path = output_dir / image_name
        if image_path.exists():
            lines.append(_image_tag(image_path, title))

    model_root = output_dir / "models"
    if model_root.exists():
        lines.append("<h2>Per-model Plots</h2>")
        for model_dir in sorted([item for item in model_root.iterdir() if item.is_dir()]):
            lines.append(f"<div class='model'><h3>{model_dir.name}</h3>")
            for image_name in [
                "actual_vs_predicted_with_marginals.png",
                "residuals_vs_predicted.png",
                "residual_distribution_qq.png",
                "learning_curve.png",
                "cv_score_distribution.png",
                "calibration_curve.png",
                "isotonic_calibration_check.png",
                "coverage_curve.png",
                "interval_width_vs_error.png",
                "uncertainty_sharpness_curve.png",
                "uncertainty_ence_reliability.png",
                "uncertainty_reliability_by_target_bins.png",
                "ci_scatter_68.png",
                "ci_scatter_90.png",
                "ci_scatter_95.png",
                "ci_scatter_multi_alpha.png",
                "feature_correlation_heatmap.png",
                "mahalanobis_distribution.png",
                "williams_plot.png",
                "doa_knn_distance_hist.png",
                "doa_local_conformal_scatter.png",
                "permutation_importance.png",
                "permutation_stability_top20.png",
                "shap_beeswarm.png",
                "shap_bar.png",
                "confidence_histogram.png",
                "entropy_histogram.png",
            ]:
                image_path = model_dir / image_name
                if image_path.exists():
                    rel = f"models/{model_dir.name}/{image_name}"
                    lines.append(f"<p><b>{image_name}</b><br><img src='{rel}' style='max-width:100%; border:1px solid #ccc;'/></p>")
            lines.append("</div>")

    lines.append("</body></html>")

    report_path = output_dir / "report.html"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path




