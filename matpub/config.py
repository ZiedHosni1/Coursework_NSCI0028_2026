from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .utils import parse_csv_arg


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in parse_csv_arg(raw):
        try:
            values.append(float(item))
        except Exception:
            continue
    return values


@dataclass
class RunConfig:
    dataset: str | None = None
    target: str | None = None
    task: str | None = None
    run_mode: str = "fast"
    models: list[str] = field(default_factory=list)
    non_interactive: bool = False
    list_datasets: bool = False
    dataset_config_file: str | None = None

    random_state: int = 42
    test_size: float = 0.2
    calibration_size: float = 0.2
    cv_folds: int = 5
    nested_cv_outer: int = 3
    nested_cv_inner: int = 3
    nested_cv_repeats: int = 5
    n_jobs: int = 1
    group_column: str | None = None
    use_group_aware_split: bool = True
    use_group_aware_cv: bool = True

    max_categorical_cardinality: int = 30
    target_transform: str | None = None
    outlier_cleaning: str = "none"  # none|target_iqr|target_zscore
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 4.0
    imputation: str = "simple"
    scaling: str = "standard"
    include_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    enable_matminer_featurizers: bool = True
    enable_material_descriptor_enrichment: bool = True

    tuning_iterations_fast: int = 8
    tuning_iterations_complete: int = 25

    pairplot_features: int = 6
    eda_sample_size: int = 2000
    tsne_sample_size: int = 1500
    shap_sample_size: int = 400
    max_shap_features: int = 1500
    learning_curve_points: int = 8

    bootstrap_repeats: int = 200
    uncertainty_alphas: list[float] = field(default_factory=lambda: [0.32, 0.10, 0.05])

    enable_external_validation: bool = True
    external_datasets: list[str] = field(default_factory=list)

    enable_ablation_study: bool = True
    ablation_models: list[str] = field(default_factory=list)

    enable_robustness_tests: bool = True
    robustness_noise_std_fraction: float = 0.02
    robustness_missingness_levels: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    ood_cluster_count: int = 5

    enable_interpretation_stability: bool = True
    importance_stability_folds: int = 5
    importance_stability_repeats: int = 3

    enable_calibration_need_check: bool = True
    calibration_improvement_threshold: float = 0.01

    enable_physics_sanity_checks: bool = True
    expected_target_positive: bool = False

    enable_reproducibility_manifest: bool = True
    enable_experiment_registry: bool = True
    experiment_registry_filename: str = "experiments_registry.csv"

    publication_export_latex: bool = True
    performance_plot_secondary_axis: bool = True
    strict_checks: bool = True

    enable_repeated_runs: bool = True
    repeated_runs: int = 10

    permutation_test_repeats: int = 2000
    enable_bayesian_model_comparison: bool = True
    bayesian_rope: float = 0.0

    enable_leave_group_protocols: bool = True
    leave_group_min_groups: int = 5

    enable_graph_models: bool = True
    graph_model_epochs: int = 40
    graph_batch_size: int = 16

    enable_subgroup_robustness_breakdown: bool = True

    enable_leakage_scan: bool = True
    leakage_proxy_corr_threshold: float = 0.995
    near_duplicate_round_decimals: int = 6

    enable_train_variance_filter: bool = True
    train_variance_threshold: float = 0.0
    enable_train_correlation_filter: bool = True
    train_correlation_threshold: float = 0.995

    force_formula_core_featurizers: bool = True
    use_alloy_featurizer_precheck: bool = True

    enable_experiment_tracking: bool = True
    tracking_backend: str = "auto"  # auto|mlflow|wandb|none
    tracking_project: str = "matpub"
    tracking_uri: str = ""

    cache_dir: str = ".cache"
    use_cache: bool = True
    output_root: str = "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publication-grade materials ML workflow")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--task", type=str, choices=["regression", "classification"])
    parser.add_argument("--run-mode", type=str, choices=["fast", "complete"], default="fast")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--dataset-config-file", type=str, default="")

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--nested-cv-outer", type=int, default=3)
    parser.add_argument("--nested-cv-inner", type=int, default=3)
    parser.add_argument("--nested-cv-repeats", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--group-column", type=str, default="")
    parser.add_argument("--disable-group-aware-split", action="store_true")
    parser.add_argument("--disable-group-aware-cv", action="store_true")

    parser.add_argument("--max-cardinality", type=int, default=30)
    parser.add_argument(
        "--target-transform",
        type=str,
        choices=["none", "log1p", "yeo-johnson", "log1p_if_skewed", "optional_log1p_if_skewed"],
        default="",
    )
    parser.add_argument(
        "--outlier-cleaning",
        type=str,
        choices=["none", "target_iqr", "target_zscore"],
        default="none",
    )
    parser.add_argument("--outlier-iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--outlier-zscore-threshold", type=float, default=4.0)
    parser.add_argument("--include-columns", type=str, default="")
    parser.add_argument("--drop-columns", type=str, default="")
    parser.add_argument("--disable-matminer-featurizers", action="store_true")
    parser.add_argument("--disable-material-enrichment", action="store_true")

    parser.add_argument("--pairplot-features", type=int, default=6)
    parser.add_argument("--eda-sample-size", type=int, default=2000)
    parser.add_argument("--tsne-sample-size", type=int, default=1500)
    parser.add_argument("--shap-sample-size", type=int, default=400)
    parser.add_argument("--max-shap-features", type=int, default=1500)
    parser.add_argument("--learning-curve-points", type=int, default=8)

    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--alphas", type=str, default="0.32,0.10,0.05")

    parser.add_argument("--disable-external-validation", action="store_true")
    parser.add_argument("--external-datasets", type=str, default="")

    parser.add_argument("--disable-ablation-study", action="store_true")
    parser.add_argument("--ablation-models", type=str, default="")

    parser.add_argument("--disable-robustness-tests", action="store_true")
    parser.add_argument("--robustness-noise-std-fraction", type=float, default=0.02)
    parser.add_argument("--robustness-missingness-levels", type=str, default="0.05,0.10,0.20")
    parser.add_argument("--ood-cluster-count", type=int, default=5)

    parser.add_argument("--disable-interpretation-stability", action="store_true")
    parser.add_argument("--importance-stability-folds", type=int, default=5)
    parser.add_argument("--importance-stability-repeats", type=int, default=3)

    parser.add_argument("--disable-calibration-need-check", action="store_true")
    parser.add_argument("--calibration-improvement-threshold", type=float, default=0.01)

    parser.add_argument("--disable-physics-sanity-checks", action="store_true")
    parser.add_argument("--expected-target-positive", action="store_true")

    parser.add_argument("--disable-reproducibility-manifest", action="store_true")
    parser.add_argument("--disable-experiment-registry", action="store_true")
    parser.add_argument("--experiment-registry-filename", type=str, default="experiments_registry.csv")

    parser.add_argument("--disable-publication-latex", action="store_true")
    parser.add_argument("--disable-performance-secondary-axis", action="store_true")
    parser.add_argument("--disable-strict-checks", action="store_true")

    parser.add_argument("--disable-repeated-runs", action="store_true")
    parser.add_argument("--repeated-runs", type=int, default=10)

    parser.add_argument("--permutation-test-repeats", type=int, default=2000)
    parser.add_argument("--disable-bayesian-comparison", action="store_true")
    parser.add_argument("--bayesian-rope", type=float, default=0.0)

    parser.add_argument("--disable-leave-group-protocols", action="store_true")
    parser.add_argument("--leave-group-min-groups", type=int, default=5)

    parser.add_argument("--disable-graph-models", action="store_true")
    parser.add_argument("--graph-model-epochs", type=int, default=40)
    parser.add_argument("--graph-batch-size", type=int, default=16)

    parser.add_argument("--disable-subgroup-robustness", action="store_true")

    parser.add_argument("--disable-leakage-scan", action="store_true")
    parser.add_argument("--leakage-proxy-corr-threshold", type=float, default=0.995)
    parser.add_argument("--near-duplicate-round-decimals", type=int, default=6)

    parser.add_argument("--disable-train-variance-filter", action="store_true")
    parser.add_argument("--train-variance-threshold", type=float, default=0.0)
    parser.add_argument("--disable-train-correlation-filter", action="store_true")
    parser.add_argument("--train-correlation-threshold", type=float, default=0.995)

    parser.add_argument("--disable-force-formula-core-featurizers", action="store_true")
    parser.add_argument("--disable-alloy-featurizer-precheck", action="store_true")

    parser.add_argument("--disable-experiment-tracking", action="store_true")
    parser.add_argument("--tracking-backend", type=str, choices=["auto", "mlflow", "wandb", "none"], default="auto")
    parser.add_argument("--tracking-project", type=str, default="matpub")
    parser.add_argument("--tracking-uri", type=str, default="")

    parser.add_argument("--cache-dir", type=str, default=".cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-root", type=str, default="outputs")
    return parser


def args_to_config(args: argparse.Namespace) -> RunConfig:
    missingness = _parse_float_list(args.robustness_missingness_levels)
    return RunConfig(
        dataset=args.dataset,
        target=args.target,
        task=args.task,
        run_mode=args.run_mode,
        models=parse_csv_arg(args.models),
        non_interactive=args.non_interactive,
        list_datasets=args.list_datasets,
        dataset_config_file=args.dataset_config_file.strip() or None,
        random_state=args.random_state,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
        cv_folds=args.cv_folds,
        nested_cv_outer=args.nested_cv_outer,
        nested_cv_inner=args.nested_cv_inner,
        nested_cv_repeats=args.nested_cv_repeats,
        n_jobs=max(1, int(args.n_jobs)),
        group_column=args.group_column.strip() or None,
        use_group_aware_split=not args.disable_group_aware_split,
        use_group_aware_cv=not args.disable_group_aware_cv,
        max_categorical_cardinality=args.max_cardinality,
        target_transform=(str(args.target_transform).strip() if str(args.target_transform or "").strip() != "" else None),
        outlier_cleaning=str(args.outlier_cleaning or "none").strip().lower(),
        outlier_iqr_multiplier=max(0.5, float(args.outlier_iqr_multiplier)),
        outlier_zscore_threshold=max(1.0, float(args.outlier_zscore_threshold)),
        include_columns=parse_csv_arg(args.include_columns),
        drop_columns=parse_csv_arg(args.drop_columns),
        enable_matminer_featurizers=not args.disable_matminer_featurizers,
        enable_material_descriptor_enrichment=not args.disable_material_enrichment,
        pairplot_features=args.pairplot_features,
        eda_sample_size=args.eda_sample_size,
        tsne_sample_size=args.tsne_sample_size,
        shap_sample_size=args.shap_sample_size,
        max_shap_features=args.max_shap_features,
        learning_curve_points=args.learning_curve_points,
        bootstrap_repeats=args.bootstrap_repeats,
        uncertainty_alphas=[float(item) for item in parse_csv_arg(args.alphas)],
        enable_external_validation=not args.disable_external_validation,
        external_datasets=parse_csv_arg(args.external_datasets),
        enable_ablation_study=not args.disable_ablation_study,
        ablation_models=parse_csv_arg(args.ablation_models),
        enable_robustness_tests=not args.disable_robustness_tests,
        robustness_noise_std_fraction=args.robustness_noise_std_fraction,
        robustness_missingness_levels=missingness if missingness else [0.05, 0.10, 0.20],
        ood_cluster_count=max(2, int(args.ood_cluster_count)),
        enable_interpretation_stability=not args.disable_interpretation_stability,
        importance_stability_folds=max(2, int(args.importance_stability_folds)),
        importance_stability_repeats=max(1, int(args.importance_stability_repeats)),
        enable_calibration_need_check=not args.disable_calibration_need_check,
        calibration_improvement_threshold=float(args.calibration_improvement_threshold),
        enable_physics_sanity_checks=not args.disable_physics_sanity_checks,
        expected_target_positive=bool(args.expected_target_positive),
        enable_reproducibility_manifest=not args.disable_reproducibility_manifest,
        enable_experiment_registry=not args.disable_experiment_registry,
        experiment_registry_filename=args.experiment_registry_filename,
        publication_export_latex=not args.disable_publication_latex,
        performance_plot_secondary_axis=not args.disable_performance_secondary_axis,
        strict_checks=not args.disable_strict_checks,
        enable_repeated_runs=not args.disable_repeated_runs,
        repeated_runs=max(1, int(args.repeated_runs)),
        permutation_test_repeats=max(100, int(args.permutation_test_repeats)),
        enable_bayesian_model_comparison=not args.disable_bayesian_comparison,
        bayesian_rope=max(0.0, float(args.bayesian_rope)),
        enable_leave_group_protocols=not args.disable_leave_group_protocols,
        leave_group_min_groups=max(2, int(args.leave_group_min_groups)),
        enable_graph_models=not args.disable_graph_models,
        graph_model_epochs=max(1, int(args.graph_model_epochs)),
        graph_batch_size=max(2, int(args.graph_batch_size)),
        enable_subgroup_robustness_breakdown=not args.disable_subgroup_robustness,
        enable_leakage_scan=not args.disable_leakage_scan,
        leakage_proxy_corr_threshold=min(0.999999, max(0.5, float(args.leakage_proxy_corr_threshold))),
        near_duplicate_round_decimals=max(0, int(args.near_duplicate_round_decimals)),
        enable_train_variance_filter=not args.disable_train_variance_filter,
        train_variance_threshold=max(0.0, float(args.train_variance_threshold)),
        enable_train_correlation_filter=not args.disable_train_correlation_filter,
        train_correlation_threshold=min(0.999999, max(0.5, float(args.train_correlation_threshold))),
        force_formula_core_featurizers=not args.disable_force_formula_core_featurizers,
        use_alloy_featurizer_precheck=not args.disable_alloy_featurizer_precheck,
        enable_experiment_tracking=not args.disable_experiment_tracking,
        tracking_backend=args.tracking_backend,
        tracking_project=args.tracking_project,
        tracking_uri=args.tracking_uri,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        output_root=args.output_root,
    )




