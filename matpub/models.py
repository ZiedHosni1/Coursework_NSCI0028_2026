from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, t as student_t, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC, SVR
from sklearn.utils.validation import has_fit_parameter

warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn\\.neural_network\\._multilayer_perceptron")

from .config import RunConfig
from .utils import finite_or_nan, normalize_series, now_seconds


@dataclass
class ModelDefinition:
    key: str
    label: str
    estimator: Any
    tune_space: dict[str, Any]


@dataclass
class ModelResult:
    key: str
    label: str
    pipeline: Any
    cv_primary: float
    cv_std: float
    nested_cv_primary: float | None
    nested_cv_mean: float | None
    nested_cv_std: float | None
    nested_cv_ci_low: float | None
    nested_cv_ci_high: float | None
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    fit_seconds: float
    cv_seconds: float
    predict_seconds: float
    total_seconds: float
    best_params: dict[str, Any] | None
    tune_space: dict[str, Any]
    learning_curve_df: pd.DataFrame
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    y_train_proba: np.ndarray | None
    y_test_proba: np.ndarray | None
    cv_scores: np.ndarray | None = None
    target_transform: str | None = None
    target_transformer: Any | None = None



class _CatBoostRegressorCompat(BaseEstimator, RegressorMixin):
    def __init__(self, **params: Any):
        self._params = dict(params)
        for k, v in self._params.items():
            setattr(self, k, v)

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return dict(self._params)

    def set_params(self, **params: Any):
        self._params.update(params)
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: Any, y: Any, sample_weight: Any = None):
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**self._params)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X, y, **fit_kwargs)
        self.model_ = model
        try:
            self.n_features_in_ = int(X.shape[1])
        except Exception:
            pass
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(X))


class _CatBoostClassifierCompat(BaseEstimator, ClassifierMixin):
    def __init__(self, **params: Any):
        self._params = dict(params)
        for k, v in self._params.items():
            setattr(self, k, v)

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return dict(self._params)

    def set_params(self, **params: Any):
        self._params.update(params)
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: Any, y: Any, sample_weight: Any = None):
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(**self._params)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X, y, **fit_kwargs)
        self.model_ = model
        try:
            self.n_features_in_ = int(X.shape[1])
        except Exception:
            pass
        self.classes_ = np.asarray(getattr(model, "classes_", np.unique(y)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X))
def _import_optional_estimators(task: str) -> dict[str, Any]:
    opt: dict[str, Any] = {}

    try:
        if task == "regression":
            from xgboost import XGBRegressor

            opt["xgboost"] = XGBRegressor
        else:
            from xgboost import XGBClassifier

            opt["xgboost"] = XGBClassifier
    except Exception:
        pass


    try:
        import catboost  # noqa: F401

        if task == "regression":
            opt["catboost"] = _CatBoostRegressorCompat
        else:
            opt["catboost"] = _CatBoostClassifierCompat
    except Exception:
        pass

    return opt


def get_primary_scorer(task: str) -> str:
    return "neg_root_mean_squared_error" if task == "regression" else "f1_weighted"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_cv(
    task: str,
    y: pd.Series,
    folds: int,
    random_state: int,
    groups: pd.Series | None = None,
    use_group_aware: bool = True,
) -> GroupKFold | KFold | StratifiedKFold:
    if use_group_aware and groups is not None:
        n_groups = int(groups.astype("string").nunique(dropna=True))
        if n_groups >= 2:
            n = max(2, min(folds, n_groups))
            return GroupKFold(n_splits=n)

    if task == "classification":
        min_class = int(y.value_counts().min()) if len(y) > 0 else 2
        n = max(2, min(folds, min_class))
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=random_state)
    n = max(2, min(folds, max(2, len(y) // 5)))
    return KFold(n_splits=n, shuffle=True, random_state=random_state)


def _has_constant_train_target_in_cv(
    cv: GroupKFold | KFold | StratifiedKFold,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
) -> bool:
    y_num = pd.to_numeric(y, errors="coerce")
    try:
        if isinstance(cv, GroupKFold):
            if groups is None:
                return False
            split_iter = cv.split(X, y_num, groups)
        else:
            split_iter = cv.split(X, y_num)

        for tr_idx, _ in split_iter:
            fold = y_num.iloc[tr_idx]
            if int(fold.nunique(dropna=True)) < 2:
                return True
    except Exception:
        return False
    return False


def _base_model_definitions(task: str, random_state: int, n_jobs: int) -> list[ModelDefinition]:
    defs: list[ModelDefinition] = []
    model_n_jobs = max(1, int(n_jobs))

    if task == "regression":
        defs.extend(
            [
                ModelDefinition("baseline", "Baseline", DummyRegressor(strategy="median"), {}),
                ModelDefinition("ridge", "Ridge", Ridge(alpha=1.0), {"alpha": loguniform(1e-3, 1e3)}),
                ModelDefinition("lasso", "Lasso", Lasso(alpha=1e-2, max_iter=5000), {"alpha": loguniform(1e-5, 1e1)}),
                ModelDefinition(
                    "elastic_net",
                    "ElasticNet",
                    ElasticNet(alpha=1e-2, l1_ratio=0.5, max_iter=5000),
                    {"alpha": loguniform(1e-5, 1e1), "l1_ratio": uniform(0.05, 0.90)},
                ),
                ModelDefinition(
                    "gaussian_process",
                    "GaussianProcess",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
                        alpha=1e-6,
                        random_state=random_state,
                        normalize_y=True,
                    ),
                    {"alpha": loguniform(1e-8, 1e-2)},
                ),
                ModelDefinition(
                    "random_forest",
                    "RandomForest",
                    RandomForestRegressor(n_estimators=350, random_state=random_state, n_jobs=model_n_jobs),
                    {
                        "n_estimators": randint(200, 900),
                        "max_depth": [None, 6, 12, 24],
                        "min_samples_leaf": randint(1, 6),
                        "max_features": ["sqrt", 0.5, 1.0],
                    },
                ),
                ModelDefinition(
                    "extra_trees",
                    "ExtraTrees",
                    ExtraTreesRegressor(n_estimators=350, random_state=random_state, n_jobs=model_n_jobs),
                    {
                        "n_estimators": randint(200, 900),
                        "max_depth": [None, 6, 12, 24],
                        "min_samples_leaf": randint(1, 6),
                        "max_features": ["sqrt", 0.5, 1.0],
                    },
                ),
                ModelDefinition(
                    "gradient_boosting",
                    "GradientBoosting",
                    GradientBoostingRegressor(random_state=random_state),
                    {
                        "n_estimators": randint(100, 700),
                        "learning_rate": loguniform(1e-3, 3e-1),
                        "max_depth": randint(2, 6),
                        "subsample": uniform(0.6, 0.4),
                    },
                ),
                ModelDefinition(
                    "svm",
                    "SVR",
                    SVR(kernel="rbf"),
                    {
                        "C": loguniform(1e-2, 1e3),
                        "gamma": loguniform(1e-4, 1e-1),
                        "epsilon": loguniform(1e-4, 5e-1),
                    },
                ),
                ModelDefinition(
                    "ann",
                    "ANN",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        random_state=random_state,
                        max_iter=800,
                        early_stopping=True,
                    ),
                    {
                        "hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
                        "alpha": loguniform(1e-6, 1e-1),
                        "learning_rate_init": loguniform(1e-4, 5e-2),
                    },
                ),
            ]
        )
    else:
        defs.extend(
            [
                ModelDefinition("baseline", "Baseline", DummyClassifier(strategy="prior"), {}),
                ModelDefinition(
                    "logistic",
                    "LogisticRegression",
                    LogisticRegression(max_iter=3000),
                    {"C": loguniform(1e-3, 1e3), "class_weight": [None, "balanced"]},
                ),
                ModelDefinition(
                    "random_forest",
                    "RandomForest",
                    RandomForestClassifier(n_estimators=350, random_state=random_state, n_jobs=model_n_jobs),
                    {
                        "n_estimators": randint(200, 900),
                        "max_depth": [None, 6, 12, 24],
                        "min_samples_leaf": randint(1, 6),
                        "class_weight": [None, "balanced"],
                        "max_features": ["sqrt", 0.5, 1.0],
                    },
                ),
                ModelDefinition(
                    "extra_trees",
                    "ExtraTrees",
                    ExtraTreesClassifier(n_estimators=350, random_state=random_state, n_jobs=model_n_jobs),
                    {
                        "n_estimators": randint(200, 900),
                        "max_depth": [None, 6, 12, 24],
                        "min_samples_leaf": randint(1, 6),
                        "class_weight": [None, "balanced"],
                        "max_features": ["sqrt", 0.5, 1.0],
                    },
                ),
                ModelDefinition(
                    "gradient_boosting",
                    "GradientBoosting",
                    GradientBoostingClassifier(random_state=random_state),
                    {
                        "n_estimators": randint(100, 700),
                        "learning_rate": loguniform(1e-3, 3e-1),
                        "max_depth": randint(2, 6),
                        "subsample": uniform(0.6, 0.4),
                    },
                ),
                ModelDefinition(
                    "svm",
                    "SVC",
                    SVC(probability=True, random_state=random_state),
                    {
                        "C": loguniform(1e-3, 1e3),
                        "gamma": loguniform(1e-4, 5e-1),
                        "class_weight": [None, "balanced"],
                    },
                ),
                ModelDefinition(
                    "ann",
                    "ANN",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        random_state=random_state,
                        max_iter=800,
                        early_stopping=True,
                    ),
                    {
                        "hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
                        "alpha": loguniform(1e-6, 1e-1),
                        "learning_rate_init": loguniform(1e-4, 5e-2),
                    },
                ),
            ]
        )

    optional = _import_optional_estimators(task)
    if "xgboost" in optional:
        if task == "regression":
            defs.append(
                ModelDefinition(
                    "xgboost",
                    "XGBoost",
                    optional["xgboost"](
                        n_estimators=450,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=model_n_jobs,
                    ),
                    {
                        "n_estimators": randint(200, 1200),
                        "max_depth": randint(2, 12),
                        "learning_rate": loguniform(1e-3, 3e-1),
                        "subsample": uniform(0.5, 0.5),
                        "colsample_bytree": uniform(0.5, 0.5),
                    },
                )
            )
        else:
            defs.append(
                ModelDefinition(
                    "xgboost",
                    "XGBoost",
                    optional["xgboost"](
                        n_estimators=450,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=model_n_jobs,
                        eval_metric="logloss",
                    ),
                    {
                        "n_estimators": randint(200, 1200),
                        "max_depth": randint(2, 12),
                        "learning_rate": loguniform(1e-3, 3e-1),
                        "subsample": uniform(0.5, 0.5),
                        "colsample_bytree": uniform(0.5, 0.5),
                    },
                )
            )


    if "catboost" in optional:
        if task == "regression":
            defs.append(
                ModelDefinition(
                    "catboost",
                    "CatBoost",
                    optional["catboost"](verbose=False, random_state=random_state),
                    {
                        "depth": randint(3, 11),
                        "learning_rate": loguniform(1e-3, 3e-1),
                        "iterations": randint(150, 1200),
                    },
                )
            )
        else:
            defs.append(
                ModelDefinition(
                    "catboost",
                    "CatBoost",
                    optional["catboost"](verbose=False, random_state=random_state),
                    {
                        "depth": randint(3, 11),
                        "learning_rate": loguniform(1e-3, 3e-1),
                        "iterations": randint(150, 1200),
                    },
                )
            )

    return defs


def get_model_definitions(cfg: RunConfig, task: str, logger: logging.Logger) -> list[ModelDefinition]:
    defs = _base_model_definitions(task, cfg.random_state, cfg.n_jobs)

    if cfg.models:
        wanted = {key.lower() for key in cfg.models}
        filtered = [item for item in defs if item.key.lower() in wanted or item.label.lower() in wanted]
        if not filtered:
            logger.warning("No requested models available from --models. Falling back to defaults.")
        else:
            defs = filtered

    if cfg.run_mode == "fast":
        preferred = ["baseline", "ridge" if task == "regression" else "logistic", "random_forest", "xgboost"]
        ordered = [item for key in preferred for item in defs if item.key == key]
        leftovers = [item for item in defs if item.key not in preferred]
        defs = ordered + leftovers
        defs = defs[: min(4, len(defs))]

    if not defs:
        raise RuntimeError("No models are available for the selected task/profile.")

    keys = [item.key for item in defs]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate model keys detected in model definitions: {keys}")

    logger.info("Selected models: %s", [item.label for item in defs])
    return defs


def serialize_space_value(value: Any) -> str:
    if hasattr(value, "dist"):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join([str(v) for v in value])
    return str(value)


def export_tuning_spaces(definitions: list[ModelDefinition]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for item in definitions:
        if not item.tune_space:
            rows.append({"model": item.label, "parameter": "<none>", "range": "fixed"})
            continue
        for param, value in item.tune_space.items():
            rows.append({"model": item.label, "parameter": param, "range": serialize_space_value(value)})
    return pd.DataFrame(rows)


def evaluate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    rmse = _rmse(y_true_np, y_pred_np)
    mae = float(mean_absolute_error(y_true_np, y_pred_np))
    medae = float(median_absolute_error(y_true_np, y_pred_np))
    bias = float(np.mean(y_pred_np - y_true_np))
    smape = float(np.mean(2.0 * np.abs(y_pred_np - y_true_np) / (np.abs(y_true_np) + np.abs(y_pred_np) + 1e-12)))

    n_obs = int(y_true_np.shape[0])
    r2_val = float(r2_score(y_true_np, y_pred_np)) if n_obs >= 2 else float("nan")
    explained_var = float(explained_variance_score(y_true_np, y_pred_np)) if n_obs >= 2 else float("nan")
    max_err = float(max_error(y_true_np, y_pred_np)) if n_obs >= 1 else float("nan")

    metrics = {
        "r2": r2_val,
        "rmse": rmse,
        "mse": float(mean_squared_error(y_true_np, y_pred_np)),
        "mae": mae,
        "medae": medae,
        "mape": float(mean_absolute_percentage_error(y_true_np, y_pred_np)),
        "smape": smape,
        "explained_variance": explained_var,
        "max_error": max_err,
        "bias": bias,
    }

    positive_mask = (y_true_np > 0) & (y_pred_np > 0)
    if positive_mask.sum() > 0:
        metrics["rmsle"] = float(np.sqrt(mean_squared_error(np.log1p(y_true_np[positive_mask]), np.log1p(y_pred_np[positive_mask]))))
    else:
        metrics["rmsle"] = float("nan")

    return metrics


def evaluate_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "precision_macro": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_np, y_pred_np)),
        "cohen_kappa": float(cohen_kappa_score(y_true_np, y_pred_np)),
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true_np, y_proba))
        except Exception:
            metrics["log_loss"] = float("nan")

        try:
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true_np, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc"] = float("nan")

        try:
            if y_proba.shape[1] == 2:
                classes = np.unique(y_true_np)
                y_bin = (y_true_np == classes[1]).astype(float)
                metrics["brier"] = float(np.mean((y_bin - y_proba[:, 1]) ** 2))
            else:
                metrics["brier"] = float("nan")
        except Exception:
            metrics["brier"] = float("nan")

    return metrics


def _empty_learning_curve_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_size": pd.Series(dtype=float),
            "train_mean": pd.Series(dtype=float),
            "train_std": pd.Series(dtype=float),
            "cv_mean": pd.Series(dtype=float),
            "cv_std": pd.Series(dtype=float),
        }
    )


def compute_learning_curve(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv: GroupKFold | KFold | StratifiedKFold,
    points: int,
    n_jobs: int = 1,
    random_state: int = 42,
    groups: pd.Series | None = None,
    use_group_aware: bool = True,
) -> pd.DataFrame:
    del n_jobs

    if len(X) < 10:
        return _empty_learning_curve_frame()

    n_samples = len(X)
    min_size_base = 20 if task == "regression" else max(20, int(max(2, y.nunique()) * 5))

    def _max_train_size(cv_used: GroupKFold | KFold | StratifiedKFold, groups_used: pd.Series | None) -> int:
        try:
            if isinstance(cv_used, GroupKFold):
                if groups_used is None:
                    return 0
                fold_train_sizes = [len(train_idx) for train_idx, _ in cv_used.split(X, y, groups_used)]
            else:
                fold_train_sizes = [len(train_idx) for train_idx, _ in cv_used.split(X, y)]
            return int(min(fold_train_sizes)) if fold_train_sizes else 0
        except Exception:
            return max(0, n_samples - 1)

    def _build_sizes(max_train_size: int, n_points: int) -> np.ndarray:
        if max_train_size < 2:
            return np.array([], dtype=int)
        start = min(min_size_base, max(2, max_train_size // 2))
        raw = np.linspace(start, max_train_size, max(2, n_points), dtype=int)
        return np.unique(np.clip(raw, 2, max_train_size))

    fallback_cv = get_cv(
        task,
        y,
        3,
        random_state,
        groups=groups,
        use_group_aware=use_group_aware,
    )

    attempts: list[tuple[GroupKFold | KFold | StratifiedKFold, int, pd.Series | None]] = [
        (cv, points, groups),
        (fallback_cv, min(5, max(2, points)), groups),
    ]

    rng = np.random.default_rng(random_state)

    for cv_used, n_points, groups_used in attempts:
        max_train_size = _max_train_size(cv_used, groups_used)
        sizes = _build_sizes(max_train_size, n_points)
        if sizes.size < 2:
            continue

        rows: list[dict[str, float]] = []
        for size in sizes:
            if int(size) >= len(X):
                subset_idx = np.arange(len(X))
            else:
                subset_idx = rng.choice(np.arange(len(X)), size=int(size), replace=False)

            X_sub = X.iloc[subset_idx]
            y_sub = y.iloc[subset_idx]
            g_sub = groups_used.iloc[subset_idx] if groups_used is not None else None

            train_scores: list[float] = []
            val_scores: list[float] = []

            try:
                if isinstance(cv_used, GroupKFold):
                    if g_sub is None or int(g_sub.astype("string").nunique(dropna=True)) < 2:
                        continue
                    split_iter = cv_used.split(X_sub, y_sub, g_sub)
                else:
                    split_iter = cv_used.split(X_sub, y_sub)

                for tr_idx, va_idx in split_iter:
                    X_tr = X_sub.iloc[tr_idx]
                    y_tr = y_sub.iloc[tr_idx]
                    X_va = X_sub.iloc[va_idx]
                    y_va = y_sub.iloc[va_idx]

                    fold_model = clone(model)
                    fold_model.fit(X_tr, y_tr)

                    pred_tr = fold_model.predict(X_tr)
                    pred_va = fold_model.predict(X_va)

                    if task == "regression":
                        train_scores.append(_rmse(np.asarray(y_tr), np.asarray(pred_tr)))
                        val_scores.append(_rmse(np.asarray(y_va), np.asarray(pred_va)))
                    else:
                        train_scores.append(float(f1_score(y_tr, pred_tr, average="weighted", zero_division=0)))
                        val_scores.append(float(f1_score(y_va, pred_va, average="weighted", zero_division=0)))

                if val_scores:
                    rows.append(
                        {
                            "train_size": float(size),
                            "train_mean": float(np.mean(train_scores)),
                            "train_std": float(np.std(train_scores)),
                            "cv_mean": float(np.mean(val_scores)),
                            "cv_std": float(np.std(val_scores)),
                        }
                    )
            except Exception:
                continue

        frame = pd.DataFrame(rows)
        if not frame.empty:
            return frame.sort_values("train_size").reset_index(drop=True)

    return _empty_learning_curve_frame()

def nested_cv_estimate(
    estimator: Any,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    tune_space: dict[str, Any],
    cfg: RunConfig,
    groups: pd.Series | None,
) -> dict[str, float] | None:
    if cfg.run_mode != "complete":
        return None

    repeats = max(1, int(cfg.nested_cv_repeats))
    scorer = get_primary_scorer(task)
    repeat_values: list[float] = []

    for rep in range(repeats):
        seed = cfg.random_state + rep * 97
        outer = get_cv(
            task,
            y,
            max(2, cfg.nested_cv_outer),
            seed,
            groups=groups,
            use_group_aware=cfg.use_group_aware_cv,
        )

        fold_values: list[float] = []
        if isinstance(outer, GroupKFold):
            if groups is None:
                continue
            split_iter = outer.split(X, y, groups)
        else:
            split_iter = outer.split(X, y)

        for train_idx, test_idx in split_iter:
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]
            g_tr = groups.iloc[train_idx] if groups is not None else None

            pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(estimator))])

            if tune_space:
                inner_cv = get_cv(
                    task,
                    y_tr,
                    max(2, cfg.nested_cv_inner),
                    seed + 13,
                    groups=g_tr,
                    use_group_aware=cfg.use_group_aware_cv,
                )

                dist = {f"model__{name}": space for name, space in tune_space.items()}
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=dist,
                    n_iter=min(8, cfg.tuning_iterations_complete),
                    scoring=scorer,
                    cv=inner_cv,
                    n_jobs=max(1, int(cfg.n_jobs)),
                    random_state=seed,
                    refit=True,
                    verbose=0,
                )
                if isinstance(inner_cv, GroupKFold) and g_tr is not None:
                    search.fit(X_tr, y_tr, groups=g_tr)
                else:
                    search.fit(X_tr, y_tr)
                best_pipe = search.best_estimator_
            else:
                best_pipe = pipe.fit(X_tr, y_tr)

            pred = best_pipe.predict(X_te)
            if task == "regression":
                fold_value = _rmse(np.asarray(y_te), np.asarray(pred))
            else:
                fold_value = float(f1_score(y_te, pred, average="weighted", zero_division=0))
            fold_values.append(fold_value)

        if fold_values:
            repeat_values.append(float(np.mean(fold_values)))

    if not repeat_values:
        return None

    arr = np.asarray(repeat_values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    half_width = float(1.96 * std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "ci_low": mean_val - half_width,
        "ci_high": mean_val + half_width,
        "n_repeats": float(len(arr)),
    }

def _prepare_sample_weight(sample_weight: pd.Series | None, index: pd.Index) -> np.ndarray | None:
    if sample_weight is None:
        return None

    aligned = pd.to_numeric(sample_weight.reindex(index), errors="coerce")
    arr = np.asarray(aligned, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)

    if not np.any(valid):
        return None

    fill = float(np.nanmedian(arr[valid]))
    if not np.isfinite(fill) or fill <= 0:
        fill = 1.0

    arr = np.where(valid, arr, fill)
    arr = arr / (float(np.mean(arr)) + 1e-12)
    return arr


def train_models(
    cfg: RunConfig,
    task: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger,
    groups_train: pd.Series | None = None,
    groups_all: pd.Series | None = None,
    definitions: list[ModelDefinition] | None = None,
    sample_weight_train: pd.Series | None = None,
) -> list[ModelResult]:
    if definitions is None:
        definitions = get_model_definitions(cfg, task, logger)

    if X_train.empty or X_test.empty:
        raise RuntimeError("Train/test features must be non-empty before model training.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise RuntimeError("X/y size mismatch detected before model training.")
    if not X_train.index.equals(y_train.index) or not X_test.index.equals(y_test.index):
        raise RuntimeError("X/y index mismatch detected before model training.")
    if not X_train.index.is_unique or not X_test.index.is_unique:
        raise RuntimeError("Feature indices must be unique for stable joins and analysis.")
    if task == "regression":
        y_train_num = pd.to_numeric(y_train, errors="coerce")
        y_test_num = pd.to_numeric(y_test, errors="coerce")
        if y_train_num.isna().any() or y_test_num.isna().any():
            raise RuntimeError("Regression target contains non-numeric or missing values at training stage.")
        train_target_is_constant = int(y_train_num.nunique(dropna=True)) < 2
    else:
        if y_train.astype("string").nunique(dropna=True) < 2:
            raise RuntimeError("Classification training requires at least 2 target classes.")
        train_target_is_constant = False

    cv = get_cv(
        task,
        y_train,
        cfg.cv_folds,
        cfg.random_state,
        groups=groups_train,
        use_group_aware=cfg.use_group_aware_cv,
    )
    scorer = get_primary_scorer(task)

    cv_signature = inspect.signature(cross_validate)
    cv_accepts_fit_params = "fit_params" in cv_signature.parameters
    cv_accepts_params = "params" in cv_signature.parameters

    sample_weight_vec = _prepare_sample_weight(sample_weight_train, X_train.index)
    training_n_jobs = max(1, int(cfg.n_jobs))

    results: list[ModelResult] = []

    for definition in definitions:
        if task == "regression" and train_target_is_constant and definition.key != "baseline":
            logger.warning("Skipping %s because training target is constant.", definition.label)
            continue

        if task == "regression" and definition.key == "catboost":
            fold_groups = groups_train if isinstance(cv, GroupKFold) else None
            if _has_constant_train_target_in_cv(cv, X_train, y_train, fold_groups):
                logger.warning(
                    "Skipping %s because at least one CV training fold has constant targets under current split.",
                    definition.label,
                )
                continue

        base_pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(definition.estimator))])
        fit_estimator: Any = base_pipe
        tune_prefix = "model__"
        applied_transform: str | None = None
        target_transformer: Any | None = None

        if task == "regression":
            transform_key = str(cfg.target_transform or "").strip().lower()
            y_numeric = pd.to_numeric(y_train, errors="coerce")

            if transform_key == "log1p" and y_numeric.notna().sum() > 0 and float(y_numeric.min()) > -1.0:
                fit_estimator = TransformedTargetRegressor(
                    regressor=base_pipe,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                )
                tune_prefix = "regressor__model__"
                applied_transform = "log1p"
            elif transform_key in {"yeo-johnson", "yeojohnson"}:
                target_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                fit_estimator = TransformedTargetRegressor(
                    regressor=base_pipe,
                    transformer=target_transformer,
                    check_inverse=False,
                )
                tune_prefix = "regressor__model__"
                applied_transform = "yeo-johnson"

        if applied_transform is not None:
            try:
                _ = clone(fit_estimator)
            except Exception as exc:
                logger.warning(
                    "Disabling target transform '%s' for %s (clone-incompatible estimator): %s",
                    applied_transform,
                    definition.label,
                    exc,
                )
                fit_estimator = base_pipe
                tune_prefix = "model__"
                applied_transform = None
                target_transformer = None

        def _fit_params_for_current_estimator() -> dict[str, Any] | None:
            if sample_weight_vec is None or not has_fit_parameter(definition.estimator, "sample_weight"):
                return None
            if applied_transform is not None:
                return {"regressor__model__sample_weight": sample_weight_vec}
            return {"model__sample_weight": sample_weight_vec}

        fit_params = _fit_params_for_current_estimator()

        start_cv = now_seconds()
        cv_kwargs: dict[str, Any] = {
            "cv": cv,
            "scoring": scorer,
            "n_jobs": training_n_jobs,
            "return_train_score": False,
            "error_score": np.nan,
        }
        if isinstance(cv, GroupKFold) and groups_train is not None:
            cv_kwargs["groups"] = groups_train
        if fit_params is not None:
            if cv_accepts_fit_params:
                cv_kwargs["fit_params"] = fit_params
            elif cv_accepts_params:
                cv_kwargs["params"] = fit_params
            else:
                raise RuntimeError("cross_validate does not support fit-parameter routing in this sklearn version.")

        try:
            cv_result = cross_validate(
                fit_estimator,
                X_train,
                y_train,
                **cv_kwargs,
            )
        except Exception as exc:
            if applied_transform is not None:
                logger.warning(
                    "Cross-validation failed for %s with target transform '%s'. Retrying without target transform. Error: %s",
                    definition.label,
                    applied_transform,
                    exc,
                )
                fit_estimator = base_pipe
                tune_prefix = "model__"
                applied_transform = None
                target_transformer = None
                fit_params = _fit_params_for_current_estimator()

                cv_kwargs_retry: dict[str, Any] = {
                    "cv": cv,
                    "scoring": scorer,
                    "n_jobs": training_n_jobs,
                    "return_train_score": False,
                    "error_score": np.nan,
                }
                if isinstance(cv, GroupKFold) and groups_train is not None:
                    cv_kwargs_retry["groups"] = groups_train
                if fit_params is not None:
                    if cv_accepts_fit_params:
                        cv_kwargs_retry["fit_params"] = fit_params
                    elif cv_accepts_params:
                        cv_kwargs_retry["params"] = fit_params
                    else:
                        raise RuntimeError("cross_validate does not support fit-parameter routing in this sklearn version.")

                try:
                    cv_result = cross_validate(
                        fit_estimator,
                        X_train,
                        y_train,
                        **cv_kwargs_retry,
                    )
                except Exception as exc_retry:
                    logger.warning("Cross-validation retry without target transform failed for %s; skipping model. Error: %s", definition.label, exc_retry)
                    continue
            else:
                logger.warning("Cross-validation failed for %s; skipping model. Error: %s", definition.label, exc)
                continue
        cv_seconds = now_seconds() - start_cv

        cv_scores_raw = np.asarray(cv_result["test_score"], dtype=float)
        cv_scores = cv_scores_raw[np.isfinite(cv_scores_raw)]
        if cv_scores.size == 0:
            logger.warning("Skipping %s because no valid CV scores were produced.", definition.label)
            continue
        if cv_scores.size < cv_scores_raw.size:
            logger.warning(
                "Model %s had %d/%d failed CV fold(s); metrics use valid folds only.",
                definition.label,
                int(cv_scores_raw.size - cv_scores.size),
                int(cv_scores_raw.size),
            )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        cv_primary = -cv_mean if task == "regression" else cv_mean

        best_params = None
        tune_seconds = 0.0
        tuned_model = fit_estimator

        if definition.tune_space and definition.key != "baseline":
            n_iter = cfg.tuning_iterations_complete if cfg.run_mode == "complete" else cfg.tuning_iterations_fast
            dist = {f"{tune_prefix}{key}": value for key, value in definition.tune_space.items()}
            search = RandomizedSearchCV(
                estimator=fit_estimator,
                param_distributions=dist,
                n_iter=n_iter,
                scoring=scorer,
                cv=cv,
                n_jobs=max(1, int(cfg.n_jobs)),
                random_state=cfg.random_state,
                refit=True,
                verbose=0,
            )
            start_tune = now_seconds()
            try:
                if isinstance(cv, GroupKFold) and groups_train is not None:
                    if fit_params is not None:
                        search.fit(X_train, y_train, groups=groups_train, **fit_params)
                    else:
                        search.fit(X_train, y_train, groups=groups_train)
                else:
                    if fit_params is not None:
                        search.fit(X_train, y_train, **fit_params)
                    else:
                        search.fit(X_train, y_train)
                tune_seconds = now_seconds() - start_tune
                tuned_model = search.best_estimator_
                best_params = search.best_params_
            except Exception as exc:
                tune_seconds = now_seconds() - start_tune
                logger.warning("Hyperparameter tuning failed for %s; using untuned model. Error: %s", definition.label, exc)
                tuned_model = fit_estimator
                best_params = None

        start_fit = now_seconds()
        try:
            if fit_params is not None:
                tuned_model.fit(X_train, y_train, **fit_params)
            else:
                tuned_model.fit(X_train, y_train)
            fit_seconds = now_seconds() - start_fit
        except Exception as exc:
            logger.warning("Skipping %s because final fit failed: %s", definition.label, exc)
            continue

        start_pred = now_seconds()
        y_train_pred = tuned_model.predict(X_train)
        y_test_pred = tuned_model.predict(X_test)
        y_train_proba = tuned_model.predict_proba(X_train) if hasattr(tuned_model, "predict_proba") else None
        y_test_proba = tuned_model.predict_proba(X_test) if hasattr(tuned_model, "predict_proba") else None
        predict_seconds = now_seconds() - start_pred

        if len(y_train_pred) != len(y_train) or len(y_test_pred) != len(y_test):
            raise RuntimeError(f"Prediction length mismatch for model {definition.label}.")

        if task == "regression":
            if not np.isfinite(np.asarray(y_train_pred, dtype=float)).all() or not np.isfinite(np.asarray(y_test_pred, dtype=float)).all():
                raise RuntimeError(f"Non-finite regression predictions detected for model {definition.label}.")
        else:
            if y_test_proba is not None:
                proba_arr = np.asarray(y_test_proba, dtype=float)
                if proba_arr.ndim != 2 or proba_arr.shape[0] != len(y_test):
                    raise RuntimeError(f"Classification probability shape mismatch for model {definition.label}.")

        if task == "regression":
            train_metrics = evaluate_regression_metrics(y_train, y_train_pred)
            test_metrics = evaluate_regression_metrics(y_test, y_test_pred)
        else:
            train_metrics = evaluate_classification_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = evaluate_classification_metrics(y_test, y_test_pred, y_test_proba)

        full_X = pd.concat([X_train, X_test], axis=0)
        full_y = pd.concat([y_train, y_test], axis=0)
        if groups_all is None and groups_train is not None:
            groups_all = groups_train.reindex(full_X.index)

        nested_summary = None
        if not (task == "regression" and applied_transform is not None):
            nested_summary = nested_cv_estimate(
                definition.estimator,
                preprocessor,
                full_X,
                full_y,
                task,
                definition.tune_space,
                cfg,
                groups_all,
            )

        lc_df = compute_learning_curve(
            tuned_model,
            X_train,
            y_train,
            task,
            cv,
            cfg.learning_curve_points,
            random_state=cfg.random_state,
            groups=groups_train,
            use_group_aware=cfg.use_group_aware_cv,
        )
        if lc_df.empty:
            logger.warning("Learning curve could not be computed for %s; continuing without it.", definition.label)

        total_seconds = cv_seconds + tune_seconds + fit_seconds + predict_seconds
        result = ModelResult(
            key=definition.key,
            label=definition.label,
            pipeline=tuned_model,
            cv_primary=cv_primary,
            cv_std=cv_std,
            nested_cv_primary=float(nested_summary["mean"]) if nested_summary else None,
            nested_cv_mean=float(nested_summary["mean"]) if nested_summary else None,
            nested_cv_std=float(nested_summary["std"]) if nested_summary else None,
            nested_cv_ci_low=float(nested_summary["ci_low"]) if nested_summary else None,
            nested_cv_ci_high=float(nested_summary["ci_high"]) if nested_summary else None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            fit_seconds=fit_seconds,
            cv_seconds=cv_seconds + tune_seconds,
            predict_seconds=predict_seconds,
            total_seconds=total_seconds,
            best_params=best_params,
            tune_space=definition.tune_space,
            learning_curve_df=lc_df,
            y_train_pred=np.asarray(y_train_pred),
            y_test_pred=np.asarray(y_test_pred),
            y_train_proba=np.asarray(y_train_proba) if y_train_proba is not None else None,
            y_test_proba=np.asarray(y_test_proba) if y_test_proba is not None else None,
            cv_scores=np.asarray(cv_scores, dtype=float),
            target_transform=applied_transform,
            target_transformer=target_transformer,
        )
        results.append(result)
        logger.info("Finished %s | cv_primary=%.4f | total_seconds=%.2f", result.label, result.cv_primary, result.total_seconds)

    return results

def choose_best_model(results: list[ModelResult], task: str) -> ModelResult:
    if task == "regression":
        return min(results, key=lambda item: item.cv_primary)
    return max(results, key=lambda item: item.cv_primary)


def rank_models_multi_objective(
    results: list[ModelResult],
    task: str,
    uncertainty_quality: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    perf_values = [item.cv_primary for item in results]
    runtime_values = [item.total_seconds for item in results]
    uncertainty_values = [uncertainty_quality.get(item.key, np.nan) for item in results]

    perf_norm = normalize_series(perf_values, reverse=(task != "regression"))
    runtime_norm = normalize_series(runtime_values, reverse=False)

    unc_filled = [0.5 if np.isnan(v) else float(v) for v in uncertainty_values]
    unc_norm = normalize_series(unc_filled, reverse=True)

    for idx, item in enumerate(results):
        score = 0.6 * perf_norm[idx] + 0.2 * runtime_norm[idx] + 0.2 * unc_norm[idx]
        rows.append(
            {
                "model": item.label,
                "model_key": item.key,
                "objective_score": float(score),
                "performance_component": float(perf_norm[idx]),
                "runtime_component": float(runtime_norm[idx]),
                "uncertainty_component": float(unc_norm[idx]),
                "cv_primary": float(item.cv_primary),
                "total_seconds": float(item.total_seconds),
            }
        )

    frame = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
    return frame


def statistical_model_comparison(
    task: str,
    results: list[ModelResult],
    y_test: pd.Series,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    if task == "regression":
        loss_map = {
            item.key: np.abs(np.asarray(y_test) - np.asarray(item.y_test_pred))
            for item in results
        }
        best_key = min(results, key=lambda item: item.test_metrics.get("rmse", np.inf)).key
    else:
        loss_map = {
            item.key: (np.asarray(y_test) != np.asarray(item.y_test_pred)).astype(float)
            for item in results
        }
        best_key = min(results, key=lambda item: item.test_metrics.get("log_loss", np.inf)).key

    best_loss = loss_map[best_key]
    out_rows: list[dict[str, float | str]] = []

    for item in results:
        current_loss = loss_map[item.key]
        observed_diff = float(np.mean(current_loss - best_loss))

        denom = float(np.std(current_loss - best_loss, ddof=1)) if len(current_loss) > 1 else 0.0
        effect_size = float(observed_diff / (denom + 1e-12))

        stats = []
        for _ in range(repeats):
            idx = rng.integers(0, len(best_loss), len(best_loss))
            diff = float(np.mean(current_loss[idx] - best_loss[idx]))
            stats.append(diff)

        stats_arr = np.asarray(stats)
        p_value = float(np.mean(np.abs(stats_arr) >= abs(observed_diff)))
        ci_low = float(np.quantile(stats_arr, 0.025)) if len(stats_arr) > 0 else float("nan")
        ci_high = float(np.quantile(stats_arr, 0.975)) if len(stats_arr) > 0 else float("nan")

        out_rows.append(
            {
                "model_key": item.key,
                "model": item.label,
                "best_model_key": best_key,
                "observed_loss_diff_vs_best": observed_diff,
                "bootstrap_p_value": p_value,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_diff_mean": float(np.mean(stats_arr)),
                "bootstrap_diff_std": float(np.std(stats_arr)),
                "effect_size_cohens_d": effect_size,
            }
        )

    return pd.DataFrame(out_rows).sort_values("bootstrap_p_value", ascending=True).reset_index(drop=True)



def permutation_model_comparison(
    task: str,
    results: list[ModelResult],
    y_test: pd.Series,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    if task == "regression":
        loss_map = {
            item.key: np.abs(np.asarray(y_test) - np.asarray(item.y_test_pred))
            for item in results
        }
    else:
        loss_map = {
            item.key: (np.asarray(y_test) != np.asarray(item.y_test_pred)).astype(float)
            for item in results
        }

    rows: list[dict[str, Any]] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            diff = np.asarray(loss_map[a.key] - loss_map[b.key], dtype=float)
            obs = float(np.mean(diff))

            perm_vals = np.empty(max(1, int(repeats)), dtype=float)
            for k in range(len(perm_vals)):
                signs = rng.choice(np.array([-1.0, 1.0]), size=len(diff), replace=True)
                perm_vals[k] = float(np.mean(diff * signs))

            p_val = float(np.mean(np.abs(perm_vals) >= abs(obs)))
            rows.append(
                {
                    "model_a": a.label,
                    "model_b": b.label,
                    "observed_mean_loss_diff_a_minus_b": obs,
                    "permutation_p_value": p_val,
                    "permutation_ci_low": float(np.quantile(perm_vals, 0.025)),
                    "permutation_ci_high": float(np.quantile(perm_vals, 0.975)),
                    "n_repeats": int(len(perm_vals)),
                }
            )

    return pd.DataFrame(rows)
def model_results_to_table(results: list[ModelResult], task: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in results:
        row: dict[str, Any] = {
            "model_key": item.key,
            "model": item.label,
            "cv_primary": item.cv_primary,
            "cv_std": item.cv_std,
            "nested_cv_primary": item.nested_cv_primary,
            "nested_cv_mean": item.nested_cv_mean,
            "nested_cv_std": item.nested_cv_std,
            "nested_cv_ci_low": item.nested_cv_ci_low,
            "nested_cv_ci_high": item.nested_cv_ci_high,
            "fit_seconds": item.fit_seconds,
            "cv_seconds": item.cv_seconds,
            "predict_seconds": item.predict_seconds,
            "total_seconds": item.total_seconds,
            "best_params": str(item.best_params) if item.best_params else "",
        }

        for k, v in item.train_metrics.items():
            row[f"train_{k}"] = finite_or_nan(v)
        for k, v in item.test_metrics.items():
            row[f"test_{k}"] = finite_or_nan(v)

        rows.append(row)

    frame = pd.DataFrame(rows)
    ascending = True if task == "regression" else False
    frame = frame.sort_values("cv_primary", ascending=ascending).reset_index(drop=True)
    return frame
















































def bayesian_correlated_ttest_comparison(
    task: str,
    results: list[ModelResult],
    rope: float = 0.0,
) -> pd.DataFrame:
    del task
    rows: list[dict[str, Any]] = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]

            sa = np.asarray(a.cv_scores, dtype=float) if a.cv_scores is not None else np.asarray([], dtype=float)
            sb = np.asarray(b.cv_scores, dtype=float) if b.cv_scores is not None else np.asarray([], dtype=float)

            n = int(min(len(sa), len(sb)))
            if n < 2:
                continue

            diff = sa[:n] - sb[:n]
            diff = diff[np.isfinite(diff)]
            n = int(len(diff))
            if n < 2:
                continue

            mean_diff = float(np.mean(diff))
            var_diff = float(np.var(diff, ddof=1)) if n > 1 else 0.0

            rho = 1.0 / max(2.0, float(n))
            corrected_var = var_diff * (1.0 / n + rho / max(1e-12, (1.0 - rho)))
            corrected_var = max(corrected_var, 1e-12)
            scale = float(np.sqrt(corrected_var))
            df = max(1, n - 1)

            p_a_better = float(1.0 - student_t.cdf(0.0, df=df, loc=mean_diff, scale=scale))
            p_b_better = float(student_t.cdf(0.0, df=df, loc=mean_diff, scale=scale))

            rope_val = max(0.0, float(rope))
            if rope_val > 0:
                p_rope = float(
                    student_t.cdf(rope_val, df=df, loc=mean_diff, scale=scale)
                    - student_t.cdf(-rope_val, df=df, loc=mean_diff, scale=scale)
                )
            else:
                p_rope = 0.0

            ci_low = float(student_t.ppf(0.025, df=df, loc=mean_diff, scale=scale))
            ci_high = float(student_t.ppf(0.975, df=df, loc=mean_diff, scale=scale))

            rows.append(
                {
                    "model_a": a.label,
                    "model_b": b.label,
                    "n_folds": int(n),
                    "mean_cv_score_diff_a_minus_b": mean_diff,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_a_better": p_a_better,
                    "p_b_better": p_b_better,
                    "p_rope": p_rope,
                    "rope": rope_val,
                }
            )

    return pd.DataFrame(rows)















