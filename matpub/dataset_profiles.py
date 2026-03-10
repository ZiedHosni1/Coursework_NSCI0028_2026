from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _base_config() -> dict[str, Any]:
    return {
        "dataset_name": "",
        "task_type": "regression",
        "target_col": None,
        "input_mode": "tabular",
        "drop_cols": [],
        "drop_col_patterns": [],
        "conditional_drop_cols": [],
        "row_filters": [],
        "raw_feature_cols": [],
        "categorical_cols": [],
        "numeric_cols": [],
        "special_nonfeature_cols": [],
        "group_split_col": None,
        "group_split_strategy": None,
        "formula_col": None,
        "formula_cols": [],
        "structure_col": None,
        "structure_cols": [],
        "dict_cols_to_flatten": [],
        "parse_formula": False,
        "use_composition_featurizers": True,
        "use_structure_featurizers": False,
        "use_site_element_featurization": False,
        "composition_featurizers": [],
        "structure_featurizers": [],
        "site_element_features": [],
        "engineered_pairwise_features": [],
        "post_featurization_drop_raw_cols": [],
        "post_flatten_drop_raw_cols": [],
        "target_transform": None,
        "outlier_cleaning": "none",
        "outlier_iqr_multiplier": 1.5,
        "outlier_zscore_threshold": 4.0,
        "imputation": "simple",
        "scaling": "standard",
        "uq_mode": "conformal",
        "dl_model": None,
        "model_keys": [],
        "target_family_cols": [],
        "target_specific_drop": {},
        "sample_weight_col": None,
        "sample_weight_mode": None,
        "site_element_cols": [],
        "enable_gaussian_process": False,
        "feature_log_patterns": [],
        "room_temperature_only": False,
    }

DATASET_CONFIG: dict[str, dict[str, Any]] = {}


def _add(name: str, updates: dict[str, Any]) -> None:
    cfg = _base_config()
    cfg.update(updates)
    cfg["dataset_name"] = name
    DATASET_CONFIG[name] = cfg


_add(
    "ricci_boltztrap_mp_tabular",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "categorical_cols": ["functional", "is_metal"],
        "target_col": "s.n [100K]",
        "target_family_cols": ["s.n [100K]", "s.n [300K]", "s.n [600K]", "pf.n [100K]", "pf.n [300K]", "pf.n [600K]"],
        "group_split_col": "material_id",
        "target_transform": None,
        "imputation": "simple",
        "scaling": "standard",
        "feature_log_patterns": ["carrier", "concentration"],
        "model_keys": ["ridge", "random_forest", "xgboost", "catboost", "ann"],
    },
)

_add(
    "boltztrap_mp",
    {
        "task_type": "regression",
        "input_mode": "hybrid",
        "target_col": "s.n [300K]",
        "drop_cols": ["mpid"],
        "parse_formula": True,
        "formula_col": "formula",
        "structure_col": "structure",
        "group_split_strategy": "reduced_formula",
        "use_structure_featurizers": True,
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital"],
        "structure_featurizers": ["DensityFeatures", "GlobalSymmetryFeatures", "SiteStatsFingerprint", "PRDF", "JarvisCFID"],
        "model_keys": ["xgboost", "catboost", "random_forest", "ann"],
        "dl_model": "m3gnet",
    },
)

_add(
    "ucsb_thermoelectrics",
    {
        "task_type": "regression",
        "input_mode": "composition",
        "target_col": "zT",
        "parse_formula": True,
        "formula_col": "formula",
        "categorical_cols": ["crystallinity", "synthesis", "spacegroup"],
        "group_split_strategy": "reduced_formula",
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital", "IonProperty"],
        "target_family_cols": ["zT", "PF", "rho", "sigma", "S", "kappa"],
        "model_keys": ["catboost", "xgboost", "random_forest", "elastic_net", "ann"],
    },
)

_add(
    "citrine_thermal_conductivity",
    {
        "task_type": "regression",
        "input_mode": "composition",
        "target_col": "k_expt",
        "parse_formula": True,
        "formula_col": "formula",
        "group_split_strategy": "reduced_formula",
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital"],
        "target_transform": "log1p",
        "room_temperature_only": True,
        "model_keys": ["catboost", "xgboost", "random_forest", "svm", "ann"],
    },
)

_add(
    "double_perovskites_gap",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "target_col": "gap gllbsc",
        "site_element_cols": ["a_1", "b_1", "a_2", "b_2"],
        "drop_cols": ["formula"],
        "model_keys": ["catboost", "xgboost", "random_forest", "svm", "ann"],
    },
)

_add(
    "double_perovskites_gap_lumo",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "target_col": "lumo",
        "site_element_cols": ["atom"],
        "imputation": "simple",
        "scaling": "standard",
        "enable_gaussian_process": True,
        "model_keys": ["ridge", "lasso", "svm", "gaussian_process", "xgboost"],
    },
)

_add(
    "wolverton_oxides",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "target_col": "gap pbe",
        "site_element_cols": ["atom a", "atom b"],
        "target_family_cols": ["e_form", "e_hull", "gap pbe", "mu_b"],
        "group_split_strategy": "chem_family_ab",
        "model_keys": ["catboost", "xgboost", "random_forest", "elastic_net"],
    },
)

_add(
    "flla",
    {
        "task_type": "regression",
        "input_mode": "hybrid",
        "target_col": "formation_energy_per_atom",
        "parse_formula": True,
        "formula_col": "formula",
        "structure_col": "structure",
        "use_structure_featurizers": True,
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital"],
        "structure_featurizers": ["DensityFeatures", "GlobalSymmetryFeatures", "SiteStatsFingerprint", "PRDF"],
        "group_split_strategy": "reduced_formula",
        "target_specific_drop": {
            "formation_energy_per_atom": ["formation_energy"],
            "e_above_hull": ["formation_energy", "formation_energy_per_atom"],
        },
        "model_keys": ["xgboost", "catboost", "random_forest", "ann"],
        "dl_model": "m3gnet",
    },
)

_add(
    "brgoch_superhard_training",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "target_col": "bulk_modulus",
        "dict_cols_to_flatten": ["brgoch_feats"],
        "drop_cols": ["material_id"],
        "categorical_cols": ["suspect_value"],
        "model_keys": ["catboost", "xgboost", "random_forest", "elastic_net", "ann"],
    },
)

_add(
    "m2ax",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "target_col": "c11",
        "numeric_cols": ["a", "c", "d_mx", "d_ma"],
        "parse_formula": True,
        "formula_col": "formula",
        "group_split_strategy": "reduced_formula",
        "enable_gaussian_process": True,
        "model_keys": ["svm", "gaussian_process", "xgboost", "catboost", "ridge"],
    },
)

_add(
    "steel_strength",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "target_col": "strength",
        "drop_cols": ["formula"],
        "enable_gaussian_process": True,
        "model_keys": ["catboost", "xgboost", "random_forest", "elastic_net", "svm", "gaussian_process"],
    },
)

_add(
    "heusler_magnetic",
    {
        "task_type": "regression",
        "input_mode": "tabular",
        "target_col": "mu_b",
        "categorical_cols": ["heusler_type", "struct_type"],
        "parse_formula": True,
        "formula_col": "formula",
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital"],
        "target_specific_drop": {
            "mu_b": ["mu_b_saturation"],
            "mu_b_saturation": ["mu_b"],
        },
        "model_keys": ["catboost", "xgboost", "random_forest", "elastic_net", "ann"],
    },
)

_add(
    "expt_formation_enthalpy",
    {
        "task_type": "regression",
        "input_mode": "composition",
        "target_col": "e_form expt",
        "parse_formula": True,
        "formula_col": "formula",
        "categorical_cols": ["pearson_symbol", "space_group"],
        "drop_cols": ["mpid", "oqmdid"],
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital"],
        "model_keys": ["catboost", "xgboost", "random_forest", "elastic_net"],
    },
)

_add(
    "expt_formation_enthalpy_kingsbury",
    {
        "task_type": "regression",
        "input_mode": "composition",
        "target_col": "expt_form_e",
        "parse_formula": True,
        "formula_col": "formula",
        "categorical_cols": ["phaseinfo"],
        "drop_cols": ["reference"],
        "sample_weight_col": "uncertainty",
        "sample_weight_mode": "inverse",
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital"],
        "model_keys": ["catboost", "xgboost", "random_forest", "elastic_net"],
    },
)

_add(
    "glass_binary_v2",
    {
        "task_type": "classification",
        "input_mode": "composition",
        "target_col": "gfa",
        "parse_formula": True,
        "formula_col": "formula",
        "group_split_strategy": "binary_system",
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital", "Miedema", "WenAlloys", "YangSolidSolution"],
        "model_keys": ["catboost", "xgboost", "random_forest", "logistic", "ann"],
    },
)

_add(
    "glass_ternary_hipt",
    {
        "task_type": "classification",
        "input_mode": "composition",
        "target_col": "gfa",
        "parse_formula": True,
        "formula_col": "formula",
        "categorical_cols": ["processing"],
        "drop_cols": ["phase"],
        "group_split_strategy": "chemical_system",
        "composition_featurizers": ["ElementProperty", "Stoichiometry", "ElementFraction", "ValenceOrbital", "Miedema", "WenAlloys", "YangSolidSolution"],
        "model_keys": ["catboost", "xgboost", "random_forest", "logistic", "ann"],
    },
)


def _normalize_external_group_strategy(raw: str | None) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    key = str(raw).strip().lower()
    if key in {"reduced_formula", "reduced_composition"}:
        return None, "reduced_formula"
    if key in {"chemical_system", "binary_system", "binary_system_optional", "system"}:
        return None, "chemical_system"
    if key in {"a_b_family", "chem_family_ab", "chemistry_family_optional", "heusler_family_optional"}:
        return None, "chem_family_ab"
    if key in {"reduced_formula_or_structure_hash"}:
        return None, "reduced_formula"
    if key in {"index_or_material_id"}:
        return "material_id", None
    return str(raw), None


def _convert_external_profile(name: str, ext_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(DATASET_CONFIG.get(name, _base_config()))
    cfg["dataset_name"] = name

    cfg["task_type"] = str(ext_cfg.get("task_type", cfg.get("task_type", "regression")))
    cfg["target_col"] = ext_cfg.get("target_col", cfg.get("target_col"))
    cfg["input_mode"] = str(ext_cfg.get("input_mode", cfg.get("input_mode", "tabular")))
    cfg["drop_cols"] = [str(v) for v in ext_cfg.get("drop_cols", cfg.get("drop_cols", [])) or []]
    cfg["drop_col_patterns"] = [str(v) for v in ext_cfg.get("drop_col_patterns", cfg.get("drop_col_patterns", [])) or []]
    cfg["conditional_drop_cols"] = [str(v) for v in ext_cfg.get("conditional_drop_cols", cfg.get("conditional_drop_cols", [])) or []]
    cfg["row_filters"] = [str(v) for v in ext_cfg.get("row_filters", cfg.get("row_filters", [])) or []]
    cfg["raw_feature_cols"] = [str(v) for v in ext_cfg.get("raw_feature_cols", cfg.get("raw_feature_cols", [])) or []]
    cfg["categorical_cols"] = [str(v) for v in ext_cfg.get("categorical_cols", cfg.get("categorical_cols", [])) or []]
    cfg["numeric_cols"] = [str(v) for v in ext_cfg.get("numeric_cols", cfg.get("numeric_cols", [])) or []]
    cfg["special_nonfeature_cols"] = [str(v) for v in ext_cfg.get("special_nonfeature_cols", cfg.get("special_nonfeature_cols", [])) or []]

    fcols = [str(v) for v in ext_cfg.get("formula_cols", cfg.get("formula_cols", [])) or []]
    scols = [str(v) for v in ext_cfg.get("structure_cols", cfg.get("structure_cols", [])) or []]
    cfg["formula_cols"] = fcols
    cfg["structure_cols"] = scols
    cfg["formula_col"] = fcols[0] if fcols else ext_cfg.get("formula_col", cfg.get("formula_col"))
    cfg["structure_col"] = scols[0] if scols else ext_cfg.get("structure_col", cfg.get("structure_col"))

    cfg["dict_cols_to_flatten"] = [str(v) for v in ext_cfg.get("dict_cols_to_flatten", cfg.get("dict_cols_to_flatten", [])) or []]
    cfg["parse_formula"] = bool(ext_cfg.get("parse_formula", cfg.get("parse_formula", False)))

    cfg["use_composition_featurizers"] = bool(ext_cfg.get("use_composition_featurizers", cfg.get("use_composition_featurizers", True)))
    cfg["use_structure_featurizers"] = bool(ext_cfg.get("use_structure_featurizers", cfg.get("use_structure_featurizers", False)))
    cfg["use_site_element_featurization"] = bool(ext_cfg.get("use_site_element_featurization", cfg.get("use_site_element_featurization", False)))

    cfg["composition_featurizers"] = [str(v) for v in ext_cfg.get("composition_featurizers", cfg.get("composition_featurizers", [])) or []]
    cfg["structure_featurizers"] = [str(v) for v in ext_cfg.get("structure_featurizers", cfg.get("structure_featurizers", [])) or []]

    cfg["site_element_cols"] = [str(v) for v in ext_cfg.get("site_element_cols", cfg.get("site_element_cols", [])) or []]
    cfg["site_element_features"] = [str(v) for v in ext_cfg.get("site_element_features", cfg.get("site_element_features", [])) or []]
    cfg["engineered_pairwise_features"] = [str(v) for v in ext_cfg.get("engineered_pairwise_features", cfg.get("engineered_pairwise_features", [])) or []]

    cfg["post_featurization_drop_raw_cols"] = [str(v) for v in ext_cfg.get("post_featurization_drop_raw_cols", cfg.get("post_featurization_drop_raw_cols", [])) or []]
    cfg["post_flatten_drop_raw_cols"] = [str(v) for v in ext_cfg.get("post_flatten_drop_raw_cols", cfg.get("post_flatten_drop_raw_cols", [])) or []]

    cfg["target_transform"] = ext_cfg.get("target_transform", cfg.get("target_transform"))
    cfg["outlier_cleaning"] = str(ext_cfg.get("outlier_cleaning", cfg.get("outlier_cleaning", "none")))
    cfg["outlier_iqr_multiplier"] = float(ext_cfg.get("outlier_iqr_multiplier", cfg.get("outlier_iqr_multiplier", 1.5)))
    cfg["outlier_zscore_threshold"] = float(ext_cfg.get("outlier_zscore_threshold", cfg.get("outlier_zscore_threshold", 4.0)))
    cfg["imputation"] = str(ext_cfg.get("imputation", cfg.get("imputation", "simple")))
    cfg["scaling"] = str(ext_cfg.get("scaling", cfg.get("scaling", "standard")))
    cfg["uq_mode"] = str(ext_cfg.get("uq_mode", cfg.get("uq_mode", "conformal")))
    cfg["dl_model"] = ext_cfg.get("dl_model", cfg.get("dl_model"))
    cfg["model_keys"] = [str(v) for v in ext_cfg.get("model_keys", cfg.get("model_keys", [])) or []]
    cfg["enable_gaussian_process"] = bool(ext_cfg.get("enable_gaussian_process", cfg.get("enable_gaussian_process", False)))
    cfg["feature_log_patterns"] = [str(v) for v in ext_cfg.get("feature_log_patterns", cfg.get("feature_log_patterns", [])) or []]
    cfg["room_temperature_only"] = bool(ext_cfg.get("room_temperature_only", cfg.get("room_temperature_only", False)))

    sw = ext_cfg.get("sample_weight_col")
    if isinstance(sw, str) and sw.lower() == "uncertainty_inverse_optional":
        cfg["sample_weight_col"] = "uncertainty"
        cfg["sample_weight_mode"] = "inverse"
    else:
        cfg["sample_weight_col"] = sw if sw is not None else cfg.get("sample_weight_col")
        cfg["sample_weight_mode"] = ext_cfg.get("sample_weight_mode", cfg.get("sample_weight_mode"))

    raw_group = ext_cfg.get("group_split_col", ext_cfg.get("group_split_strategy", cfg.get("group_split_col")))
    gcol, gstrategy = _normalize_external_group_strategy(raw_group)
    cfg["group_split_col"] = gcol
    cfg["group_split_strategy"] = gstrategy or cfg.get("group_split_strategy")

    if ext_cfg.get("target_family_cols"):
        cfg["target_family_cols"] = [str(v) for v in ext_cfg.get("target_family_cols", [])]
    if ext_cfg.get("target_specific_drop"):
        cfg["target_specific_drop"] = ext_cfg.get("target_specific_drop", {})

    return cfg
def load_external_dataset_config_map(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")

    raw_map: dict[str, Any] | None = None

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            if isinstance(loaded.get("DATASET_CONFIGS"), dict):
                raw_map = loaded["DATASET_CONFIGS"]
            elif isinstance(loaded.get("DATASET_CONFIG"), dict):
                raw_map = loaded["DATASET_CONFIG"]
            elif all(isinstance(v, dict) for v in loaded.values()):
                raw_map = loaded
    except Exception:
        raw_map = None

    if raw_map is None:
        tree = ast.parse(text, filename=str(p))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"DATASET_CONFIGS", "DATASET_CONFIG"}:
                        raw_map = ast.literal_eval(node.value)
                        break

    if raw_map is None or not isinstance(raw_map, dict):
        raise RuntimeError(f"Could not find DATASET_CONFIGS mapping in {p}")

    out: dict[str, dict[str, Any]] = {}
    for name, cfg in raw_map.items():
        if isinstance(name, str) and isinstance(cfg, dict):
            out[name] = _convert_external_profile(name, cfg)
    return out
def get_dataset_config(dataset_name: str | None, external_config_file: str | None = None) -> dict[str, Any] | None:
    if dataset_name is None:
        return None

    if external_config_file:
        try:
            ext = load_external_dataset_config_map(external_config_file)
            if dataset_name in ext:
                return deepcopy(ext[dataset_name])
            lower = {k.lower(): k for k in ext}
            hit = lower.get(str(dataset_name).lower())
            if hit is not None:
                return deepcopy(ext[hit])
        except Exception:
            pass

    key = dataset_name.strip()
    if key in DATASET_CONFIG:
        return deepcopy(DATASET_CONFIG[key])

    lower_map = {name.lower(): name for name in DATASET_CONFIG}
    hit = lower_map.get(key.lower())
    if hit is None:
        return None
    return deepcopy(DATASET_CONFIG[hit])


def resolve_target_specific_drops(profile: dict[str, Any], target_col: str) -> list[str]:
    drops = list(profile.get("drop_cols", []))

    target_family = profile.get("target_family_cols", []) or []
    for col in target_family:
        if col != target_col:
            drops.append(col)

    extra = profile.get("target_specific_drop", {}) or {}
    drops.extend(extra.get(target_col, []))

    return sorted(set([str(item) for item in drops if str(item)]))





