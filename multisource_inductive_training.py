"""
Train a drug-inductive model on harmonized GDSC plus optional CTRPv2/CCLE data.

This is intended for the reviewer-requested generalization experiment:
- no drug identity feature
- rich transferable drug features
- leave-drug-out model selection
- optional external datasets included in training rather than only validation
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from benchmarking import (
    FeatureConfig,
    ReviewerBenchmarkSuite,
    calibration_summary,
    grouped_regression_metrics,
    regression_metrics,
)
from external_validation import ExternalValidationRunner


def _safe_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _safe_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def build_inductive_feature_config(
    fingerprint_bits: int,
    fingerprint_radius: int,
    include_omics: bool,
) -> FeatureConfig:
    return FeatureConfig(
        include_expression=True,
        include_target=True,
        include_pathway=True,
        include_drug_identity=False,
        include_fingerprints=True,
        include_structure_missing_flag=True,
        include_rdkit_descriptors=True,
        include_mechanism_multihot=True,
        include_target_expression_interactions=True,
        include_mutations=include_omics,
        include_copy_number=include_omics,
        include_fusions=include_omics,
        include_rppa=include_omics,
        include_tissue=True,
        top_genes=1000,
        fingerprint_bits=fingerprint_bits,
        fingerprint_radius=fingerprint_radius,
        scale_numeric=False,
    )


def normalize_external_frame(
    frame: pd.DataFrame,
    dataset_name: str,
    response_values: pd.Series,
    model_mapping: pd.DataFrame,
) -> pd.DataFrame:
    tissue_cols = ["ModelID", "OncotreeLineage", "OncotreePrimaryDisease"]
    tissue = model_mapping[tissue_cols].drop_duplicates("ModelID")

    normalized = pd.DataFrame(
        {
            "DRUG_NAME": frame["drug_name"].astype(str),
            "ModelID": frame["ModelID"].astype(str),
            "PUTATIVE_TARGET": frame["target"].fillna("Unknown").astype(str),
            "PATHWAY_NAME": frame["pathway"].fillna("Unknown").astype(str),
            "AUC": response_values.astype(float),
            "dataset_source": dataset_name,
        }
    )
    normalized = normalized.merge(tissue, on="ModelID", how="left")
    normalized["SANGER_MODEL_ID"] = normalized["ModelID"]
    normalized["CELL_LINE_NAME"] = frame["cell_line_name"].astype(str)
    normalized["OncotreeLineage"] = normalized["OncotreeLineage"].fillna("Unknown")
    normalized["OncotreePrimaryDisease"] = normalized["OncotreePrimaryDisease"].fillna("Unknown")
    return normalized


def first_broad_id(value: object) -> str:
    if pd.isna(value):
        return ""
    for part in str(value).replace("|", ";").replace(",", ";").split(";"):
        token = part.strip()
        if token.startswith("BRD:"):
            token = token.split("BRD:", 1)[1]
        if token.startswith("BRD-"):
            return token
    return str(value).strip()


def dataset_sample_weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    """Return per-row weights so each source can contribute comparable total mass."""
    if mode == "none":
        return None
    if mode not in {"equal_dataset", "sqrt_inverse_dataset"}:
        raise ValueError(f"Unsupported source weighting mode: {mode}")

    counts = frame["dataset_source"].value_counts()
    n_sources = max(len(counts), 1)
    if mode == "equal_dataset":
        weights = frame["dataset_source"].map(lambda source: len(frame) / (n_sources * counts[source]))
    else:
        weights = frame["dataset_source"].map(lambda source: np.sqrt(len(frame) / (n_sources * counts[source])))
        weights = weights / weights.mean()
    return weights.to_numpy(dtype=np.float32)


def weighted_source_summary(frame: pd.DataFrame, weights: np.ndarray | None) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    if weights is None:
        for source, count in frame["dataset_source"].value_counts().items():
            rows[str(source)] = {"rows": int(count), "weight_sum": float(count), "mean_weight": 1.0}
        return rows

    weighted = frame[["dataset_source"]].copy()
    weighted["weight"] = weights
    for source, group in weighted.groupby("dataset_source"):
        rows[str(source)] = {
            "rows": int(len(group)),
            "weight_sum": float(group["weight"].sum()),
            "mean_weight": float(group["weight"].mean()),
        }
    return rows


def balanced_dataset_r2(predictions: pd.DataFrame) -> float:
    """Mean per-dataset R2. This prevents large datasets from dominating tuning."""
    scores: List[float] = []
    for _, group in predictions.groupby("dataset_source"):
        if len(group) < 10 or group["AUC"].nunique() < 2:
            continue
        score = regression_metrics(group["AUC"], group["prediction"])["r2"]
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


def fit_source_calibrators(frame: pd.DataFrame, pred_col: str = "prediction") -> Dict[str, Dict[str, float]]:
    calibrators: Dict[str, Dict[str, float]] = {}
    for source, group in frame.groupby("dataset_source"):
        if len(group) >= 10 and group[pred_col].nunique() >= 2:
            slope, intercept = np.polyfit(group[pred_col].astype(float), group["AUC"].astype(float), deg=1)
        else:
            slope, intercept = 1.0, 0.0
        calibrators[str(source)] = {"slope": float(slope), "intercept": float(intercept)}
    return calibrators


def apply_source_calibrators(frame: pd.DataFrame, calibrators: Dict[str, Dict[str, float]]) -> np.ndarray:
    calibrated = []
    for _, row in frame.iterrows():
        params = calibrators.get(str(row["dataset_source"]), {"slope": 1.0, "intercept": 0.0})
        calibrated.append(float(params["slope"]) * float(row["prediction"]) + float(params["intercept"]))
    return np.asarray(calibrated, dtype=np.float32)


class MultiSourceInductiveTrainer:
    def __init__(
        self,
        data_dir: str,
        new_data_dir: str,
        output_dir: str,
        device: str,
        random_state: int,
    ) -> None:
        self.suite = ReviewerBenchmarkSuite(data_dir=data_dir, output_dir=output_dir)
        self.external = ExternalValidationRunner(data_dir=data_dir, new_data_dir=new_data_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.random_state = random_state

    def tune_xgboost_balanced(
        self,
        train_df: pd.DataFrame,
        config: FeatureConfig,
        source_weighting: str,
        validation_size: float = 0.2,
    ) -> tuple[Dict[str, object], pd.DataFrame]:
        inner_train_df, inner_valid_df, _ = self.suite.split_response(
            train_df,
            split_strategy="leave_drug_out",
            test_size=validation_size,
            random_state=self.random_state + 1000,
        )
        train_x, valid_x, _ = self.suite.prepare_features(inner_train_df, inner_valid_df, config)
        train_weights = dataset_sample_weights(inner_train_df, source_weighting)
        y_train = inner_train_df["AUC"].to_numpy(dtype=float)
        y_valid = inner_valid_df["AUC"].to_numpy(dtype=float)

        param_grid: List[Dict[str, object]] = [
            {
                "n_estimators": 350,
                "max_depth": 4,
                "learning_rate": 0.04,
                "min_child_weight": 8,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.5,
                "reg_lambda": 3.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.035,
                "min_child_weight": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.25,
                "reg_lambda": 2.0,
            },
            {
                "n_estimators": 700,
                "max_depth": 4,
                "learning_rate": 0.025,
                "min_child_weight": 10,
                "subsample": 0.9,
                "colsample_bytree": 0.65,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
            {
                "n_estimators": 450,
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_child_weight": 12,
                "subsample": 0.9,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 4.0,
            },
        ]

        rows: List[Dict[str, object]] = []
        best_params: Dict[str, object] = {}
        best_score = -np.inf
        for idx, params in enumerate(param_grid, start=1):
            print(f"Tuning weighted XGBoost candidate {idx}/{len(param_grid)} on inner leave-drug-out split...")
            model = self.suite._build_model(device=self.device, random_state=self.random_state + idx, params=params)
            fit_kwargs = {"sample_weight": train_weights} if train_weights is not None else {}
            model.fit(train_x, y_train, **fit_kwargs)
            pred = model.predict(valid_x)
            metrics = regression_metrics(y_valid, pred)
            validation_predictions = inner_valid_df[["dataset_source", "AUC"]].copy()
            validation_predictions["prediction"] = pred
            balanced_r2 = balanced_dataset_r2(validation_predictions)
            row = {
                "candidate": idx,
                **params,
                **metrics,
                "balanced_dataset_r2": balanced_r2,
            }
            rows.append(row)
            selection_score = balanced_r2 if not np.isnan(balanced_r2) else metrics["r2"]
            if selection_score > best_score:
                best_score = selection_score
                best_params = params

        return best_params, pd.DataFrame(rows).sort_values("balanced_dataset_r2", ascending=False).reset_index(drop=True)

    def load_response_pool(
        self,
        include_external: Sequence[str],
        min_samples_per_drug: int,
        max_external_rows: int | None,
    ) -> pd.DataFrame:
        self.suite.load()
        assert self.suite.response_df is not None
        assert self.suite.expression_df is not None
        assert self.suite.pipeline is not None
        assert self.suite.pipeline.model_mapping is not None

        gdsc = self.suite.filtered_response(min_samples_per_drug)
        gdsc["dataset_source"] = "GDSC"

        self.external.expression_df = self.suite.expression_df
        self.external.model_map = self.external._build_model_id_map()

        external_frames: List[pd.DataFrame] = []
        smiles_frames: List[pd.DataFrame] = []
        for dataset in include_external:
            dataset_key = dataset.lower()
            if dataset_key == "ctrp":
                raw = self.external.load_ctrp()
                endpoint = self.external.endpoint_transform(raw, "raw_response", higher_raw_is_resistant=True)
            elif dataset_key == "ccle":
                raw = self.external.load_ccle()
                endpoint = self.external.endpoint_transform(raw, "raw_response", higher_raw_is_resistant=False)
            elif dataset_key == "prism":
                raw = self.load_prism()
                endpoint = self.external.endpoint_transform(raw, "raw_response", higher_raw_is_resistant=True)
            else:
                raise ValueError(f"Unsupported external dataset: {dataset}")

            if max_external_rows is not None and len(raw) > max_external_rows:
                raw = raw.sample(n=max_external_rows, random_state=self.random_state).reset_index(drop=True)
                endpoint = self.external.endpoint_transform(
                    raw,
                    "raw_response",
                    higher_raw_is_resistant=(dataset_key in {"ctrp", "prism"}),
                )

            response = normalize_external_frame(
                frame=raw,
                dataset_name=dataset_key.upper(),
                response_values=endpoint.transform(raw["raw_response"]),
                model_mapping=self.suite.pipeline.model_mapping,
            )
            external_frames.append(response)
            smiles_frames.append(raw[["drug_name", "smiles"]].rename(columns={"drug_name": "DRUG_NAME", "smiles": "SMILES"}))
            print(f"Added {len(response):,} harmonized {dataset_key.upper()} rows to the training pool")

        if smiles_frames:
            self.suite.add_smiles_records(pd.concat(smiles_frames, ignore_index=True))

        frames = [gdsc] + external_frames
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined[combined["AUC"].notna() & combined["ModelID"].isin(self.suite.expression_df.index)].copy()
        combined["PUTATIVE_TARGET"] = combined["PUTATIVE_TARGET"].fillna("Unknown")
        combined["PATHWAY_NAME"] = combined["PATHWAY_NAME"].fillna("Unknown")
        combined["OncotreeLineage"] = combined["OncotreeLineage"].fillna("Unknown")
        combined["OncotreePrimaryDisease"] = combined["OncotreePrimaryDisease"].fillna("Unknown")
        return combined.reset_index(drop=True)

    def load_prism(self) -> pd.DataFrame:
        base = self.external.new_data_dir / "PRISM"
        matrix_path = base / "Repurposing_Public_23Q2_Extended_Primary_Data_Matrix.csv"
        compound_path = base / "Repurposing_Public_23Q2_Extended_Primary_Compound_List.csv"
        treatment_path = base / "Repurposing_Public_23Q2_Treatment_Meta_Data.csv"
        old_treatment_path = base / "primary-screen-replicate-collapsed-treatment-info.csv"

        matrix = pd.read_csv(matrix_path, low_memory=False)
        id_col = matrix.columns[0]
        matrix = matrix.rename(columns={id_col: "broad_id"})
        matrix["broad_id"] = matrix["broad_id"].map(first_broad_id)
        long = matrix.melt(id_vars="broad_id", var_name="ModelID", value_name="raw_response")
        long = long[long["raw_response"].notna()].copy()

        compounds = pd.read_csv(compound_path, low_memory=False)
        compounds["broad_id"] = compounds["IDs"].map(first_broad_id)
        compounds = compounds.rename(
            columns={
                "Drug.Name": "drug_name",
                "repurposing_target": "target",
                "MOA": "pathway",
            }
        )
        compounds = compounds[["broad_id", "drug_name", "target", "pathway"]].drop_duplicates("broad_id")

        smiles_lookup: Dict[str, str] = {}
        if treatment_path.exists():
            treatment = pd.read_csv(treatment_path, low_memory=False)
            if {"broad_id", "name"}.issubset(treatment.columns):
                for _, row in treatment.dropna(subset=["broad_id", "name"]).iterrows():
                    smiles_lookup.setdefault(str(row["broad_id"]), "")
        if old_treatment_path.exists():
            old_treatment = pd.read_csv(old_treatment_path, low_memory=False)
            if {"broad_id", "smiles"}.issubset(old_treatment.columns):
                for _, row in old_treatment.dropna(subset=["broad_id", "smiles"]).iterrows():
                    smiles_lookup.setdefault(str(row["broad_id"]), str(row["smiles"]).strip())

        frame = long.merge(compounds, on="broad_id", how="left")
        frame["drug_name"] = frame["drug_name"].fillna(frame["broad_id"])
        frame["target"] = frame["target"].fillna("Unknown")
        frame["pathway"] = frame["pathway"].fillna("Unknown")
        frame["smiles"] = frame["broad_id"].map(smiles_lookup)
        frame["cell_line_name"] = frame["ModelID"]
        frame["dataset"] = "PRISM"
        frame["response_type"] = "PRISM_23Q2_LFC"
        return frame[frame["ModelID"].notna() & frame["raw_response"].notna()].copy()

    def train(
        self,
        include_external: Sequence[str],
        split_strategy: str,
        min_samples_per_drug: int,
        test_size: float,
        max_external_rows: int | None,
        fingerprint_bits: int,
        fingerprint_radius: int,
        include_omics: bool,
        tune_xgboost: bool,
        shap_samples: int,
        source_weighting: str,
    ) -> Dict[str, object]:
        config = build_inductive_feature_config(fingerprint_bits, fingerprint_radius, include_omics)
        response = self.load_response_pool(include_external, min_samples_per_drug, max_external_rows)
        train_df, test_df, split_meta = self.suite.split_response(
            response,
            split_strategy=split_strategy,
            test_size=test_size,
            random_state=self.random_state,
        )

        train_x, test_x, feature_names = self.suite.prepare_features(train_df, test_df, config)
        best_params: Dict[str, object] | None = None
        tuning_results = pd.DataFrame()
        if tune_xgboost:
            best_params, tuning_results = self.tune_xgboost_balanced(
                train_df=train_df,
                config=config,
                source_weighting=source_weighting,
            )

        model = self.suite._build_model(device=self.device, random_state=self.random_state, params=best_params)
        train_weights = dataset_sample_weights(train_df, source_weighting)
        fit_kwargs = {"sample_weight": train_weights} if train_weights is not None else {}
        model.fit(train_x, train_df["AUC"].to_numpy(dtype=float), **fit_kwargs)
        pred = model.predict(test_x)
        train_pred = model.predict(train_x)
        y_test = test_df["AUC"].to_numpy(dtype=float)
        metrics = regression_metrics(y_test, pred)
        calibration_stats, calibration_table = calibration_summary(y_test, pred)

        train_predictions = train_df[["dataset_source", "AUC"]].copy()
        train_predictions["prediction"] = train_pred
        source_calibrators = fit_source_calibrators(train_predictions)

        predictions = test_df[
            ["dataset_source", "DRUG_NAME", "ModelID", "OncotreeLineage", "OncotreePrimaryDisease", "AUC"]
        ].copy()
        predictions["prediction"] = pred
        predictions["prediction_source_calibrated"] = apply_source_calibrators(predictions, source_calibrators)
        per_drug = grouped_regression_metrics(predictions, "DRUG_NAME", min_samples=10)
        per_dataset = grouped_regression_metrics(predictions, "dataset_source", min_samples=10)
        balanced_r2 = balanced_dataset_r2(predictions)
        calibrated_metrics = regression_metrics(predictions["AUC"], predictions["prediction_source_calibrated"])
        calibrated_predictions = predictions.rename(columns={"prediction": "raw_prediction"}).rename(
            columns={"prediction_source_calibrated": "prediction"}
        )
        per_dataset_calibrated = grouped_regression_metrics(calibrated_predictions, "dataset_source", min_samples=10)
        balanced_calibrated_r2 = balanced_dataset_r2(calibrated_predictions)

        out_dir = self.output_dir / "multisource_inductive"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "feature_config": config.__dict__,
            "include_external": list(include_external),
            "split_strategy": split_strategy,
            "split": split_meta,
            "metrics": metrics,
            "balanced_dataset_r2": balanced_r2,
            "source_calibrated_metrics": calibrated_metrics,
            "source_calibrated_balanced_dataset_r2": balanced_calibrated_r2,
            "source_calibrators": source_calibrators,
            "calibration": calibration_stats,
            "best_xgboost_params": best_params or {},
            "n_features": len(feature_names),
            "source_weighting": source_weighting,
            "train_weight_summary": weighted_source_summary(train_df, train_weights),
            "pool_rows": len(response),
            "pool_drugs": int(response["DRUG_NAME"].nunique()),
            "pool_cell_lines": int(response["ModelID"].nunique()),
            "pool_by_dataset": response["dataset_source"].value_counts().to_dict(),
        }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(_safe_json(summary), handle, indent=2)
        predictions.to_csv(out_dir / "predictions.csv", index=False)
        per_drug.to_csv(out_dir / "per_drug_metrics.csv", index=False)
        per_dataset.to_csv(out_dir / "per_dataset_metrics.csv", index=False)
        per_dataset_calibrated.to_csv(out_dir / "per_dataset_metrics_source_calibrated.csv", index=False)
        calibration_table.to_csv(out_dir / "calibration.csv", index=False)
        if not tuning_results.empty:
            tuning_results.to_csv(out_dir / "leave_drug_out_tuning.csv", index=False)

        with open(out_dir / "model.pkl", "wb") as handle:
            pickle.dump(model, handle)
        with open(out_dir / "feature_names.pkl", "wb") as handle:
            pickle.dump(feature_names, handle)

        if shap_samples > 0:
            shap_outputs = self.suite._xgboost_shap_outputs(
                model=model,
                test_x=test_x,
                feature_names=feature_names,
                prediction_frame=predictions.rename(columns={"dataset_source": "dataset"}),
                max_samples=shap_samples,
                random_state=self.random_state,
            )
            for name, frame in shap_outputs.items():
                frame.to_csv(out_dir / f"{name}.csv", index=False)

        return summary


def parse_external(value: str) -> List[str]:
    if value.lower() in {"none", "off", ""}:
        return []
    if value.lower() == "all":
        return ["ctrp", "ccle", "prism"]
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train drug-inductive model with optional external data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--new-data-dir", default="data/new")
    parser.add_argument("--output-dir", default="analysis_results/multisource_training")
    parser.add_argument("--include-external", default="all", help="'none', 'all', or comma-separated ctrp,ccle,prism")
    parser.add_argument(
        "--split",
        default="leave_drug_out",
        choices=["leave_drug_out", "leave_both_out", "leave_cell_out", "stratified_random"],
    )
    parser.add_argument("--min-samples-per-drug", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-external-rows", type=int, default=None)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--include-omics", action="store_true", help="Add mutation, copy-number, fusion, and RPPA blocks.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--source-weighting",
        default="sqrt_inverse_dataset",
        choices=["none", "equal_dataset", "sqrt_inverse_dataset"],
        help="Use sqrt_inverse_dataset for soft source balancing, or equal_dataset for equal total source mass.",
    )
    parser.add_argument("--tune-xgboost", action="store_true")
    parser.add_argument("--shap-samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    trainer = MultiSourceInductiveTrainer(
        data_dir=args.data_dir,
        new_data_dir=args.new_data_dir,
        output_dir=args.output_dir,
        device=args.device,
        random_state=args.random_state,
    )
    summary = trainer.train(
        include_external=parse_external(args.include_external),
        split_strategy=args.split,
        min_samples_per_drug=args.min_samples_per_drug,
        test_size=args.test_size,
        max_external_rows=args.max_external_rows,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
        include_omics=args.include_omics,
        tune_xgboost=args.tune_xgboost,
        shap_samples=args.shap_samples,
        source_weighting=args.source_weighting,
    )
    metrics = summary["metrics"]
    calibrated = summary["source_calibrated_metrics"]
    print(
        "Multisource inductive result: "
        f"R2={metrics['r2']:.4f}, Pearson={metrics['pearson']:.4f}, "
        f"Spearman={metrics['spearman']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"balanced_dataset_R2={summary['balanced_dataset_r2']:.4f}"
    )
    print(
        "Source-calibrated result: "
        f"R2={calibrated['r2']:.4f}, Pearson={calibrated['pearson']:.4f}, "
        f"Spearman={calibrated['spearman']:.4f}, RMSE={calibrated['rmse']:.4f}, "
        f"balanced_dataset_R2={summary['source_calibrated_balanced_dataset_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
