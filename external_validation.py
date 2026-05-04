"""
External validation for the saved GDSC-trained drug sensitivity model.

This script adapts CTRPv2 and legacy CCLE pharmacology files into the feature
layout expected by saved_model/model.pkl, predicts in chunks, harmonizes each
external endpoint into a GDSC-like resistance score, and writes metrics/results
under analysis_results/external_validation/.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from benchmarking import grouped_regression_metrics, regression_metrics


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def safe_json(value: object) -> object:
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


@dataclass
class EndpointTransform:
    response_column: str
    raw_min: float
    raw_max: float
    clip_low: float
    clip_high: float
    higher_raw_is_resistant: bool

    def transform(self, values: pd.Series) -> pd.Series:
        clipped = values.astype(float).clip(self.clip_low, self.clip_high)
        denom = self.clip_high - self.clip_low
        if denom <= 0:
            raise ValueError(f"Invalid endpoint transform range for {self.response_column}")
        scaled = (clipped - self.clip_low) / denom
        if not self.higher_raw_is_resistant:
            scaled = 1.0 - scaled
        return scaled.clip(0.0, 1.0)


class ExternalValidationRunner:
    def __init__(
        self,
        data_dir: str = "data",
        new_data_dir: str = "data/new",
        model_dir: str = "saved_model",
        output_dir: str = "analysis_results/external_validation",
        chunk_size: int = 50000,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.new_data_dir = Path(new_data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size

        self.model = None
        self.scaler = None
        self.imputer = None
        self.drug_encoders = None
        self.feature_names: List[str] = []
        self.gene_features: List[str] = []
        self.expression_df: Optional[pd.DataFrame] = None
        self.model_map: Dict[str, str] = {}
        self.smiles_cache: Dict[str, np.ndarray] = {}
        self.fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)

    def load_artifacts(self) -> None:
        with open(self.model_dir / "model.pkl", "rb") as handle:
            self.model = pickle.load(handle)
        with open(self.model_dir / "scaler.pkl", "rb") as handle:
            self.scaler = pickle.load(handle)
        with open(self.model_dir / "imputer.pkl", "rb") as handle:
            self.imputer = pickle.load(handle)
        with open(self.model_dir / "feature_names.pkl", "rb") as handle:
            self.feature_names = pickle.load(handle)
        with open(self.model_dir / "drug_encoders.pkl", "rb") as handle:
            self.drug_encoders = pickle.load(handle)

        self.gene_features = [name for name in self.feature_names if "(" in name]

        expr_file = self.data_dir / "DepMap" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
        expr_cols = ["ModelID", "IsDefaultEntryForModel"] + self.gene_features
        expression = pd.read_csv(expr_file, usecols=expr_cols)
        expression = expression[expression["IsDefaultEntryForModel"].eq("Yes")].copy()
        expression = expression.drop(columns=["IsDefaultEntryForModel"]).set_index("ModelID")
        self.expression_df = expression

        self.model_map = self._build_model_id_map()

    def _build_model_id_map(self) -> Dict[str, str]:
        model_file = self.data_dir / "DepMap" / "Model.csv"
        model_df = pd.read_csv(model_file)
        assert self.expression_df is not None
        model_df = model_df[model_df["ModelID"].isin(self.expression_df.index)].copy()

        mapping: Dict[str, str] = {}
        source_cols = ["CellLineName", "StrippedCellLineName"]
        if "CCLEName" in model_df.columns:
            source_cols.append("CCLEName")

        for _, row in model_df.iterrows():
            model_id = row["ModelID"]
            for col in source_cols:
                key = normalize_name(row.get(col))
                if key and key not in mapping:
                    mapping[key] = model_id

            ccle_name = str(row.get("CCLEName", ""))
            if ccle_name and ccle_name != "nan":
                prefix_key = normalize_name(ccle_name.split("_")[0])
                if prefix_key and prefix_key not in mapping:
                    mapping[prefix_key] = model_id

        return mapping

    def map_cell_line(self, value: object) -> Optional[str]:
        key = normalize_name(value)
        if key in self.model_map:
            return self.model_map[key]

        prefix = str(value).split("_")[0]
        return self.model_map.get(normalize_name(prefix))

    def encode_category(self, encoder_name: str, value: object, unknown_value: int = -1) -> int:
        encoder = self.drug_encoders[encoder_name]
        value_str = "Unknown" if pd.isna(value) or str(value).strip() == "" else str(value).strip()
        if value_str in encoder.classes_:
            return int(encoder.transform([value_str])[0])
        if "Unknown" in encoder.classes_:
            return int(encoder.transform(["Unknown"])[0])
        return unknown_value

    def encode_drug_identity(self, drug_name: object) -> int:
        encoder = self.drug_encoders["drug"]
        value = "" if pd.isna(drug_name) else str(drug_name).strip()
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
        return len(encoder.classes_) // 2

    def fingerprint_from_smiles(self, smiles: object) -> np.ndarray:
        if pd.isna(smiles) or str(smiles).strip() == "":
            return np.zeros(256, dtype=np.float32)

        smiles_key = str(smiles).strip().rstrip(",")
        if smiles_key in self.smiles_cache:
            return self.smiles_cache[smiles_key]

        mol = Chem.MolFromSmiles(smiles_key)
        if mol is None:
            fp_array = np.zeros(256, dtype=np.float32)
        else:
            fp = self.fingerprint_generator.GetFingerprint(mol)
            fp_array = np.asarray(fp, dtype=np.float32)

        self.smiles_cache[smiles_key] = fp_array
        return fp_array

    def load_ctrp(self) -> pd.DataFrame:
        base = self.new_data_dir / "CTRPv2.0_2015_ctd2_ExpandedDataset"
        curves = pd.read_csv(
            base / "v20.data.curves_post_qc.txt",
            sep="\t",
            usecols=["experiment_id", "master_cpd_id", "area_under_curve", "apparent_ec50_umol", "pred_pv_high_conc"],
        )
        experiments = pd.read_csv(
            base / "v20.meta.per_experiment.txt",
            sep="\t",
            usecols=["experiment_id", "master_ccl_id"],
        )
        cells = pd.read_csv(base / "v20.meta.per_cell_line.txt", sep="\t")
        compounds = pd.read_csv(base / "v20.meta.per_compound.txt", sep="\t")

        frame = (
            curves.merge(experiments, on="experiment_id", how="inner")
            .merge(cells[["master_ccl_id", "ccl_name", "ccle_primary_site"]], on="master_ccl_id", how="left")
            .merge(
                compounds[
                    [
                        "master_cpd_id",
                        "cpd_name",
                        "cpd_smiles",
                        "gene_symbol_of_protein_target",
                        "target_or_activity_of_compound",
                    ]
                ],
                on="master_cpd_id",
                how="left",
            )
        )
        frame["ModelID"] = frame["ccl_name"].map(self.map_cell_line)
        frame["dataset"] = "CTRPv2"
        frame["cell_line_name"] = frame["ccl_name"]
        frame["drug_name"] = frame["cpd_name"]
        frame["smiles"] = frame["cpd_smiles"]
        frame["target"] = frame["gene_symbol_of_protein_target"].fillna(frame["target_or_activity_of_compound"])
        frame["pathway"] = "Unknown"
        frame["raw_response"] = frame["area_under_curve"]
        frame["response_type"] = "CTRP_area_under_curve"
        return frame[frame["ModelID"].notna() & frame["raw_response"].notna()].copy()

    def load_ccle_smiles_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        ctrp_compounds = self.new_data_dir / "CTRPv2.0_2015_ctd2_ExpandedDataset" / "v20.meta.per_compound.txt"
        if ctrp_compounds.exists():
            compounds = pd.read_csv(ctrp_compounds, sep="\t", usecols=["cpd_name", "cpd_smiles"])
            for _, row in compounds.dropna(subset=["cpd_name", "cpd_smiles"]).iterrows():
                lookup.setdefault(normalize_name(row["cpd_name"]), row["cpd_smiles"])

        prism_smiles = self.data_dir / "DRUG SENSITIVITY AND MUTATIONS" / "secondary-screen-dose-response-curve-parameters.csv"
        if prism_smiles.exists():
            prism = pd.read_csv(prism_smiles, low_memory=False, usecols=["name", "smiles"])
            for _, row in prism.dropna(subset=["name", "smiles"]).iterrows():
                lookup.setdefault(normalize_name(row["name"]), str(row["smiles"]).strip().rstrip(","))

        return lookup

    def load_ccle(self) -> pd.DataFrame:
        response_file = self.new_data_dir / "CCLE_NP24.2009_Drug_data_2015.02.24.csv"
        profiling_file = self.new_data_dir / "CCLE_NP24.2009_profiling_2012.02.20.csv"

        response = pd.read_csv(response_file)
        profiling = pd.read_csv(profiling_file, encoding="latin1")
        profiling = profiling.rename(
            columns={
                "Compound (code or generic name)": "Compound",
                "Target(s)": "profile_target",
                "Mechanism of action": "profile_mechanism",
                "Class": "profile_class",
            }
        )

        frame = response.merge(
            profiling[["Compound", "profile_target", "profile_mechanism", "profile_class"]],
            on="Compound",
            how="left",
        )
        frame["ModelID"] = frame["CCLE Cell Line Name"].map(self.map_cell_line)
        smiles_lookup = self.load_ccle_smiles_lookup()

        frame["dataset"] = "CCLE"
        frame["cell_line_name"] = frame["CCLE Cell Line Name"]
        frame["drug_name"] = frame["Compound"]
        frame["smiles"] = frame["Compound"].map(lambda name: smiles_lookup.get(normalize_name(name)))
        frame["target"] = frame["profile_target"].fillna(frame["Target"])
        frame["pathway"] = "Unknown"
        frame["raw_response"] = frame["ActArea"]
        frame["response_type"] = "CCLE_ActArea"
        return frame[frame["ModelID"].notna() & frame["raw_response"].notna()].copy()

    @staticmethod
    def endpoint_transform(frame: pd.DataFrame, response_column: str, higher_raw_is_resistant: bool) -> EndpointTransform:
        values = frame[response_column].astype(float)
        clip_low = float(values.quantile(0.01))
        clip_high = float(values.quantile(0.99))
        if clip_low == clip_high:
            clip_low = float(values.min())
            clip_high = float(values.max())
        return EndpointTransform(
            response_column=response_column,
            raw_min=float(values.min()),
            raw_max=float(values.max()),
            clip_low=clip_low,
            clip_high=clip_high,
            higher_raw_is_resistant=higher_raw_is_resistant,
        )

    def build_feature_chunk(self, chunk: pd.DataFrame) -> np.ndarray:
        assert self.expression_df is not None

        gene_matrix = self.expression_df.loc[chunk["ModelID"].tolist(), self.gene_features].to_numpy(dtype=np.float32)
        target_encoded = chunk["target"].map(lambda value: self.encode_category("target", value)).to_numpy(dtype=np.float32)
        pathway_encoded = chunk["pathway"].map(lambda value: self.encode_category("pathway", value)).to_numpy(dtype=np.float32)
        drug_encoded = chunk["drug_name"].map(self.encode_drug_identity).to_numpy(dtype=np.float32)

        fp_matrix = np.vstack([self.fingerprint_from_smiles(smiles) for smiles in chunk["smiles"]]).astype(np.float32)
        metadata = np.column_stack([target_encoded, pathway_encoded, drug_encoded]).astype(np.float32)
        return np.hstack([gene_matrix, metadata, fp_matrix]).astype(np.float32)

    def predict_external(self, frame: pd.DataFrame, endpoint: EndpointTransform) -> pd.DataFrame:
        assert self.model is not None and self.scaler is not None and self.imputer is not None

        predictions: List[np.ndarray] = []
        for start in range(0, len(frame), self.chunk_size):
            stop = min(start + self.chunk_size, len(frame))
            chunk = frame.iloc[start:stop]
            features = self.build_feature_chunk(chunk)
            features = self.imputer.transform(features)
            features = self.scaler.transform(features)
            predictions.append(self.model.predict(features))
            print(f"Predicted rows {start + 1:,}-{stop:,} of {len(frame):,}")

        output = frame.copy()
        output["harmonized_auc"] = endpoint.transform(output["raw_response"])
        output["prediction"] = np.concatenate(predictions)
        output["prediction_clipped"] = output["prediction"].clip(0.0, 1.0)
        output["known_training_drug"] = output["drug_name"].isin(self.drug_encoders["drug"].classes_)
        output["has_smiles"] = output["smiles"].notna() & output["smiles"].astype(str).str.strip().ne("")
        return output

    def evaluate_dataset(self, dataset: str, max_rows: Optional[int] = None) -> Dict[str, object]:
        if self.model is None:
            self.load_artifacts()

        dataset_key = dataset.lower()
        if dataset_key == "ctrp":
            frame = self.load_ctrp()
            endpoint = self.endpoint_transform(frame, "raw_response", higher_raw_is_resistant=True)
        elif dataset_key == "ccle":
            frame = self.load_ccle()
            endpoint = self.endpoint_transform(frame, "raw_response", higher_raw_is_resistant=False)
        else:
            raise ValueError(f"Unsupported external dataset: {dataset}")

        if max_rows is not None and len(frame) > max_rows:
            frame = frame.sample(n=max_rows, random_state=42).reset_index(drop=True)
            endpoint = self.endpoint_transform(
                frame,
                "raw_response",
                higher_raw_is_resistant=(dataset_key == "ctrp"),
            )

        frame = frame.reset_index(drop=True)
        print(f"\nValidating {dataset.upper()} on {len(frame):,} matched rows")
        print(f"Unique cell lines: {frame['ModelID'].nunique():,}")
        print(f"Unique drugs: {frame['drug_name'].nunique():,}")

        predictions = self.predict_external(frame, endpoint)
        metrics = regression_metrics(predictions["harmonized_auc"], predictions["prediction_clipped"])
        raw_direction_metrics = regression_metrics(predictions["harmonized_auc"], predictions["prediction"])
        per_drug = grouped_regression_metrics(
            predictions.rename(columns={"drug_name": "drug"}),
            group_col="drug",
            truth_col="harmonized_auc",
            pred_col="prediction_clipped",
            min_samples=10,
        )

        summary = {
            "dataset": dataset.upper(),
            "n_rows": int(len(predictions)),
            "n_cell_lines": int(predictions["ModelID"].nunique()),
            "n_drugs": int(predictions["drug_name"].nunique()),
            "n_known_training_drugs": int(predictions.loc[predictions["known_training_drug"], "drug_name"].nunique()),
            "rows_known_training_drug": int(predictions["known_training_drug"].sum()),
            "rows_with_smiles": int(predictions["has_smiles"].sum()),
            "endpoint_transform": asdict(endpoint),
            "metrics_clipped_prediction": metrics,
            "metrics_raw_prediction": raw_direction_metrics,
        }

        out_dir = self.output_dir / dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        predictions_path = out_dir / "predictions.csv"
        per_drug_path = out_dir / "per_drug_metrics.csv"

        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(safe_json(summary), handle, indent=2)
        predictions[
            [
                "dataset",
                "ModelID",
                "cell_line_name",
                "drug_name",
                "target",
                "pathway",
                "raw_response",
                "harmonized_auc",
                "prediction",
                "prediction_clipped",
                "known_training_drug",
                "has_smiles",
            ]
        ].to_csv(predictions_path, index=False)
        per_drug.to_csv(per_drug_path, index=False)

        print(f"Saved summary: {summary_path}")
        print(f"Saved predictions: {predictions_path}")
        print(f"Saved per-drug metrics: {per_drug_path}")
        return summary


def iter_datasets(value: str) -> Iterable[str]:
    if value.lower() == "all":
        return ["ctrp", "ccle"]
    return [value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the saved GDSC model on external CTRPv2/CCLE data.")
    parser.add_argument("--dataset", choices=["ctrp", "ccle", "all"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--new-data-dir", default="data/new")
    parser.add_argument("--model-dir", default="saved_model")
    parser.add_argument("--output-dir", default="analysis_results/external_validation")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runner = ExternalValidationRunner(
        data_dir=args.data_dir,
        new_data_dir=args.new_data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )

    summaries = []
    for dataset in iter_datasets(args.dataset):
        summaries.append(runner.evaluate_dataset(dataset, max_rows=args.max_rows))

    print("\nExternal validation summary")
    for summary in summaries:
        metrics = summary["metrics_clipped_prediction"]
        print(
            f"{summary['dataset']}: n={summary['n_rows']:,}, "
            f"R2={metrics['r2']:.4f}, Pearson={metrics['pearson']:.4f}, "
            f"Spearman={metrics['spearman']:.4f}, RMSE={metrics['rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
