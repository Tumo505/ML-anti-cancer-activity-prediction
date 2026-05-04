"""
- one-hot categorical handling for drug metadata
- a binary structure-missing indicator for absent SMILES/fingerprints
- stricter train/test splits (random, leave-cell-out, leave-drug-out, leave-both-out)
- baseline models and modality ablations
- per-drug and per-tissue summaries
- simple regression calibration summaries

Outputs are written under analysis_results/benchmarks/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from pipeline import DrugSensitivityPipeline


def _make_one_hot_encoder() -> OneHotEncoder:
    """Build an encoder compatible with newer and older sklearn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _safe_correlation(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(func(y_true, y_pred)[0])


def _clean_token(value: object) -> str:
    token = str(value).strip().upper()
    token = token.replace(" ", "_")
    token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in token)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def split_mechanism_terms(value: object) -> List[str]:
    """Split target/pathway annotations into reusable mechanism tokens."""
    if pd.isna(value):
        return []

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "unknown", "none"}:
        return []

    separators = [",", ";", "|", "/", "\\", "&", "+", " and "]
    normalized = raw
    for sep in separators:
        normalized = normalized.replace(sep, ";")

    terms: List[str] = []
    for part in normalized.split(";"):
        token = _clean_token(part)
        if token and token not in {"UNKNOWN", "NAN", "NONE", "NA"}:
            terms.append(token)
    return sorted(set(terms))


def gene_symbol_from_column(column_name: str) -> str:
    return column_name.split(" (", 1)[0].strip().upper()


def normalize_model_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "pearson": _safe_correlation(pearsonr, y_true_arr, y_pred_arr),
        "spearman": _safe_correlation(spearmanr, y_true_arr, y_pred_arr),
    }


def grouped_regression_metrics(
    frame: pd.DataFrame,
    group_col: str,
    truth_col: str = "AUC",
    pred_col: str = "prediction",
    min_samples: int = 10,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for group_name, group_df in frame.groupby(group_col):
        if len(group_df) < min_samples:
            continue
        metrics = regression_metrics(group_df[truth_col].to_numpy(), group_df[pred_col].to_numpy())
        metrics[group_col] = group_name
        metrics["n_samples"] = int(len(group_df))
        rows.append(metrics)
    if not rows:
        return pd.DataFrame(columns=[group_col, "n_samples", "r2", "rmse", "mae", "pearson", "spearman"])
    ordered_cols = [group_col, "n_samples", "r2", "rmse", "mae", "pearson", "spearman"]
    return pd.DataFrame(rows)[ordered_cols].sort_values("r2", ascending=False).reset_index(drop=True)


def calibration_summary(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    bins: int = 10,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    frame = pd.DataFrame({"true": y_true, "pred": y_pred})
    summary = {
        "calibration_slope": float("nan"),
        "calibration_intercept": float("nan"),
        "mean_prediction": float(frame["pred"].mean()),
        "mean_observed": float(frame["true"].mean()),
    }

    if frame["pred"].nunique() >= 2:
        slope, intercept = np.polyfit(frame["pred"], frame["true"], deg=1)
        summary["calibration_slope"] = float(slope)
        summary["calibration_intercept"] = float(intercept)

    n_bins = min(bins, int(frame["pred"].nunique()))
    if n_bins < 2:
        return summary, pd.DataFrame(columns=["bin", "pred_mean", "true_mean", "gap", "count"])

    binned = frame.copy()
    binned["bin"] = pd.qcut(binned["pred"], q=n_bins, duplicates="drop")
    calibration_table = (
        binned.groupby("bin", observed=False)
        .agg(
            pred_mean=("pred", "mean"),
            true_mean=("true", "mean"),
            count=("true", "size"),
        )
        .reset_index()
    )
    calibration_table["gap"] = calibration_table["true_mean"] - calibration_table["pred_mean"]
    calibration_table["bin"] = calibration_table["bin"].astype(str)
    return summary, calibration_table


@dataclass(frozen=True)
class FeatureConfig:
    include_expression: bool = True
    include_target: bool = True
    include_pathway: bool = True
    include_drug_identity: bool = False
    include_fingerprints: bool = True
    include_structure_missing_flag: bool = True
    include_rdkit_descriptors: bool = False
    include_mechanism_multihot: bool = False
    include_target_expression_interactions: bool = False
    include_mutations: bool = False
    include_copy_number: bool = False
    include_fusions: bool = False
    include_rppa: bool = False
    include_tissue: bool = False
    top_genes: int = 1000
    fingerprint_bits: int = 256
    fingerprint_radius: int = 2
    mechanism_top_terms: int = 200
    interaction_top_terms: int = 75
    top_mutation_genes: int = 300
    top_fusion_features: int = 150
    top_rppa_features: int = 200
    scale_numeric: bool = False


def reviewer_ablation_configs() -> Dict[str, FeatureConfig]:
    return {
        "full_with_identity": FeatureConfig(include_drug_identity=True, include_tissue=False),
        "full_no_identity": FeatureConfig(include_drug_identity=False, include_tissue=False),
        "full_no_identity_plus_tissue": FeatureConfig(include_drug_identity=False, include_tissue=True),
        "inductive_rich": FeatureConfig(
            include_drug_identity=False,
            include_tissue=True,
            include_rdkit_descriptors=True,
            include_mechanism_multihot=True,
            include_target_expression_interactions=True,
            fingerprint_bits=1024,
            fingerprint_radius=2,
        ),
        "inductive_rich_fp2048_r3": FeatureConfig(
            include_drug_identity=False,
            include_tissue=True,
            include_rdkit_descriptors=True,
            include_mechanism_multihot=True,
            include_target_expression_interactions=True,
            fingerprint_bits=2048,
            fingerprint_radius=3,
        ),
        "inductive_multiomics": FeatureConfig(
            include_drug_identity=False,
            include_tissue=True,
            include_rdkit_descriptors=True,
            include_mechanism_multihot=True,
            include_target_expression_interactions=True,
            include_mutations=True,
            include_copy_number=True,
            include_fusions=True,
            include_rppa=True,
            fingerprint_bits=1024,
            fingerprint_radius=2,
        ),
        "expression_only": FeatureConfig(
            include_target=False,
            include_pathway=False,
            include_drug_identity=False,
            include_fingerprints=False,
            include_structure_missing_flag=False,
            include_tissue=False,
        ),
        "drug_only_all_metadata": FeatureConfig(
            include_expression=False,
            include_target=True,
            include_pathway=True,
            include_drug_identity=True,
            include_fingerprints=False,
            include_structure_missing_flag=False,
            include_tissue=False,
        ),
        "drug_identity_only": FeatureConfig(
            include_expression=False,
            include_target=False,
            include_pathway=False,
            include_drug_identity=True,
            include_fingerprints=False,
            include_structure_missing_flag=False,
            include_tissue=False,
        ),
        "target_pathway_only": FeatureConfig(
            include_expression=False,
            include_target=True,
            include_pathway=True,
            include_drug_identity=False,
            include_fingerprints=False,
            include_structure_missing_flag=False,
            include_tissue=False,
        ),
        "fingerprint_only": FeatureConfig(
            include_expression=False,
            include_target=False,
            include_pathway=False,
            include_drug_identity=False,
            include_fingerprints=True,
            include_structure_missing_flag=True,
            include_tissue=False,
        ),
        "tissue_only": FeatureConfig(
            include_expression=False,
            include_target=False,
            include_pathway=False,
            include_drug_identity=False,
            include_fingerprints=False,
            include_structure_missing_flag=False,
            include_tissue=True,
        ),
    }


class ReviewerBenchmarkSuite:
    def __init__(self, data_dir: str = "data", output_dir: str = "analysis_results/benchmarks"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.pipeline: DrugSensitivityPipeline | None = None
        self.response_df: pd.DataFrame | None = None
        self.expression_df: pd.DataFrame | None = None
        self.gene_columns: List[str] = []
        self.fingerprint_cache: Dict[Tuple[int, int], pd.DataFrame] = {}
        self.descriptor_cache: pd.DataFrame | None = None
        self.copy_number_df: pd.DataFrame | None = None
        self.mutation_events: pd.DataFrame | None = None
        self.fusion_events: pd.DataFrame | None = None
        self.rppa_df: pd.DataFrame | None = None

    def load(self) -> None:
        if self.response_df is not None and self.expression_df is not None:
            return

        pipeline = DrugSensitivityPipeline(data_dir=str(self.data_dir))
        pipeline.load_gdsc_data()
        pipeline.load_depmap_expression()
        pipeline.load_model_mapping()
        pipeline.load_smiles_data()

        model_columns = [
            "ModelID",
            "SangerModelID",
            "CellLineName",
            "OncotreeLineage",
            "OncotreePrimaryDisease",
        ]
        model_mapping = pipeline.model_mapping[model_columns].copy()

        response = pipeline.gdsc_data.merge(
            model_mapping,
            left_on="SANGER_MODEL_ID",
            right_on="SangerModelID",
            how="inner",
        )
        response = response[response["ModelID"].isin(pipeline.expression_data.index)].copy()
        response = response[response["AUC"].notna()].copy()

        response["PUTATIVE_TARGET"] = response["PUTATIVE_TARGET"].fillna("Unknown")
        response["PATHWAY_NAME"] = response["PATHWAY_NAME"].fillna("Unknown")
        response["OncotreeLineage"] = response["OncotreeLineage"].fillna("Unknown")
        response["OncotreePrimaryDisease"] = response["OncotreePrimaryDisease"].fillna("Unknown")

        self.pipeline = pipeline
        self.response_df = response.reset_index(drop=True)
        self.expression_df = pipeline.expression_data.copy()
        self.gene_columns = list(self.expression_df.columns)

        print("-" * 60)
        print("Loaded compact reviewer benchmark dataset")
        print("-" * 60)
        print(f"Response rows: {len(self.response_df):,}")
        print(f"Unique drugs: {self.response_df['DRUG_NAME'].nunique()}")
        print(f"Unique cell lines: {self.response_df['ModelID'].nunique()}")
        print(f"Expression matrix: {self.expression_df.shape[0]} x {self.expression_df.shape[1]}")

    def filtered_response(self, min_samples_per_drug: int) -> pd.DataFrame:
        self.load()
        assert self.response_df is not None

        counts = self.response_df["DRUG_NAME"].value_counts()
        valid_drugs = counts[counts >= min_samples_per_drug].index
        filtered = self.response_df[self.response_df["DRUG_NAME"].isin(valid_drugs)].copy()
        print(f"Using {len(valid_drugs)} drugs with >= {min_samples_per_drug} samples")
        print(f"Filtered response rows: {len(filtered):,}")
        return filtered.reset_index(drop=True)

    def add_smiles_records(self, drug_smiles: pd.DataFrame) -> None:
        """Extend the local SMILES table, then invalidate chemistry-derived caches."""
        self.load()
        assert self.pipeline is not None
        if self.pipeline.smiles_data is None:
            self.pipeline.load_smiles_data()
        assert self.pipeline.smiles_data is not None

        clean = drug_smiles[["DRUG_NAME", "SMILES"]].dropna().copy()
        clean["DRUG_NAME"] = clean["DRUG_NAME"].astype(str).str.strip()
        clean["SMILES"] = clean["SMILES"].astype(str).str.strip().str.rstrip(",")
        clean = clean[clean["DRUG_NAME"].ne("") & clean["SMILES"].ne("")]

        self.pipeline.smiles_data = (
            pd.concat([self.pipeline.smiles_data, clean], ignore_index=True)
            .drop_duplicates("DRUG_NAME", keep="last")
            .reset_index(drop=True)
        )
        self.fingerprint_cache.clear()
        self.descriptor_cache = None

    def _ensure_fingerprints(self, bits: int, radius: int) -> pd.DataFrame:
        key = (bits, radius)
        if key not in self.fingerprint_cache:
            assert self.pipeline is not None
            fp_df = self.pipeline.generate_molecular_fingerprints(
                fp_type="morgan",
                radius=radius,
                n_bits=bits,
            )
            self.fingerprint_cache[key] = fp_df.astype(np.float32)
        return self.fingerprint_cache[key]

    def _ensure_rdkit_descriptors(self) -> pd.DataFrame:
        if self.descriptor_cache is not None:
            return self.descriptor_cache

        assert self.pipeline is not None
        if self.pipeline.smiles_data is None:
            self.pipeline.load_smiles_data()
        assert self.pipeline.smiles_data is not None

        rows: List[Dict[str, float | str]] = []
        for _, row in self.pipeline.smiles_data.iterrows():
            smiles = str(row["SMILES"]).strip().rstrip(",")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            rows.append(
                {
                    "DRUG_NAME": row["DRUG_NAME"],
                    "desc_mol_wt": Descriptors.MolWt(mol),
                    "desc_exact_mol_wt": Descriptors.ExactMolWt(mol),
                    "desc_logp": Crippen.MolLogP(mol),
                    "desc_mr": Crippen.MolMR(mol),
                    "desc_tpsa": rdMolDescriptors.CalcTPSA(mol),
                    "desc_h_donors": Lipinski.NumHDonors(mol),
                    "desc_h_acceptors": Lipinski.NumHAcceptors(mol),
                    "desc_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                    "desc_heavy_atoms": Lipinski.HeavyAtomCount(mol),
                    "desc_ring_count": Lipinski.RingCount(mol),
                    "desc_aromatic_rings": Lipinski.NumAromaticRings(mol),
                    "desc_aliphatic_rings": Lipinski.NumAliphaticRings(mol),
                    "desc_saturated_rings": Lipinski.NumSaturatedRings(mol),
                    "desc_hetero_atoms": Lipinski.NumHeteroatoms(mol),
                    "desc_fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
                    "desc_formal_charge": Chem.GetFormalCharge(mol),
                }
            )

        if rows:
            descriptors = pd.DataFrame(rows).drop_duplicates("DRUG_NAME").set_index("DRUG_NAME")
        else:
            descriptors = pd.DataFrame(index=[], columns=["desc_mol_wt"])

        self.descriptor_cache = descriptors.astype(np.float32)
        return self.descriptor_cache

    def _ensure_copy_number(self) -> pd.DataFrame:
        if self.copy_number_df is not None:
            return self.copy_number_df

        cn_path = self.data_dir / "DepMap" / "PortalOmicsCNGeneLog2.csv"
        if not cn_path.exists():
            cn_path = self.data_dir / "DepMap" / "OmicsCNGeneWGS.csv"
        if not cn_path.exists():
            self.copy_number_df = pd.DataFrame()
            return self.copy_number_df

        cn = pd.read_csv(cn_path, low_memory=False)
        id_col = "ModelID" if "ModelID" in cn.columns else cn.columns[0]
        cn = cn.rename(columns={id_col: "ModelID"})
        cn = cn.set_index("ModelID")
        drop_cols = [
            col
            for col in ["SequencingID", "ModelConditionID", "IsDefaultEntryForModel", "IsDefaultEntryForMC"]
            if col in cn.columns
        ]
        cn = cn.drop(columns=drop_cols, errors="ignore")
        self.copy_number_df = cn.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        return self.copy_number_df

    def _ensure_mutation_events(self) -> pd.DataFrame:
        if self.mutation_events is not None:
            return self.mutation_events

        path = self.data_dir / "DepMap" / "OmicsSomaticMutations.csv"
        if not path.exists():
            self.mutation_events = pd.DataFrame(columns=["ModelID", "HugoSymbol"])
            return self.mutation_events

        usecols = [
            "ModelID",
            "IsDefaultEntryForModel",
            "HugoSymbol",
            "VepImpact",
            "LikelyLoF",
            "Hotspot",
            "OncogeneHighImpact",
            "TumorSuppressorHighImpact",
        ]
        mut = pd.read_csv(path, usecols=lambda col: col in usecols, low_memory=False)
        if "IsDefaultEntryForModel" in mut.columns:
            mut = mut[mut["IsDefaultEntryForModel"].eq("Yes")].copy()
        mut = mut[mut["ModelID"].notna() & mut["HugoSymbol"].notna()].copy()
        high_impact = mut.get("VepImpact", pd.Series(index=mut.index, dtype=object)).isin(["HIGH", "MODERATE"])
        flagged = pd.Series(False, index=mut.index)
        for col in ["LikelyLoF", "Hotspot", "OncogeneHighImpact", "TumorSuppressorHighImpact"]:
            if col in mut.columns:
                flagged = flagged | mut[col].fillna(False).map(
                    lambda value: str(value).strip().lower() in {"true", "1", "yes"}
                )
        mut = mut[high_impact | flagged].copy()
        mut["HugoSymbol"] = mut["HugoSymbol"].astype(str).str.upper()
        self.mutation_events = mut[["ModelID", "HugoSymbol"]].drop_duplicates().reset_index(drop=True)
        return self.mutation_events

    def _ensure_fusion_events(self) -> pd.DataFrame:
        if self.fusion_events is not None:
            return self.fusion_events

        path = self.data_dir / "DepMap" / "OmicsFusionFiltered.csv"
        if not path.exists():
            self.fusion_events = pd.DataFrame(columns=["ModelID", "fusion_feature"])
            return self.fusion_events

        fusion = pd.read_csv(
            path,
            usecols=lambda col: col
            in {"ModelID", "IsDefaultEntryForModel", "CanonicalFusionName", "Gene1", "Gene2"},
            low_memory=False,
        )
        if "IsDefaultEntryForModel" in fusion.columns:
            fusion = fusion[fusion["IsDefaultEntryForModel"].eq("Yes")].copy()
        fusion = fusion[fusion["ModelID"].notna()].copy()
        feature = fusion["CanonicalFusionName"].fillna("").astype(str)
        feature = feature.where(feature.str.len() > 0, fusion["Gene1"].fillna("").astype(str) + "--" + fusion["Gene2"].fillna("").astype(str))
        fusion["fusion_feature"] = feature.str.upper()
        fusion = fusion[fusion["fusion_feature"].str.len() > 2]
        self.fusion_events = fusion[["ModelID", "fusion_feature"]].drop_duplicates().reset_index(drop=True)
        return self.fusion_events

    def _ensure_rppa(self) -> pd.DataFrame:
        if self.rppa_df is not None:
            return self.rppa_df

        path = self.data_dir / "DepMap" / "CCLE_RPPA_20180123.csv"
        if not path.exists():
            self.rppa_df = pd.DataFrame()
            return self.rppa_df

        assert self.pipeline is not None
        rppa = pd.read_csv(path, low_memory=False).rename(columns={pd.read_csv(path, nrows=0).columns[0]: "rppa_name"})
        mapping: Dict[str, str] = {}
        if self.pipeline.model_mapping is not None:
            for _, row in self.pipeline.model_mapping.iterrows():
                model_id = row.get("ModelID")
                if pd.isna(model_id):
                    continue
                for col in ["CCLEName", "CellLineName", "StrippedCellLineName"]:
                    if col in row.index:
                        key = normalize_model_name(row.get(col))
                        if key:
                            mapping.setdefault(key, model_id)
        rppa["ModelID"] = rppa["rppa_name"].map(lambda name: mapping.get(normalize_model_name(name)))
        rppa = rppa[rppa["ModelID"].notna()].drop_duplicates("ModelID")
        rppa = rppa.drop(columns=["rppa_name"]).set_index("ModelID")
        self.rppa_df = rppa.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        return self.rppa_df

    def split_response(
        self,
        response: pd.DataFrame,
        split_strategy: str,
        test_size: float,
        random_state: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        if split_strategy == "stratified_random":
            train_df, test_df = train_test_split(
                response,
                test_size=test_size,
                random_state=random_state,
                stratify=response["DRUG_NAME"],
            )
            meta = {"discarded_samples": 0}
        elif split_strategy == "leave_cell_out":
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(response, groups=response["ModelID"]))
            train_df = response.iloc[train_idx].copy()
            test_df = response.iloc[test_idx].copy()
            meta = {"discarded_samples": 0}
        elif split_strategy == "leave_drug_out":
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(response, groups=response["DRUG_NAME"]))
            train_df = response.iloc[train_idx].copy()
            test_df = response.iloc[test_idx].copy()
            meta = {"discarded_samples": 0}
        elif split_strategy == "leave_both_out":
            train_df, test_df, meta = self._leave_both_out_split(response, test_size, random_state)
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")

        meta.update(
            {
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
                "train_drugs": int(train_df["DRUG_NAME"].nunique()),
                "test_drugs": int(test_df["DRUG_NAME"].nunique()),
                "train_cells": int(train_df["ModelID"].nunique()),
                "test_cells": int(test_df["ModelID"].nunique()),
            }
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True), meta

    def _leave_both_out_split(
        self,
        response: pd.DataFrame,
        test_size: float,
        random_state: int,
        max_attempts: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        unique_drugs = np.array(sorted(response["DRUG_NAME"].unique()))
        unique_cells = np.array(sorted(response["ModelID"].unique()))
        n_test_drugs = max(1, int(round(len(unique_drugs) * test_size)))
        n_test_cells = max(1, int(round(len(unique_cells) * test_size)))

        for attempt in range(max_attempts):
            rng = np.random.default_rng(random_state + attempt)
            test_drugs = set(rng.choice(unique_drugs, size=n_test_drugs, replace=False).tolist())
            test_cells = set(rng.choice(unique_cells, size=n_test_cells, replace=False).tolist())

            test_mask = response["DRUG_NAME"].isin(test_drugs) & response["ModelID"].isin(test_cells)
            train_mask = (~response["DRUG_NAME"].isin(test_drugs)) & (~response["ModelID"].isin(test_cells))

            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue

            discarded = int((~(test_mask | train_mask)).sum())
            meta = {"discarded_samples": discarded}
            return response.loc[train_mask].copy(), response.loc[test_mask].copy(), meta

        raise RuntimeError("Could not construct a leave_both_out split with non-empty train and test sets.")

    def select_top_genes(self, train_df: pd.DataFrame, top_genes: int) -> List[str]:
        assert self.expression_df is not None
        model_counts = train_df["ModelID"].value_counts()
        expr_subset = self.expression_df.loc[model_counts.index]

        weights = model_counts.to_numpy(dtype=np.float64)
        expr_values = expr_subset.to_numpy(dtype=np.float32, copy=False)
        total_weight = weights.sum()
        weighted_mean = np.average(expr_values, axis=0, weights=weights)
        centered = expr_values - weighted_mean
        weighted_var = np.einsum("i,ij,ij->j", weights, centered, centered) / max(total_weight - 1, 1)

        k = min(top_genes, expr_subset.shape[1])
        top_idx = np.argpartition(weighted_var, -k)[-k:]
        top_idx = top_idx[np.argsort(weighted_var[top_idx])[::-1]]
        return expr_subset.columns[top_idx].tolist()

    def _numeric_matrix(
        self,
        frame: pd.DataFrame,
        config: FeatureConfig,
        selected_genes: Sequence[str],
    ) -> Tuple[np.ndarray, List[str]]:
        assert self.expression_df is not None
        parts: List[np.ndarray] = []
        feature_names: List[str] = []

        if config.include_expression:
            gene_matrix = self.expression_df.loc[frame["ModelID"], selected_genes].to_numpy(dtype=np.float32, copy=True)
            parts.append(gene_matrix)
            feature_names.extend(selected_genes)

        if config.include_fingerprints:
            fp_df = self._ensure_fingerprints(config.fingerprint_bits, config.fingerprint_radius)
            aligned_fp = fp_df.reindex(frame["DRUG_NAME"].tolist())
            structure_missing = aligned_fp.isna().all(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)
            fp_matrix = aligned_fp.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
            parts.append(fp_matrix)
            feature_names.extend(fp_df.columns.tolist())
            if config.include_structure_missing_flag:
                parts.append(structure_missing)
                feature_names.append("structure_missing")

        if config.include_rdkit_descriptors:
            descriptor_df = self._ensure_rdkit_descriptors()
            aligned_desc = descriptor_df.reindex(frame["DRUG_NAME"].tolist())
            desc_matrix = aligned_desc.to_numpy(dtype=np.float32, copy=False)
            parts.append(desc_matrix)
            feature_names.extend(descriptor_df.columns.tolist())

        if not parts:
            return np.empty((len(frame), 0), dtype=np.float32), []

        return np.hstack(parts).astype(np.float32, copy=False), feature_names

    @staticmethod
    def _categorical_columns(config: FeatureConfig) -> List[str]:
        cols: List[str] = []
        if config.include_target:
            cols.append("PUTATIVE_TARGET")
        if config.include_pathway:
            cols.append("PATHWAY_NAME")
        if config.include_drug_identity:
            cols.append("DRUG_NAME")
        if config.include_tissue:
            cols.append("OncotreeLineage")
        return cols

    @staticmethod
    def _fit_mechanism_vocab(train_df: pd.DataFrame, config: FeatureConfig) -> List[Tuple[str, str]]:
        counts: Dict[Tuple[str, str], int] = {}
        for source_col, prefix in [("PUTATIVE_TARGET", "target"), ("PATHWAY_NAME", "pathway")]:
            for value in train_df[source_col].fillna("Unknown"):
                for term in split_mechanism_terms(value):
                    key = (prefix, term)
                    counts[key] = counts.get(key, 0) + 1

        ordered = sorted(counts, key=lambda item: (-counts[item], item[0], item[1]))
        return ordered[: config.mechanism_top_terms]

    @staticmethod
    def _mechanism_matrix(frame: pd.DataFrame, vocab: Sequence[Tuple[str, str]]) -> sp.csr_matrix:
        if not vocab:
            return sp.csr_matrix((len(frame), 0), dtype=np.float32)

        vocab_index = {term: idx for idx, term in enumerate(vocab)}
        rows: List[int] = []
        cols: List[int] = []
        for row_idx, (_, row) in enumerate(frame.iterrows()):
            active_terms: set[Tuple[str, str]] = set()
            for source_col, prefix in [("PUTATIVE_TARGET", "target"), ("PATHWAY_NAME", "pathway")]:
                active_terms.update((prefix, term) for term in split_mechanism_terms(row[source_col]))
            for term in active_terms:
                col_idx = vocab_index.get(term)
                if col_idx is not None:
                    rows.append(row_idx)
                    cols.append(col_idx)

        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(frame), len(vocab)), dtype=np.float32)

    @staticmethod
    def _fit_target_expression_terms(
        train_df: pd.DataFrame,
        selected_genes: Sequence[str],
        max_terms: int,
    ) -> List[Tuple[str, str]]:
        symbol_to_gene = {gene_symbol_from_column(gene): gene for gene in selected_genes}
        counts: Dict[str, int] = {}
        for value in train_df["PUTATIVE_TARGET"].fillna("Unknown"):
            for term in split_mechanism_terms(value):
                if term in symbol_to_gene:
                    counts[term] = counts.get(term, 0) + 1

        ordered_terms = sorted(counts, key=lambda term: (-counts[term], term))[:max_terms]
        return [(term, symbol_to_gene[term]) for term in ordered_terms]

    def _target_expression_interaction_matrix(
        self,
        frame: pd.DataFrame,
        term_gene_pairs: Sequence[Tuple[str, str]],
    ) -> np.ndarray:
        assert self.expression_df is not None
        if not term_gene_pairs:
            return np.empty((len(frame), 0), dtype=np.float32)

        output = np.zeros((len(frame), len(term_gene_pairs)), dtype=np.float32)
        target_terms = frame["PUTATIVE_TARGET"].map(lambda value: set(split_mechanism_terms(value)))
        for col_idx, (term, gene_col) in enumerate(term_gene_pairs):
            active = target_terms.map(lambda terms: term in terms).to_numpy(dtype=bool)
            if active.any():
                output[active, col_idx] = self.expression_df.loc[frame.loc[active, "ModelID"], gene_col].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
        return output

    @staticmethod
    def _append_numeric_block(
        base_matrix: np.ndarray,
        base_names: List[str],
        block: np.ndarray,
        block_names: Sequence[str],
    ) -> Tuple[np.ndarray, List[str]]:
        if block.shape[1] == 0:
            return base_matrix, base_names
        if base_matrix.shape[1] == 0:
            combined = block.astype(np.float32, copy=False)
        else:
            combined = np.hstack([base_matrix, block]).astype(np.float32, copy=False)
        return combined, base_names + list(block_names)

    def _copy_number_block(
        self,
        frame: pd.DataFrame,
        selected_genes: Sequence[str],
    ) -> Tuple[np.ndarray, List[str]]:
        cn = self._ensure_copy_number()
        if cn.empty:
            return np.empty((len(frame), 0), dtype=np.float32), []

        cn_cols = [gene for gene in selected_genes if gene in cn.columns]
        if not cn_cols:
            return np.empty((len(frame), 0), dtype=np.float32), []
        matrix = cn.reindex(frame["ModelID"].tolist())[cn_cols].to_numpy(dtype=np.float32, copy=True)
        return matrix, [f"cn_{col}" for col in cn_cols]

    def _mutation_block(
        self,
        train_df: pd.DataFrame,
        frame: pd.DataFrame,
        max_genes: int,
    ) -> Tuple[np.ndarray, List[str]]:
        mut = self._ensure_mutation_events()
        if mut.empty:
            return np.empty((len(frame), 0), dtype=np.float32), []

        train_models = set(train_df["ModelID"])
        train_mut = mut[mut["ModelID"].isin(train_models)]
        top_genes = train_mut["HugoSymbol"].value_counts().head(max_genes).index.tolist()
        if not top_genes:
            return np.empty((len(frame), 0), dtype=np.float32), []

        active = mut[mut["HugoSymbol"].isin(top_genes) & mut["ModelID"].isin(frame["ModelID"])].copy()
        if active.empty:
            return np.zeros((len(frame), len(top_genes)), dtype=np.float32), [f"mut_{gene}" for gene in top_genes]
        pivot = pd.crosstab(active["ModelID"], active["HugoSymbol"]).clip(upper=1)
        pivot = pivot.reindex(index=frame["ModelID"].tolist(), columns=top_genes, fill_value=0)
        return pivot.to_numpy(dtype=np.float32, copy=False), [f"mut_{gene}" for gene in top_genes]

    def _fusion_block(
        self,
        train_df: pd.DataFrame,
        frame: pd.DataFrame,
        max_features: int,
    ) -> Tuple[np.ndarray, List[str]]:
        fusion = self._ensure_fusion_events()
        if fusion.empty:
            return np.empty((len(frame), 0), dtype=np.float32), []

        train_models = set(train_df["ModelID"])
        train_fusion = fusion[fusion["ModelID"].isin(train_models)]
        top_features = train_fusion["fusion_feature"].value_counts().head(max_features).index.tolist()
        if not top_features:
            return np.empty((len(frame), 0), dtype=np.float32), []

        active = fusion[fusion["fusion_feature"].isin(top_features) & fusion["ModelID"].isin(frame["ModelID"])].copy()
        if active.empty:
            return np.zeros((len(frame), len(top_features)), dtype=np.float32), [
                f"fusion_{feature}" for feature in top_features
            ]
        pivot = pd.crosstab(active["ModelID"], active["fusion_feature"]).clip(upper=1)
        pivot = pivot.reindex(index=frame["ModelID"].tolist(), columns=top_features, fill_value=0)
        return pivot.to_numpy(dtype=np.float32, copy=False), [f"fusion_{feature}" for feature in top_features]

    def _rppa_block(
        self,
        train_df: pd.DataFrame,
        frame: pd.DataFrame,
        max_features: int,
    ) -> Tuple[np.ndarray, List[str]]:
        rppa = self._ensure_rppa()
        if rppa.empty:
            return np.empty((len(frame), 0), dtype=np.float32), []

        train_models = [model_id for model_id in train_df["ModelID"].unique() if model_id in rppa.index]
        if not train_models:
            return np.empty((len(frame), 0), dtype=np.float32), []
        variance = rppa.loc[train_models].var(axis=0, skipna=True).sort_values(ascending=False)
        selected = variance.head(max_features).index.tolist()
        matrix = rppa.reindex(frame["ModelID"].tolist())[selected].to_numpy(dtype=np.float32, copy=True)
        return matrix, [f"rppa_{feature}" for feature in selected]

    def prepare_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: FeatureConfig,
    ) -> Tuple[sp.csr_matrix | np.ndarray, sp.csr_matrix | np.ndarray, List[str]]:
        selected_genes = self.select_top_genes(train_df, config.top_genes) if config.include_expression else []

        train_num, numeric_names = self._numeric_matrix(train_df, config, selected_genes)
        test_num, _ = self._numeric_matrix(test_df, config, selected_genes)

        if config.include_copy_number:
            train_block, block_names = self._copy_number_block(train_df, selected_genes)
            test_block, _ = self._copy_number_block(test_df, selected_genes)
            train_num, numeric_names = self._append_numeric_block(train_num, numeric_names, train_block, block_names)
            test_num, _ = self._append_numeric_block(test_num, [], test_block, block_names)

        if config.include_mutations:
            train_block, block_names = self._mutation_block(train_df, train_df, config.top_mutation_genes)
            test_block, _ = self._mutation_block(train_df, test_df, config.top_mutation_genes)
            train_num, numeric_names = self._append_numeric_block(train_num, numeric_names, train_block, block_names)
            test_num, _ = self._append_numeric_block(test_num, [], test_block, block_names)

        if config.include_fusions:
            train_block, block_names = self._fusion_block(train_df, train_df, config.top_fusion_features)
            test_block, _ = self._fusion_block(train_df, test_df, config.top_fusion_features)
            train_num, numeric_names = self._append_numeric_block(train_num, numeric_names, train_block, block_names)
            test_num, _ = self._append_numeric_block(test_num, [], test_block, block_names)

        if config.include_rppa:
            train_block, block_names = self._rppa_block(train_df, train_df, config.top_rppa_features)
            test_block, _ = self._rppa_block(train_df, test_df, config.top_rppa_features)
            train_num, numeric_names = self._append_numeric_block(train_num, numeric_names, train_block, block_names)
            test_num, _ = self._append_numeric_block(test_num, [], test_block, block_names)

        if config.include_target_expression_interactions:
            term_gene_pairs = self._fit_target_expression_terms(
                train_df=train_df,
                selected_genes=selected_genes,
                max_terms=config.interaction_top_terms,
            )
            train_interactions = self._target_expression_interaction_matrix(train_df, term_gene_pairs)
            test_interactions = self._target_expression_interaction_matrix(test_df, term_gene_pairs)
            if train_interactions.shape[1] > 0:
                train_num = np.hstack([train_num, train_interactions]).astype(np.float32, copy=False)
                test_num = np.hstack([test_num, test_interactions]).astype(np.float32, copy=False)
                numeric_names.extend([f"target_expr_interaction_{term}__{gene}" for term, gene in term_gene_pairs])

        if numeric_names:
            imputer = SimpleImputer(strategy="median")
            train_num = imputer.fit_transform(train_num).astype(np.float32)
            test_num = imputer.transform(test_num).astype(np.float32)

            if config.scale_numeric:
                scaler = StandardScaler()
                train_num = scaler.fit_transform(train_num).astype(np.float32)
                test_num = scaler.transform(test_num).astype(np.float32)

        cat_cols = self._categorical_columns(config)
        cat_feature_names: List[str] = []
        train_cat = None
        test_cat = None
        if cat_cols:
            encoder = _make_one_hot_encoder()
            train_cat = encoder.fit_transform(train_df[cat_cols].fillna("Unknown").astype(str))
            test_cat = encoder.transform(test_df[cat_cols].fillna("Unknown").astype(str))
            cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()

        if config.include_mechanism_multihot:
            mechanism_vocab = self._fit_mechanism_vocab(train_df, config)
            train_mech = self._mechanism_matrix(train_df, mechanism_vocab)
            test_mech = self._mechanism_matrix(test_df, mechanism_vocab)
            mech_names = [f"{prefix}_token_{term}" for prefix, term in mechanism_vocab]
            if train_cat is None:
                train_cat = train_mech
                test_cat = test_mech
            else:
                train_cat = sp.hstack([train_cat, train_mech], format="csr")
                test_cat = sp.hstack([test_cat, test_mech], format="csr")
            cat_feature_names.extend(mech_names)

        if not numeric_names and not cat_feature_names:
            raise ValueError("FeatureConfig produced no usable features.")

        if numeric_names and cat_feature_names:
            train_x = sp.hstack([sp.csr_matrix(train_num), train_cat], format="csr")
            test_x = sp.hstack([sp.csr_matrix(test_num), test_cat], format="csr")
        elif numeric_names:
            train_x = train_num
            test_x = test_num
        else:
            train_x = train_cat
            test_x = test_cat

        return train_x, test_x, numeric_names + cat_feature_names

    @staticmethod
    def _build_model(
        device: str,
        random_state: int,
        params: Dict[str, object] | None = None,
    ) -> xgb.XGBRegressor:
        model_params = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "device": device,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if params:
            model_params.update(params)
        return xgb.XGBRegressor(
            **model_params,
        )

    @staticmethod
    def _build_baseline_model(
        model_name: str,
        device: str,
        random_state: int,
        xgboost_params: Dict[str, object] | None = None,
    ):
        if model_name == "xgboost":
            return ReviewerBenchmarkSuite._build_model(device=device, random_state=random_state, params=xgboost_params)
        if model_name == "ridge":
            return Ridge(alpha=1.0, random_state=random_state)
        if model_name == "elasticnet":
            return ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=5000, random_state=random_state)
        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=3,
                n_jobs=1,
                random_state=random_state,
            )
        if model_name == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm is not installed. Install lightgbm or choose another model.")
            return lgb.LGBMRegressor(
                objective="regression",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        raise ValueError(f"Unknown model_name: {model_name}")

    @staticmethod
    def _model_feature_importance(model, model_name: str, feature_names: Sequence[str]) -> pd.DataFrame:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
            importance_name = "importance"
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
            importance_name = "abs_coefficient"
        else:
            values = np.zeros(len(feature_names), dtype=float)
            importance_name = "importance"

        return pd.DataFrame(
            {
                "feature": feature_names,
                importance_name: values,
                "model": model_name,
            }
        ).sort_values(importance_name, ascending=False)

    def _xgboost_shap_outputs(
        self,
        model,
        test_x,
        feature_names: Sequence[str],
        prediction_frame: pd.DataFrame,
        max_samples: int,
        random_state: int,
        top_n_local: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        if max_samples <= 0:
            return {}

        n_samples = test_x.shape[0]
        if n_samples == 0:
            return {}

        if n_samples > max_samples:
            rng = np.random.default_rng(random_state)
            sample_positions = np.sort(rng.choice(n_samples, size=max_samples, replace=False))
        else:
            sample_positions = np.arange(n_samples)

        test_sample = test_x[sample_positions]
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(test_sample)
        contrib = booster.predict(dmatrix, pred_contribs=True)
        shap_values = contrib[:, :-1]
        base_values = contrib[:, -1]

        abs_mean = np.abs(shap_values).mean(axis=0)
        mean_value = shap_values.mean(axis=0)
        global_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": abs_mean,
                "mean_shap": mean_value,
            }
        )
        global_df["modality"] = global_df["feature"].map(self._feature_modality)
        global_df = global_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        modality_df = (
            global_df.groupby("modality", as_index=False)
            .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum"))
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        local_rows: List[Dict[str, object]] = []
        sampled_predictions = prediction_frame.iloc[sample_positions].reset_index(drop=True)
        top_n = min(top_n_local, shap_values.shape[1])
        for local_i in range(shap_values.shape[0]):
            top_idx = np.argsort(np.abs(shap_values[local_i]))[-top_n:][::-1]
            row_meta = sampled_predictions.iloc[local_i]
            for rank, feature_idx in enumerate(top_idx, start=1):
                local_rows.append(
                    {
                        "sample_position": int(sample_positions[local_i]),
                        "rank": rank,
                        "feature": feature_names[feature_idx],
                        "modality": self._feature_modality(feature_names[feature_idx]),
                        "shap_value": float(shap_values[local_i, feature_idx]),
                        "abs_shap": float(abs(shap_values[local_i, feature_idx])),
                        "base_value": float(base_values[local_i]),
                        "prediction": float(row_meta["prediction"]),
                        "true_auc": float(row_meta["AUC"]),
                        "drug": row_meta["DRUG_NAME"],
                        "model_id": row_meta["ModelID"],
                    }
                )

        return {
            "shap_global": global_df,
            "shap_modality": modality_df,
            "shap_local_top": pd.DataFrame(local_rows),
        }

    def tune_xgboost_on_leave_drug_out(
        self,
        train_df: pd.DataFrame,
        config: FeatureConfig,
        device: str,
        random_state: int,
        validation_size: float = 0.2,
    ) -> Tuple[Dict[str, object], pd.DataFrame]:
        """Select XGBoost hyperparameters using an inner unseen-drug validation split."""
        inner_train_df, inner_valid_df, _ = self.split_response(
            train_df,
            split_strategy="leave_drug_out",
            test_size=validation_size,
            random_state=random_state + 1000,
        )
        train_x, valid_x, _ = self.prepare_features(inner_train_df, inner_valid_df, config)
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
        best_r2 = -np.inf
        for idx, params in enumerate(param_grid, start=1):
            print(f"Tuning XGBoost candidate {idx}/{len(param_grid)} on inner leave-drug-out split...")
            model = self._build_model(device=device, random_state=random_state + idx, params=params)
            model.fit(train_x, y_train)
            pred = model.predict(valid_x)
            metrics = regression_metrics(y_valid, pred)
            row = {"candidate": idx, **params, **metrics}
            rows.append(row)
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_params = params

        return best_params, pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)

    @staticmethod
    def baseline_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        global_mean = float(train_df["AUC"].mean())

        drug_mean = train_df.groupby("DRUG_NAME")["AUC"].mean()
        per_drug_pred = test_df["DRUG_NAME"].map(drug_mean).fillna(global_mean).to_numpy(dtype=float)

        tissue_mean = train_df.groupby("OncotreeLineage")["AUC"].mean()
        per_tissue_pred = test_df["OncotreeLineage"].map(tissue_mean).fillna(global_mean).to_numpy(dtype=float)

        return {
            "global_mean": np.full(len(test_df), global_mean, dtype=float),
            "per_drug_mean": per_drug_pred,
            "tissue_mean": per_tissue_pred,
        }

    @staticmethod
    def _feature_modality(feature_name: str) -> str:
        if feature_name == "structure_missing":
            return "structure_missing"
        if feature_name.startswith("fp_"):
            return "fingerprint"
        if feature_name.startswith("desc_"):
            return "rdkit_descriptor"
        if feature_name.startswith("target_token_"):
            return "target_multihot"
        if feature_name.startswith("pathway_token_"):
            return "pathway_multihot"
        if feature_name.startswith("target_expr_interaction_"):
            return "target_expression_interaction"
        if feature_name.startswith("cn_"):
            return "copy_number"
        if feature_name.startswith("mut_"):
            return "mutation"
        if feature_name.startswith("fusion_"):
            return "fusion"
        if feature_name.startswith("rppa_"):
            return "rppa"
        if feature_name.startswith("PUTATIVE_TARGET_"):
            return "target"
        if feature_name.startswith("PATHWAY_NAME_"):
            return "pathway"
        if feature_name.startswith("DRUG_NAME_"):
            return "drug_identity"
        if feature_name.startswith("OncotreeLineage_"):
            return "tissue"
        return "gene_expression"

    def evaluate_config(
        self,
        config_name: str,
        config: FeatureConfig,
        model_name: str,
        split_strategy: str,
        test_size: float,
        random_state: int,
        min_samples_per_drug: int,
        device: str,
        min_group_samples: int,
        shap_samples: int,
        tune_xgboost: bool = False,
    ) -> Dict[str, object]:
        filtered = self.filtered_response(min_samples_per_drug)
        train_df, test_df, split_meta = self.split_response(filtered, split_strategy, test_size, random_state)

        print("\n" + "=" * 60)
        print(f"Running config: {config_name}")
        print(f"Model: {model_name}")
        print(f"Split strategy: {split_strategy}")
        print("=" * 60)
        print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

        train_x, test_x, feature_names = self.prepare_features(train_df, test_df, config)
        tuning_results = pd.DataFrame()
        best_xgboost_params: Dict[str, object] | None = None
        if model_name == "xgboost" and tune_xgboost:
            best_xgboost_params, tuning_results = self.tune_xgboost_on_leave_drug_out(
                train_df=train_df,
                config=config,
                device=device,
                random_state=random_state,
            )
            print(f"Best leave-drug-out tuned XGBoost params: {best_xgboost_params}")

        model = self._build_baseline_model(
            model_name=model_name,
            device=device,
            random_state=random_state,
            xgboost_params=best_xgboost_params,
        )
        model.fit(train_x, train_df["AUC"].to_numpy())

        y_test = test_df["AUC"].to_numpy(dtype=float)
        y_pred = model.predict(test_x)

        overall_metrics = regression_metrics(y_test, y_pred)
        baseline_metrics = {
            name: regression_metrics(y_test, preds)
            for name, preds in self.baseline_predictions(train_df, test_df).items()
        }
        calibration_stats, calibration_table = calibration_summary(y_test, y_pred)

        prediction_frame = test_df[
            ["DRUG_NAME", "ModelID", "OncotreeLineage", "OncotreePrimaryDisease", "AUC"]
        ].copy()
        prediction_frame["prediction"] = y_pred

        per_drug = grouped_regression_metrics(
            prediction_frame,
            group_col="DRUG_NAME",
            min_samples=min_group_samples,
        )
        per_tissue = grouped_regression_metrics(
            prediction_frame,
            group_col="OncotreeLineage",
            min_samples=min_group_samples,
        )

        sample_size_effect = float("nan")
        if not per_drug.empty and per_drug["n_samples"].nunique() > 1:
            sample_size_effect = _safe_correlation(
                pearsonr,
                per_drug["n_samples"].to_numpy(dtype=float),
                per_drug["r2"].to_numpy(dtype=float),
            )

        importance = self._model_feature_importance(model, model_name, feature_names)
        importance["modality"] = importance["feature"].map(self._feature_modality)

        shap_outputs = {}
        if model_name == "xgboost" and shap_samples > 0:
            print(f"Computing native XGBoost SHAP contributions on up to {shap_samples:,} test rows...")
            shap_outputs = self._xgboost_shap_outputs(
                model=model,
                test_x=test_x,
                feature_names=feature_names,
                prediction_frame=prediction_frame,
                max_samples=shap_samples,
                random_state=random_state,
            )

        summary = {
            "config_name": config_name,
            "model_name": model_name,
            "feature_config": asdict(config),
            "split_strategy": split_strategy,
            "split": split_meta,
            "overall_metrics": overall_metrics,
            "baseline_metrics": baseline_metrics,
            "calibration": calibration_stats,
            "per_drug_sample_size_r2_correlation": sample_size_effect,
            "n_features": int(len(feature_names)),
            "shap_samples": int(shap_samples if model_name == "xgboost" else 0),
            "tuned_on_leave_drug_out": bool(model_name == "xgboost" and tune_xgboost),
            "best_xgboost_params": best_xgboost_params or {},
        }

        result = {
            "summary": summary,
            "predictions": prediction_frame,
            "per_drug": per_drug,
            "per_tissue": per_tissue,
            "calibration_table": calibration_table,
            "feature_importance": importance,
            "tuning_results": tuning_results,
        }
        result.update(shap_outputs)
        return result

    def save_result(self, split_strategy: str, model_name: str, config_name: str, result: Dict[str, object]) -> None:
        run_dir = self.output_dir / split_strategy / model_name / config_name
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(result["summary"], handle, indent=2)

        result["predictions"].to_csv(run_dir / "predictions.csv", index=False)
        result["per_drug"].to_csv(run_dir / "per_drug_metrics.csv", index=False)
        result["per_tissue"].to_csv(run_dir / "per_tissue_metrics.csv", index=False)
        result["calibration_table"].to_csv(run_dir / "calibration.csv", index=False)
        result["feature_importance"].to_csv(run_dir / "feature_importance.csv", index=False)
        if "tuning_results" in result and not result["tuning_results"].empty:
            result["tuning_results"].to_csv(run_dir / "leave_drug_out_tuning.csv", index=False)
        if "shap_global" in result:
            result["shap_global"].to_csv(run_dir / "shap_global.csv", index=False)
            result["shap_modality"].to_csv(run_dir / "shap_modality.csv", index=False)
            result["shap_local_top"].to_csv(run_dir / "shap_local_top.csv", index=False)

    def run_suite(
        self,
        configs: Dict[str, FeatureConfig],
        model_names: Sequence[str],
        split_strategy: str,
        test_size: float,
        random_state: int,
        min_samples_per_drug: int,
        device: str,
        min_group_samples: int,
        shap_samples: int,
        tune_xgboost: bool = False,
    ) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for model_name in model_names:
            for config_name, config in configs.items():
                result = self.evaluate_config(
                    config_name=config_name,
                    config=config,
                    model_name=model_name,
                    split_strategy=split_strategy,
                    test_size=test_size,
                    random_state=random_state,
                    min_samples_per_drug=min_samples_per_drug,
                    device=device,
                    min_group_samples=min_group_samples,
                    shap_samples=shap_samples,
                    tune_xgboost=tune_xgboost,
                )
                self.save_result(split_strategy, model_name, config_name, result)

                summary = result["summary"]
                overall = summary["overall_metrics"]
                baselines = summary["baseline_metrics"]
                rows.append(
                    {
                        "config": config_name,
                        "model": model_name,
                        "split_strategy": split_strategy,
                        "test_r2": overall["r2"],
                        "test_rmse": overall["rmse"],
                        "test_mae": overall["mae"],
                        "test_pearson": overall["pearson"],
                        "test_spearman": overall["spearman"],
                        "global_mean_r2": baselines["global_mean"]["r2"],
                        "per_drug_mean_r2": baselines["per_drug_mean"]["r2"],
                        "tissue_mean_r2": baselines["tissue_mean"]["r2"],
                        "calibration_slope": summary["calibration"]["calibration_slope"],
                        "calibration_intercept": summary["calibration"]["calibration_intercept"],
                        "sample_size_r2_corr": summary["per_drug_sample_size_r2_correlation"],
                        "n_features": summary["n_features"],
                        "tuned_on_leave_drug_out": summary["tuned_on_leave_drug_out"],
                        "train_samples": summary["split"]["train_samples"],
                        "test_samples": summary["split"]["test_samples"],
                        "discarded_samples": summary["split"]["discarded_samples"],
                    }
                )

        summary_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
        summary_dir = self.output_dir / split_strategy
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_dir / "summary.csv", index=False)
        return summary_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reviewer-focused drug sensitivity benchmarks.")
    parser.add_argument(
        "--split",
        default="stratified_random",
        choices=["stratified_random", "leave_cell_out", "leave_drug_out", "leave_both_out", "all"],
        help="Evaluation split strategy. Use 'all' for all reviewer regimes.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Single config name from the reviewer ablation suite. If omitted, the full suite runs.",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "ridge", "elasticnet", "random_forest", "lightgbm", "all"],
        help="Model family to train. Use 'all' for all implemented model baselines.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction for supported splits.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-samples-per-drug", type=int, default=100, help="Minimum samples per drug.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="XGBoost device.")
    parser.add_argument("--min-group-samples", type=int, default=10, help="Minimum samples for grouped metrics.")
    parser.add_argument(
        "--tune-xgboost",
        action="store_true",
        help="Tune XGBoost hyperparameters on an inner leave-drug-out validation split.",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=1000,
        help="Number of test rows for native XGBoost SHAP contribution exports. Use 0 to disable.",
    )
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument(
        "--output-dir",
        default="analysis_results/benchmarks",
        help="Directory where benchmark outputs should be saved.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configs = reviewer_ablation_configs()
    if args.config:
        if args.config not in configs:
            raise ValueError(f"Unknown config '{args.config}'. Available configs: {', '.join(sorted(configs))}")
        configs = {args.config: configs[args.config]}

    if args.model == "all":
        model_names = ["xgboost", "ridge", "elasticnet", "random_forest", "lightgbm"]
    else:
        model_names = [args.model]

    if args.split == "all":
        split_strategies = ["stratified_random", "leave_cell_out", "leave_drug_out", "leave_both_out"]
    else:
        split_strategies = [args.split]

    suite = ReviewerBenchmarkSuite(data_dir=args.data_dir, output_dir=args.output_dir)
    summaries = []
    for split_strategy in split_strategies:
        summaries.append(
            suite.run_suite(
                configs=configs,
                model_names=model_names,
                split_strategy=split_strategy,
                test_size=args.test_size,
                random_state=args.random_state,
                min_samples_per_drug=args.min_samples_per_drug,
                device=args.device,
                min_group_samples=args.min_group_samples,
                shap_samples=args.shap_samples,
                tune_xgboost=args.tune_xgboost,
            )
        )
    summary_df = pd.concat(summaries, ignore_index=True)

    print("\n" + "=" * 60)
    print("Benchmark summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
