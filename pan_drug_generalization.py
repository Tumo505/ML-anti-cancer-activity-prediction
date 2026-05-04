"""
Pan-drug generalisation stress tests.

This script adds stronger evidence for pan-drug generalisation than a single
leave-drug-out split by running complementary drug-novelty regimes:

- repeated leave-drug-out splits with different random seeds
- chemical scaffold holdouts using Bemis-Murcko scaffolds
- mechanism holdouts by pathway or target annotation

The goal is not to "prove" generalisation in the mathematical sense, but to
quantify whether model performance persists across multiple definitions of
unseen drugs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

from benchmarking import FeatureConfig, regression_metrics
from multisource_inductive_training import (
    MultiSourceInductiveTrainer,
    build_inductive_feature_config,
    dataset_sample_weights,
    parse_external,
)


FINAL_XGBOOST_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.035,
    "min_child_weight": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.25,
    "reg_lambda": 2.0,
}


FAST_XGBOOST_PARAMS: Dict[str, object] = {
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
}


def safe_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): safe_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def metric_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap row-level confidence intervals for the main regression metrics."""
    point = regression_metrics(y_true, y_pred)
    if n_bootstrap <= 0 or len(y_true) < 10:
        return {name: {"point": value, "ci_low": float("nan"), "ci_high": float("nan")} for name, value in point.items()}

    rng = np.random.default_rng(random_state)
    samples: Dict[str, List[float]] = {name: [] for name in point}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics = regression_metrics(y_true[idx], y_pred[idx])
        for name, value in metrics.items():
            if not np.isnan(value):
                samples[name].append(value)

    intervals: Dict[str, Dict[str, float]] = {}
    for name, value in point.items():
        values = np.asarray(samples[name], dtype=float)
        if len(values) == 0:
            intervals[name] = {"point": value, "ci_low": float("nan"), "ci_high": float("nan")}
        else:
            intervals[name] = {
                "point": value,
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
    return intervals


def summarize_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    return {
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_drugs": int(train_df["DRUG_NAME"].nunique()),
        "test_drugs": int(test_df["DRUG_NAME"].nunique()),
        "train_cells": int(train_df["ModelID"].nunique()),
        "test_cells": int(test_df["ModelID"].nunique()),
        "shared_drugs": int(len(set(train_df["DRUG_NAME"]) & set(test_df["DRUG_NAME"]))),
        "shared_cells": int(len(set(train_df["ModelID"]) & set(test_df["ModelID"]))),
    }


def group_split(
    response: pd.DataFrame,
    group_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(response, groups=response[group_col]))
    return response.iloc[train_idx].copy(), response.iloc[test_idx].copy()


def smiles_lookup(trainer: MultiSourceInductiveTrainer) -> Dict[str, str]:
    assert trainer.suite.pipeline is not None
    if trainer.suite.pipeline.smiles_data is None:
        trainer.suite.pipeline.load_smiles_data()
    assert trainer.suite.pipeline.smiles_data is not None
    lookup: Dict[str, str] = {}
    for _, row in trainer.suite.pipeline.smiles_data.dropna(subset=["DRUG_NAME", "SMILES"]).iterrows():
        name = str(row["DRUG_NAME"]).strip()
        smiles = str(row["SMILES"]).strip().rstrip(",")
        if name and smiles:
            lookup.setdefault(name, smiles)
    return lookup


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold or ""


def add_scaffold_groups(response: pd.DataFrame, trainer: MultiSourceInductiveTrainer) -> Tuple[pd.DataFrame, Dict[str, int]]:
    lookup = smiles_lookup(trainer)
    scaffold_by_drug: Dict[str, str] = {}
    known = 0
    for drug in sorted(response["DRUG_NAME"].astype(str).unique()):
        scaffold = ""
        smiles = lookup.get(drug, "")
        if smiles:
            scaffold = scaffold_from_smiles(smiles)
        if scaffold:
            known += 1
            scaffold_by_drug[drug] = f"scaffold::{scaffold}"
        else:
            scaffold_by_drug[drug] = f"missing_or_acyclic::{drug}"

    out = response.copy()
    out["scaffold_group"] = out["DRUG_NAME"].astype(str).map(scaffold_by_drug)
    meta = {
        "drugs_with_known_scaffold": known,
        "total_drugs": int(response["DRUG_NAME"].nunique()),
        "scaffold_groups": int(out["scaffold_group"].nunique()),
    }
    return out, meta


def build_holdout_splits(
    response: pd.DataFrame,
    trainer: MultiSourceInductiveTrainer,
    regimes: Sequence[str],
    repeats: int,
    test_size: float,
    random_state: int,
) -> List[Tuple[str, int, pd.DataFrame, pd.DataFrame, Dict[str, object]]]:
    splits: List[Tuple[str, int, pd.DataFrame, pd.DataFrame, Dict[str, object]]] = []
    scaffold_response: pd.DataFrame | None = None
    scaffold_meta: Dict[str, int] = {}

    for regime in regimes:
        for repeat in range(repeats):
            seed = random_state + repeat
            meta: Dict[str, object] = {"seed": seed}
            if regime == "repeated_drug":
                train_df, test_df = group_split(response, "DRUG_NAME", test_size, seed)
            elif regime == "scaffold":
                if scaffold_response is None:
                    scaffold_response, scaffold_meta = add_scaffold_groups(response, trainer)
                train_df, test_df = group_split(scaffold_response, "scaffold_group", test_size, seed)
                meta.update(scaffold_meta)
                meta["train_scaffold_groups"] = int(train_df["scaffold_group"].nunique())
                meta["test_scaffold_groups"] = int(test_df["scaffold_group"].nunique())
                meta["shared_scaffold_groups"] = int(len(set(train_df["scaffold_group"]) & set(test_df["scaffold_group"])))
            elif regime == "pathway":
                grouped = response.copy()
                grouped["pathway_group"] = grouped["PATHWAY_NAME"].fillna("Unknown").astype(str)
                train_df, test_df = group_split(grouped, "pathway_group", test_size, seed)
                meta["train_pathways"] = int(train_df["pathway_group"].nunique())
                meta["test_pathways"] = int(test_df["pathway_group"].nunique())
                meta["shared_pathways"] = int(len(set(train_df["pathway_group"]) & set(test_df["pathway_group"])))
            elif regime == "target":
                grouped = response.copy()
                grouped["target_group"] = grouped["PUTATIVE_TARGET"].fillna("Unknown").astype(str)
                train_df, test_df = group_split(grouped, "target_group", test_size, seed)
                meta["train_targets"] = int(train_df["target_group"].nunique())
                meta["test_targets"] = int(test_df["target_group"].nunique())
                meta["shared_targets"] = int(len(set(train_df["target_group"]) & set(test_df["target_group"])))
            else:
                raise ValueError(f"Unknown generalisation regime: {regime}")

            meta.update(summarize_split(train_df, test_df))
            splits.append((regime, repeat + 1, train_df.reset_index(drop=True), test_df.reset_index(drop=True), meta))
    return splits


def evaluate_split(
    trainer: MultiSourceInductiveTrainer,
    config: FeatureConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Dict[str, object],
    device: str,
    source_weighting: str,
    random_state: int,
    n_bootstrap: int,
    save_predictions_path: Path | None,
) -> Dict[str, object]:
    train_x, test_x, feature_names = trainer.suite.prepare_features(train_df, test_df, config)
    model = trainer.suite._build_model(device=device, random_state=random_state, params=params)
    weights = dataset_sample_weights(train_df, source_weighting)
    fit_kwargs = {"sample_weight": weights} if weights is not None else {}
    model.fit(train_x, train_df["AUC"].to_numpy(dtype=float), **fit_kwargs)

    y_true = test_df["AUC"].to_numpy(dtype=float)
    y_pred = model.predict(test_x)
    metrics = regression_metrics(y_true, y_pred)
    intervals = metric_confidence_intervals(y_true, y_pred, n_bootstrap=n_bootstrap, random_state=random_state)

    predictions = test_df[["dataset_source", "DRUG_NAME", "ModelID", "PUTATIVE_TARGET", "PATHWAY_NAME", "AUC"]].copy()
    predictions["prediction"] = y_pred
    per_dataset_rows: List[Dict[str, object]] = []
    for source, group in predictions.groupby("dataset_source"):
        row = {"dataset_source": source, "n_samples": int(len(group))}
        row.update(regression_metrics(group["AUC"], group["prediction"]))
        per_dataset_rows.append(row)

    if save_predictions_path is not None:
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(save_predictions_path, index=False)

    return {
        "metrics": metrics,
        "metric_ci": intervals,
        "n_features": int(len(feature_names)),
        "per_dataset": per_dataset_rows,
    }


def aggregate_results(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["r2", "rmse", "mae", "pearson", "spearman"]
    grouped_rows: List[Dict[str, object]] = []
    for regime, group in rows.groupby("regime"):
        row: Dict[str, object] = {"regime": regime, "n_runs": int(len(group))}
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_sd"] = float("nan")
                row[f"{metric}_min"] = float("nan")
                row[f"{metric}_max"] = float("nan")
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_max"] = float(values.max())
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows).sort_values("regime").reset_index(drop=True)


def write_progress(
    output_dir: Path,
    run_rows: List[Dict[str, object]],
    per_dataset_rows: List[Dict[str, object]],
    summary_extra: Dict[str, object],
) -> None:
    """Persist completed runs after each model so long jobs are resumable."""
    runs = pd.DataFrame(run_rows)
    per_dataset = pd.DataFrame(per_dataset_rows)
    runs.to_csv(output_dir / "runs.csv", index=False)
    per_dataset.to_csv(output_dir / "per_dataset.csv", index=False)

    if runs.empty:
        aggregate = pd.DataFrame()
    else:
        aggregate = aggregate_results(runs)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)

    summary = {
        **summary_extra,
        "completed_runs": len(run_rows),
        "runs": run_rows,
        "aggregate": aggregate.to_dict(orient="records"),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(safe_json(summary), handle, indent=2)


def parse_regimes(value: str) -> List[str]:
    if value.lower() == "all":
        return ["repeated_drug", "scaffold", "pathway", "target"]
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pan-drug generalisation stress tests.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--new-data-dir", default="data/new")
    parser.add_argument("--output-dir", default="analysis_results/pan_drug_generalization")
    parser.add_argument("--include-external", default="ctrp,ccle")
    parser.add_argument("--min-samples-per-drug", type=int, default=100)
    parser.add_argument("--max-external-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--regimes", default="all", help="'all' or comma-separated repeated_drug,scaffold,pathway,target")
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--include-omics", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--source-weighting", default="sqrt_inverse_dataset", choices=["none", "equal_dataset", "sqrt_inverse_dataset"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--fast", action="store_true", help="Use a smaller XGBoost model for smoke tests.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip regime/repeat pairs already present in runs.csv.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = MultiSourceInductiveTrainer(
        data_dir=args.data_dir,
        new_data_dir=args.new_data_dir,
        output_dir=str(output_dir),
        device=args.device,
        random_state=args.random_state,
    )
    config = build_inductive_feature_config(
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
        include_omics=args.include_omics,
    )
    response = trainer.load_response_pool(
        include_external=parse_external(args.include_external),
        min_samples_per_drug=args.min_samples_per_drug,
        max_external_rows=args.max_external_rows,
    )
    regimes = parse_regimes(args.regimes)
    splits = build_holdout_splits(
        response=response,
        trainer=trainer,
        regimes=regimes,
        repeats=args.repeats,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    params = FAST_XGBOOST_PARAMS if args.fast else FINAL_XGBOOST_PARAMS

    run_rows: List[Dict[str, object]] = []
    per_dataset_rows: List[Dict[str, object]] = []
    all_metadata: List[Dict[str, object]] = []

    completed: set[tuple[str, int]] = set()
    if args.resume and (output_dir / "runs.csv").exists():
        previous_runs = pd.read_csv(output_dir / "runs.csv")
        run_rows = previous_runs.to_dict(orient="records")
        completed = {
            (str(row["regime"]), int(row["repeat"]))
            for row in run_rows
            if "regime" in row and "repeat" in row
        }
        if (output_dir / "per_dataset.csv").exists():
            per_dataset_rows = pd.read_csv(output_dir / "per_dataset.csv").to_dict(orient="records")

    summary_extra = {
        "feature_config": config.__dict__,
        "xgboost_params": params,
        "include_external": parse_external(args.include_external),
        "pool_rows": int(len(response)),
        "pool_drugs": int(response["DRUG_NAME"].nunique()),
        "pool_cell_lines": int(response["ModelID"].nunique()),
        "pool_by_dataset": response["dataset_source"].value_counts().to_dict(),
        "regimes": regimes,
        "repeats": args.repeats,
        "test_size": args.test_size,
        "source_weighting": args.source_weighting,
        "bootstrap": args.bootstrap,
    }

    for regime, repeat, train_df, test_df, split_meta in splits:
        if (regime, repeat) in completed:
            print(f"\n[{regime} repeat {repeat}] already completed; skipping because --resume is set.")
            continue
        print(
            f"\n[{regime} repeat {repeat}] "
            f"train={len(train_df):,}, test={len(test_df):,}, "
            f"test_drugs={test_df['DRUG_NAME'].nunique():,}"
        )
        pred_path = None
        if args.save_predictions:
            pred_path = output_dir / regime / f"repeat_{repeat}" / "predictions.csv"
        result = evaluate_split(
            trainer=trainer,
            config=config,
            train_df=train_df,
            test_df=test_df,
            params=params,
            device=args.device,
            source_weighting=args.source_weighting,
            random_state=args.random_state + repeat,
            n_bootstrap=args.bootstrap,
            save_predictions_path=pred_path,
        )
        metrics = result["metrics"]
        row = {
            "regime": regime,
            "repeat": repeat,
            **split_meta,
            "n_features": result["n_features"],
            **metrics,
        }
        for metric, interval in result["metric_ci"].items():
            row[f"{metric}_ci_low"] = interval["ci_low"]
            row[f"{metric}_ci_high"] = interval["ci_high"]
        run_rows.append(row)
        all_metadata.append({"regime": regime, "repeat": repeat, "split": split_meta})
        for dataset_row in result["per_dataset"]:
            per_dataset_rows.append({"regime": regime, "repeat": repeat, **dataset_row})
        write_progress(
            output_dir=output_dir,
            run_rows=run_rows,
            per_dataset_rows=per_dataset_rows,
            summary_extra={**summary_extra, "split_metadata": all_metadata},
        )

    runs = pd.DataFrame(run_rows)
    aggregate = aggregate_results(runs)
    per_dataset = pd.DataFrame(per_dataset_rows)

    runs.to_csv(output_dir / "runs.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    per_dataset.to_csv(output_dir / "per_dataset.csv", index=False)

    summary = {
        "feature_config": config.__dict__,
        "xgboost_params": params,
        "include_external": parse_external(args.include_external),
        "pool_rows": int(len(response)),
        "pool_drugs": int(response["DRUG_NAME"].nunique()),
        "pool_cell_lines": int(response["ModelID"].nunique()),
        "pool_by_dataset": response["dataset_source"].value_counts().to_dict(),
        "regimes": regimes,
        "repeats": args.repeats,
        "test_size": args.test_size,
        "source_weighting": args.source_weighting,
        "bootstrap": args.bootstrap,
        "runs": run_rows,
        "aggregate": aggregate.to_dict(orient="records"),
        "split_metadata": all_metadata,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(safe_json(summary), handle, indent=2)

    print("\nPan-drug generalisation stress-test summary")
    print(aggregate.to_string(index=False))
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
