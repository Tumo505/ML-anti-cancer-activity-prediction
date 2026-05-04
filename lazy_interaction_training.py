"""
Lazy indexed two-tower training for large PRISM-style interaction matrices.

Instead of materializing one dense row per drug-cell experiment, this script
stores:
- one feature vector per unique cell line
- one feature vector per unique drug
- one feature vector per dataset/source
- integer indices for each observed interaction

That makes sampled or full PRISM training much more feasible on a laptop.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchmarking import FeatureConfig, ReviewerBenchmarkSuite, regression_metrics
from multisource_inductive_training import MultiSourceInductiveTrainer, parse_external


def require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("PyTorch is required for lazy indexed training.") from exc
    return torch, nn, DataLoader, Dataset


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def _dense(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


class IndexedInteractionDataset(require_torch()[3]):
    def __init__(
        self,
        cell_features,
        drug_features,
        source_features,
        cell_idx: np.ndarray,
        drug_idx: np.ndarray,
        source_idx: np.ndarray,
        y: np.ndarray,
    ) -> None:
        torch, _, _, _ = require_torch()
        self.cell_features = torch.from_numpy(cell_features.astype(np.float32, copy=False))
        self.drug_features = torch.from_numpy(drug_features.astype(np.float32, copy=False))
        self.source_features = torch.from_numpy(source_features.astype(np.float32, copy=False))
        self.cell_idx = torch.from_numpy(cell_idx.astype(np.int64, copy=False))
        self.drug_idx = torch.from_numpy(drug_idx.astype(np.int64, copy=False))
        self.source_idx = torch.from_numpy(source_idx.astype(np.int64, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.cell_features[self.cell_idx[idx]],
            self.drug_features[self.drug_idx[idx]],
            self.source_features[self.source_idx[idx]],
            self.y[idx],
        )


class LazyFeatureBuilder:
    def __init__(
        self,
        suite: ReviewerBenchmarkSuite,
        top_genes: int,
        fingerprint_bits: int,
        fingerprint_radius: int,
        include_omics: bool,
    ) -> None:
        self.suite = suite
        self.top_genes = top_genes
        self.fingerprint_bits = fingerprint_bits
        self.fingerprint_radius = fingerprint_radius
        self.include_omics = include_omics
        self.selected_genes: List[str] = []
        self.mechanism_vocab: List[Tuple[str, str]] = []
        self.cell_imputer = SimpleImputer(strategy="median")
        self.cell_scaler = StandardScaler()
        self.drug_imputer = SimpleImputer(strategy="median")
        self.drug_scaler = StandardScaler()
        self.tissue_encoder = _make_one_hot_encoder()
        self.target_pathway_encoder = _make_one_hot_encoder()
        self.source_encoder = _make_one_hot_encoder()

    @staticmethod
    def _unique_cell_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[["ModelID", "OncotreeLineage"]].drop_duplicates("ModelID").reset_index(drop=True)

    @staticmethod
    def _unique_drug_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame[["DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME"]]
            .drop_duplicates("DRUG_NAME")
            .reset_index(drop=True)
        )

    def _cell_raw(self, cell_frame: pd.DataFrame, train_df: pd.DataFrame, fit: bool) -> np.ndarray:
        assert self.suite.expression_df is not None
        expression = self.suite.expression_df.loc[cell_frame["ModelID"], self.selected_genes].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        blocks = [expression]
        if self.include_omics:
            cn, _ = self.suite._copy_number_block(cell_frame, self.selected_genes)
            mut, _ = self.suite._mutation_block(train_df, cell_frame, 300)
            fusion, _ = self.suite._fusion_block(train_df, cell_frame, 150)
            rppa, _ = self.suite._rppa_block(train_df, cell_frame, 200)
            blocks.extend([cn, mut, fusion, rppa])
        tissue = (
            self.tissue_encoder.fit_transform(cell_frame[["OncotreeLineage"]].fillna("Unknown").astype(str))
            if fit
            else self.tissue_encoder.transform(cell_frame[["OncotreeLineage"]].fillna("Unknown").astype(str))
        )
        blocks.append(_dense(tissue))
        return np.hstack([block for block in blocks if block.shape[1] > 0]).astype(np.float32, copy=False)

    def _drug_raw(self, drug_frame: pd.DataFrame, fit: bool) -> np.ndarray:
        fp_df = self.suite._ensure_fingerprints(self.fingerprint_bits, self.fingerprint_radius)
        fp_aligned = fp_df.reindex(drug_frame["DRUG_NAME"].tolist())
        structure_missing = fp_aligned.isna().all(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)
        fp_matrix = fp_aligned.fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        desc = self.suite._ensure_rdkit_descriptors().reindex(drug_frame["DRUG_NAME"].tolist())
        desc_matrix = desc.to_numpy(dtype=np.float32, copy=False)

        target_pathway = (
            self.target_pathway_encoder.fit_transform(
                drug_frame[["PUTATIVE_TARGET", "PATHWAY_NAME"]].fillna("Unknown").astype(str)
            )
            if fit
            else self.target_pathway_encoder.transform(
                drug_frame[["PUTATIVE_TARGET", "PATHWAY_NAME"]].fillna("Unknown").astype(str)
            )
        )
        mechanism = self.suite._mechanism_matrix(drug_frame, self.mechanism_vocab)
        return np.hstack([fp_matrix, structure_missing, desc_matrix, _dense(target_pathway), _dense(mechanism)]).astype(
            np.float32,
            copy=False,
        )

    def build(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, object]:
        self.selected_genes = self.suite.select_top_genes(train_df, self.top_genes)
        self.mechanism_vocab = self.suite._fit_mechanism_vocab(train_df, FeatureConfig(mechanism_top_terms=200))

        train_cells = self._unique_cell_frame(train_df)
        test_cells = self._unique_cell_frame(test_df)
        all_cells = pd.concat([train_cells, test_cells], ignore_index=True).drop_duplicates("ModelID").reset_index(drop=True)
        train_drugs = self._unique_drug_frame(train_df)
        test_drugs = self._unique_drug_frame(test_df)
        all_drugs = pd.concat([train_drugs, test_drugs], ignore_index=True).drop_duplicates("DRUG_NAME").reset_index(drop=True)

        train_cell_raw = self._cell_raw(train_cells, train_df, fit=True)
        self.cell_imputer.fit(train_cell_raw)
        self.cell_scaler.fit(self.cell_imputer.transform(train_cell_raw))
        cell_raw = self._cell_raw(all_cells, train_df, fit=False)
        cell_features = self.cell_scaler.transform(self.cell_imputer.transform(cell_raw)).astype(np.float32)

        train_drug_raw = self._drug_raw(train_drugs, fit=True)
        self.drug_imputer.fit(train_drug_raw)
        self.drug_scaler.fit(self.drug_imputer.transform(train_drug_raw))
        drug_raw = self._drug_raw(all_drugs, fit=False)
        drug_features = self.drug_scaler.transform(self.drug_imputer.transform(drug_raw)).astype(np.float32)

        source_values = train_df[["dataset_source"]].fillna("Unknown").astype(str).drop_duplicates()
        self.source_encoder.fit(source_values)
        all_sources = pd.DataFrame(
            {"dataset_source": sorted(set(train_df["dataset_source"].astype(str)) | set(test_df["dataset_source"].astype(str)))}
        )
        source_features = _dense(self.source_encoder.transform(all_sources[["dataset_source"]].astype(str)))

        cell_lookup = {value: idx for idx, value in enumerate(all_cells["ModelID"])}
        drug_lookup = {value: idx for idx, value in enumerate(all_drugs["DRUG_NAME"])}
        source_lookup = {value: idx for idx, value in enumerate(all_sources["dataset_source"])}

        def indices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            return (
                frame["ModelID"].map(cell_lookup).to_numpy(dtype=np.int64),
                frame["DRUG_NAME"].map(drug_lookup).to_numpy(dtype=np.int64),
                frame["dataset_source"].astype(str).map(source_lookup).to_numpy(dtype=np.int64),
                frame["AUC"].to_numpy(dtype=np.float32),
            )

        train_indices = indices(train_df)
        test_indices = indices(test_df)
        return {
            "cell_features": cell_features,
            "drug_features": drug_features,
            "source_features": source_features.astype(np.float32),
            "train_indices": train_indices,
            "test_indices": test_indices,
            "n_cells": len(all_cells),
            "n_drugs": len(all_drugs),
            "n_sources": len(all_sources),
        }


def build_model(cell_dim: int, drug_dim: int, source_dim: int, hidden_dim: int):
    torch, nn, _, _ = require_torch()

    class LazyTwoTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell_encoder = nn.Sequential(
                nn.Linear(cell_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.drug_encoder = nn.Sequential(
                nn.Linear(drug_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            head_dim = hidden_dim * 2 + source_dim
            self.head = nn.Sequential(
                nn.Linear(head_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, cell_x, drug_x, source_x):
            cell_z = self.cell_encoder(cell_x)
            drug_z = self.drug_encoder(drug_x)
            interaction = torch.cat([cell_z, drug_z, cell_z * drug_z, torch.abs(cell_z - drug_z), source_x], dim=1)
            return self.head(interaction).squeeze(-1)

    return LazyTwoTower()


def evaluate(model, loader, device) -> np.ndarray:
    torch, _, _, _ = require_torch()
    model.eval()
    preds = []
    with torch.no_grad():
        for cell_x, drug_x, source_x, _ in loader:
            pred = model(cell_x.to(device), drug_x.to(device), source_x.to(device))
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds)


def grouped_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_name, group in predictions.groupby(group_col):
        if len(group) < 10:
            continue
        row = {group_col: group_name, "n_samples": len(group)}
        row.update(regression_metrics(group["AUC"], group["prediction"]))
        rows.append(row)
    return pd.DataFrame(rows)


def train(args: argparse.Namespace) -> Dict[str, object]:
    torch, nn, DataLoader, _ = require_torch()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    trainer = MultiSourceInductiveTrainer(
        data_dir=args.data_dir,
        new_data_dir=args.new_data_dir,
        output_dir=args.output_dir,
        device=args.device,
        random_state=args.random_state,
    )
    response = trainer.load_response_pool(
        include_external=parse_external(args.include_external),
        min_samples_per_drug=args.min_samples_per_drug,
        max_external_rows=None if args.max_external_rows == 0 else args.max_external_rows,
    )
    train_df, test_df, split_meta = trainer.suite.split_response(
        response,
        split_strategy=args.split,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    builder = LazyFeatureBuilder(
        suite=trainer.suite,
        top_genes=args.top_genes,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
        include_omics=args.include_omics,
    )
    features = builder.build(train_df, test_df)
    train_dataset = IndexedInteractionDataset(
        features["cell_features"],
        features["drug_features"],
        features["source_features"],
        *features["train_indices"],
    )
    test_dataset = IndexedInteractionDataset(
        features["cell_features"],
        features["drug_features"],
        features["source_features"],
        *features["test_indices"],
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(
        features["cell_features"].shape[1],
        features["drug_features"].shape[1],
        features["source_features"].shape[1],
        args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    y_test = features["test_indices"][3]
    history = []
    best_state = None
    best_pred = None
    best_r2 = -np.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for cell_x, drug_x, source_x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(cell_x.to(device), drug_x.to(device), source_x.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        pred = evaluate(model, test_loader, device)
        metrics = regression_metrics(y_test, pred)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_epoch = epoch
            best_pred = pred.copy()
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.5f} "
            f"r2={metrics['r2']:.4f} pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f}"
        )

    predictions = test_df[["dataset_source", "DRUG_NAME", "ModelID", "AUC"]].copy()
    predictions["prediction"] = best_pred if best_pred is not None else pred
    per_dataset = grouped_metrics(predictions, "dataset_source")
    per_drug = grouped_metrics(predictions, "DRUG_NAME")

    out_dir = Path(args.output_dir) / "lazy_interaction"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": split_meta,
        "pool_rows": len(response),
        "pool_by_dataset": response["dataset_source"].value_counts().to_dict(),
        "include_external": parse_external(args.include_external),
        "max_external_rows": args.max_external_rows,
        "include_omics": args.include_omics,
        "n_cells": features["n_cells"],
        "n_drugs": features["n_drugs"],
        "n_sources": features["n_sources"],
        "cell_dim": int(features["cell_features"].shape[1]),
        "drug_dim": int(features["drug_features"].shape[1]),
        "source_dim": int(features["source_features"].shape[1]),
        "history": history,
        "best_epoch": best_epoch,
        "best_metrics": history[best_epoch - 1] if best_epoch else history[-1],
        "device": str(device),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(_safe_json(summary), handle, indent=2)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    per_dataset.to_csv(out_dir / "per_dataset_metrics.csv", index=False)
    per_drug.to_csv(out_dir / "per_drug_metrics.csv", index=False)
    torch.save(best_state if best_state is not None else model.state_dict(), out_dir / "model.pt")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lazy indexed two-tower training for large interaction datasets.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--new-data-dir", default="data/new")
    parser.add_argument("--output-dir", default="analysis_results/lazy_interaction_training")
    parser.add_argument("--include-external", default="prism")
    parser.add_argument("--max-external-rows", type=int, default=50000)
    parser.add_argument("--min-samples-per-drug", type=int, default=700)
    parser.add_argument("--split", default="leave_drug_out")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-genes", type=int, default=1000)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--include-omics", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = train(args)
    best = summary["best_metrics"]
    print(
        "Best lazy indexed result: "
        f"epoch={summary['best_epoch']}, R2={best['r2']:.4f}, "
        f"Pearson={best['pearson']:.4f}, Spearman={best['spearman']:.4f}, RMSE={best['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
