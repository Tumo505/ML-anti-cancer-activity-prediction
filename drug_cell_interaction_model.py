"""
Two-tower neural drug-cell interaction baseline.

The model trains on mini-batches instead of materializing every interaction as a
huge dense XGBoost matrix, so it is the practical path for sampled PRISM runs.
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
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install torch with CUDA support before running "
            "the neural drug-cell interaction baseline."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


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


class DrugCellInteractionMLP:
    """Factory wrapper so torch imports stay optional until runtime."""

    @staticmethod
    def build(cell_dim: int, drug_dim: int, hidden_dim: int = 256):
        torch, nn, _, _ = require_torch()

        class Model(nn.Module):
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
                interaction_dim = hidden_dim * 2
                self.head = nn.Sequential(
                    nn.Linear(interaction_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, cell_x, drug_x):
                cell_z = self.cell_encoder(cell_x)
                drug_z = self.drug_encoder(drug_x)
                interaction = torch.cat([cell_z, drug_z, cell_z * drug_z, torch.abs(cell_z - drug_z)], dim=1)
                return self.head(interaction).squeeze(-1)

        return Model()


class NeuralFeatureBuilder:
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
        self.cell_imputer = SimpleImputer(strategy="median")
        self.cell_scaler = StandardScaler()
        self.drug_imputer = SimpleImputer(strategy="median")
        self.drug_scaler = StandardScaler()
        self.tissue_encoder = _make_one_hot_encoder()
        self.source_encoder = _make_one_hot_encoder()
        self.target_pathway_encoder = _make_one_hot_encoder()
        self.mechanism_vocab: List[Tuple[str, str]] = []

    @staticmethod
    def _to_dense(matrix) -> np.ndarray:
        if sp.issparse(matrix):
            return matrix.toarray().astype(np.float32)
        return np.asarray(matrix, dtype=np.float32)

    def _cell_raw(self, frame: pd.DataFrame, train_df: pd.DataFrame | None = None, fit: bool = False) -> np.ndarray:
        assert self.suite.expression_df is not None
        expression = self.suite.expression_df.loc[frame["ModelID"], self.selected_genes].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        blocks = [expression]

        if self.include_omics:
            cn_block, _ = self.suite._copy_number_block(frame, self.selected_genes)
            mut_block, _ = self.suite._mutation_block(train_df if train_df is not None else frame, frame, 300)
            fusion_block, _ = self.suite._fusion_block(train_df if train_df is not None else frame, frame, 150)
            rppa_block, _ = self.suite._rppa_block(train_df if train_df is not None else frame, frame, 200)
            blocks.extend([cn_block, mut_block, fusion_block, rppa_block])

        if fit:
            tissue = self.tissue_encoder.fit_transform(frame[["OncotreeLineage"]].fillna("Unknown").astype(str))
            source = self.source_encoder.fit_transform(frame[["dataset_source"]].fillna("Unknown").astype(str))
        else:
            tissue = self.tissue_encoder.transform(frame[["OncotreeLineage"]].fillna("Unknown").astype(str))
            source = self.source_encoder.transform(frame[["dataset_source"]].fillna("Unknown").astype(str))
        blocks.append(self._to_dense(tissue))
        blocks.append(self._to_dense(source))
        return np.hstack([block for block in blocks if block.shape[1] > 0]).astype(np.float32, copy=False)

    def _drug_raw(self, frame: pd.DataFrame, fit: bool = False) -> np.ndarray:
        fp_df = self.suite._ensure_fingerprints(self.fingerprint_bits, self.fingerprint_radius)
        fp_aligned = fp_df.reindex(frame["DRUG_NAME"].tolist())
        structure_missing = fp_aligned.isna().all(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)
        fp_matrix = fp_aligned.fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        descriptor_df = self.suite._ensure_rdkit_descriptors()
        desc_matrix = descriptor_df.reindex(frame["DRUG_NAME"].tolist()).to_numpy(dtype=np.float32, copy=False)

        if fit:
            target_pathway = self.target_pathway_encoder.fit_transform(
                frame[["PUTATIVE_TARGET", "PATHWAY_NAME"]].fillna("Unknown").astype(str)
            )
        else:
            target_pathway = self.target_pathway_encoder.transform(
                frame[["PUTATIVE_TARGET", "PATHWAY_NAME"]].fillna("Unknown").astype(str)
            )

        mechanism = self.suite._mechanism_matrix(frame, self.mechanism_vocab)
        return np.hstack(
            [
                fp_matrix,
                structure_missing,
                desc_matrix,
                self._to_dense(target_pathway),
                self._to_dense(mechanism),
            ]
        ).astype(np.float32, copy=False)

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.selected_genes = self.suite.select_top_genes(train_df, self.top_genes)
        config = FeatureConfig(mechanism_top_terms=200)
        self.mechanism_vocab = self.suite._fit_mechanism_vocab(train_df, config)

        train_cell = self._cell_raw(train_df, train_df=train_df, fit=True)
        test_cell = self._cell_raw(test_df, train_df=train_df, fit=False)
        train_drug = self._drug_raw(train_df, fit=True)
        test_drug = self._drug_raw(test_df, fit=False)

        train_cell = self.cell_scaler.fit_transform(self.cell_imputer.fit_transform(train_cell)).astype(np.float32)
        test_cell = self.cell_scaler.transform(self.cell_imputer.transform(test_cell)).astype(np.float32)
        train_drug = self.drug_scaler.fit_transform(self.drug_imputer.fit_transform(train_drug)).astype(np.float32)
        test_drug = self.drug_scaler.transform(self.drug_imputer.transform(test_drug)).astype(np.float32)
        return train_cell, train_drug, test_cell, test_drug


def grouped_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_name, group in predictions.groupby(group_col):
        if len(group) < 10:
            continue
        row = {group_col: group_name, "n_samples": len(group)}
        row.update(regression_metrics(group["AUC"], group["prediction"]))
        rows.append(row)
    return pd.DataFrame(rows)


def train_model(args: argparse.Namespace) -> Dict[str, object]:
    torch, nn, DataLoader, TensorDataset = require_torch()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    trainer = MultiSourceInductiveTrainer(
        data_dir=args.data_dir,
        new_data_dir=args.new_data_dir,
        output_dir=args.output_dir,
        device="cuda" if args.device == "cuda" else "cpu",
        random_state=args.random_state,
    )
    response = trainer.load_response_pool(
        include_external=parse_external(args.include_external),
        min_samples_per_drug=args.min_samples_per_drug,
        max_external_rows=args.max_external_rows,
    )
    train_df, test_df, split_meta = trainer.suite.split_response(
        response,
        split_strategy=args.split,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    feature_builder = NeuralFeatureBuilder(
        suite=trainer.suite,
        top_genes=args.top_genes,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
        include_omics=args.include_omics,
    )
    train_cell, train_drug, test_cell, test_drug = feature_builder.fit_transform(train_df, test_df)
    y_train = train_df["AUC"].to_numpy(dtype=np.float32)
    y_test = test_df["AUC"].to_numpy(dtype=np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(train_cell),
        torch.from_numpy(train_drug),
        torch.from_numpy(y_train),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_cell),
        torch.from_numpy(test_drug),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = DrugCellInteractionMLP.build(train_cell.shape[1], train_drug.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    best_metric = -np.inf
    best_state = None
    best_epoch = 0
    best_predictions = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for cell_x, drug_x, y in train_loader:
            cell_x = cell_x.to(device, non_blocking=True)
            drug_x = drug_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(cell_x, drug_x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        preds = []
        with torch.no_grad():
            for cell_x, drug_x, _ in test_loader:
                pred = model(cell_x.to(device), drug_x.to(device))
                preds.append(pred.detach().cpu().numpy())
        test_pred = np.concatenate(preds)
        metrics = regression_metrics(y_test, test_pred)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        selection_metric = metrics["r2"] if not np.isnan(metrics["r2"]) else -np.inf
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            best_predictions = test_pred.copy()
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.5f} "
            f"r2={metrics['r2']:.4f} pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f}"
        )

    if best_predictions is None:
        best_predictions = test_pred
    predictions = test_df[["dataset_source", "DRUG_NAME", "ModelID", "AUC"]].copy()
    predictions["prediction"] = best_predictions
    per_dataset = grouped_metrics(predictions, "dataset_source")
    per_drug = grouped_metrics(predictions, "DRUG_NAME")

    out_dir = Path(args.output_dir) / "interaction_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": split_meta,
        "pool_rows": len(response),
        "pool_by_dataset": response["dataset_source"].value_counts().to_dict(),
        "include_external": parse_external(args.include_external),
        "include_omics": args.include_omics,
        "max_external_rows": args.max_external_rows,
        "cell_dim": int(train_cell.shape[1]),
        "drug_dim": int(train_drug.shape[1]),
        "history": history,
        "final_metrics": history[-1],
        "best_epoch": best_epoch,
        "best_metrics": history[best_epoch - 1] if best_epoch else history[-1],
        "device": str(device),
        "random_state": args.random_state,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(_safe_json(summary), handle, indent=2)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    per_dataset.to_csv(out_dir / "per_dataset_metrics.csv", index=False)
    per_drug.to_csv(out_dir / "per_drug_metrics.csv", index=False)
    torch.save(best_state if best_state is not None else model.state_dict(), out_dir / "model.pt")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a two-tower neural drug-cell interaction baseline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--new-data-dir", default="data/new")
    parser.add_argument("--output-dir", default="analysis_results/interaction_mlp")
    parser.add_argument("--include-external", default="prism", help="'none', 'all', or comma-separated ctrp,ccle,prism")
    parser.add_argument("--max-external-rows", type=int, default=5000)
    parser.add_argument("--min-samples-per-drug", type=int, default=700)
    parser.add_argument("--split", default="leave_drug_out", choices=["leave_drug_out", "leave_cell_out", "leave_both_out", "stratified_random"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-genes", type=int, default=1000)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--include-omics", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = train_model(args)
    final = summary["final_metrics"]
    best = summary["best_metrics"]
    print(
        "Final neural interaction result: "
        f"R2={final['r2']:.4f}, Pearson={final['pearson']:.4f}, "
        f"Spearman={final['spearman']:.4f}, RMSE={final['rmse']:.4f}"
    )
    print(
        "Best neural interaction result: "
        f"epoch={summary['best_epoch']}, R2={best['r2']:.4f}, Pearson={best['pearson']:.4f}, "
        f"Spearman={best['spearman']:.4f}, RMSE={best['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
