"""
Run repeated sampled PRISM two-tower training chunks.

This gives a feasible laptop-scale approximation to full PRISM training by
training multiple independent PRISM samples and aggregating their metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

from drug_cell_interaction_model import train_model


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


def run_chunks(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    summaries = []
    for chunk_idx in range(args.n_chunks):
        seed = args.random_state + chunk_idx
        chunk_dir = output_dir / f"chunk_{chunk_idx + 1:02d}"
        print("\n" + "=" * 80)
        print(f"Training PRISM chunk {chunk_idx + 1}/{args.n_chunks} with seed={seed}")
        print("=" * 80)

        train_args = SimpleNamespace(
            data_dir=args.data_dir,
            new_data_dir=args.new_data_dir,
            output_dir=str(chunk_dir),
            include_external=args.include_external,
            max_external_rows=args.prism_rows_per_chunk,
            min_samples_per_drug=args.min_samples_per_drug,
            split=args.split,
            test_size=args.test_size,
            top_genes=args.top_genes,
            fingerprint_bits=args.fingerprint_bits,
            fingerprint_radius=args.fingerprint_radius,
            include_omics=args.include_omics,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            random_state=seed,
        )
        summary = train_model(train_args)
        summaries.append(summary)
        best = summary["best_metrics"]
        final = summary["final_metrics"]
        rows.append(
            {
                "chunk": chunk_idx + 1,
                "seed": seed,
                "rows": summary["pool_rows"],
                "best_epoch": summary["best_epoch"],
                "best_r2": best["r2"],
                "best_rmse": best["rmse"],
                "best_mae": best["mae"],
                "best_pearson": best["pearson"],
                "best_spearman": best["spearman"],
                "final_r2": final["r2"],
                "final_pearson": final["pearson"],
                "final_spearman": final["spearman"],
                "cell_dim": summary["cell_dim"],
                "drug_dim": summary["drug_dim"],
                "chunk_dir": str(chunk_dir),
            }
        )

    chunk_metrics = pd.DataFrame(rows)
    chunk_metrics.to_csv(output_dir / "chunk_metrics.csv", index=False)

    aggregate = {
        "n_chunks": args.n_chunks,
        "prism_rows_per_chunk": args.prism_rows_per_chunk,
        "include_external": args.include_external,
        "include_omics": args.include_omics,
        "metrics_mean": {
            col: float(chunk_metrics[col].mean())
            for col in ["best_r2", "best_rmse", "best_mae", "best_pearson", "best_spearman"]
        },
        "metrics_std": {
            col: float(chunk_metrics[col].std(ddof=1)) if len(chunk_metrics) > 1 else 0.0
            for col in ["best_r2", "best_rmse", "best_mae", "best_pearson", "best_spearman"]
        },
        "best_chunk": chunk_metrics.sort_values("best_r2", ascending=False).iloc[0].to_dict(),
    }
    with open(output_dir / "aggregate_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_safe_json(aggregate), handle, indent=2)

    print("\nChunked PRISM aggregate")
    print(chunk_metrics[["chunk", "best_epoch", "best_r2", "best_pearson", "best_spearman"]].to_string(index=False))
    print(
        f"mean best R2={aggregate['metrics_mean']['best_r2']:.4f} "
        f"+/- {aggregate['metrics_std']['best_r2']:.4f}"
    )
    return aggregate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeated sampled PRISM two-tower training.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--new-data-dir", default="data/new")
    parser.add_argument("--output-dir", default="analysis_results/chunked_prism_training")
    parser.add_argument("--include-external", default="prism")
    parser.add_argument("--n-chunks", type=int, default=5)
    parser.add_argument("--prism-rows-per-chunk", type=int, default=10000)
    parser.add_argument("--min-samples-per-drug", type=int, default=700)
    parser.add_argument("--split", default="leave_drug_out")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-genes", type=int, default=1000)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--include-omics", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_chunks(args)


if __name__ == "__main__":
    main()
