"""
Generate manuscript-ready figures from validation and XAI outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("analysis_results")
OUT = ROOT / "manuscript_figures"


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close()


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_metrics(summary: dict, key: str) -> dict:
    if key in summary:
        return summary[key]
    if key == "best_metrics" and "history" in summary:
        return max(summary["history"], key=lambda row: row.get("r2", float("-inf")))
    if "final_metrics" in summary:
        return summary["final_metrics"]
    raise KeyError(key)


def plot_split_performance() -> None:
    rows = []
    split_labels = {
        "stratified_random": "Random\nseen drug",
        "leave_cell_out": "Leave-cell\nout",
        "leave_drug_out": "Leave-drug\nout",
        "leave_both_out": "Leave-both\nout",
    }
    base = ROOT / "revision_benchmarks_gpu"
    for split, label in split_labels.items():
        path = base / split / "summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for config in ["full_with_identity", "full_no_identity", "target_pathway_only", "fingerprint_only"]:
            row = df[df["config"].eq(config)]
            if not row.empty:
                rows.append({"split": label, "config": config, "r2": float(row.iloc[0]["test_r2"])})
    data = pd.DataFrame(rows)
    if data.empty:
        return

    order = list(split_labels.values())
    configs = data["config"].unique().tolist()
    x = np.arange(len(order))
    width = 0.18
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    plt.figure(figsize=(9, 4.8))
    for i, config in enumerate(configs):
        vals = [data[(data["split"].eq(split)) & (data["config"].eq(config))]["r2"].mean() for split in order]
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=config.replace("_", " "), color=colors[i % len(colors)])
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x, order)
    plt.ylabel("R2")
    plt.title("Generalisation Across Validation Regimes")
    plt.legend(frameon=False, fontsize=8)
    savefig("fig1_split_performance")


def plot_unseen_drug_comparison() -> None:
    rows = []
    candidates = [
        ("GDSC rich XGBoost", ROOT / "inductive_rich_full/leave_drug_out/xgboost/inductive_rich/summary.json", "overall_metrics"),
        (
            "GDSC multiomics XGBoost",
            ROOT / "gdsc_multiomics_shap_final/leave_drug_out/xgboost/inductive_multiomics/summary.json",
            "overall_metrics",
        ),
        ("GDSC+CTRP+CCLE XGBoost", ROOT / "multisource_inductive_full/multisource_inductive/summary.json", "metrics"),
        ("Multiomics XGBoost", ROOT / "multisource_multiomics_shap_final/multisource_inductive/summary.json", "metrics"),
        ("Sampled PRISM two-tower", ROOT / "interaction_mlp_prism_sample/interaction_mlp/summary.json", "best_metrics"),
        ("Full PRISM lazy two-tower", ROOT / "lazy_prism_full/lazy_interaction/summary.json", "best_metrics"),
    ]
    for label, path, key in candidates:
        if not path.exists():
            continue
        summary = read_json(path)
        metrics = get_metrics(summary, key)
        rows.append(
            {
                "model": label,
                "r2": metrics["r2"],
                "pearson": metrics["pearson"],
                "spearman": metrics["spearman"],
            }
        )
    data = pd.DataFrame(rows)
    if data.empty:
        return

    x = np.arange(len(data))
    width = 0.25
    plt.figure(figsize=(10, 5.2))
    plt.bar(x - width, data["r2"], width, label="R2", color="#457b9d")
    plt.bar(x, data["pearson"], width, label="Pearson", color="#2a9d8f")
    plt.bar(x + width, data["spearman"], width, label="Spearman", color="#f4a261")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x, data["model"], rotation=30, ha="right")
    plt.ylabel("Metric")
    plt.title("Unseen-Drug Model Comparison")
    plt.legend(frameon=False)
    savefig("fig2_unseen_drug_comparison")


def plot_shap_modality() -> None:
    path = ROOT / "multisource_multiomics_shap_final/multisource_inductive/shap_modality.csv"
    if not path.exists():
        return
    data = pd.read_csv(path).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8, 5.8))
    plt.barh(data["modality"].str.replace("_", " "), data["mean_abs_shap"], color="#264653")
    plt.xlabel("Mean absolute SHAP contribution")
    plt.title("Final Multi-Source Multiomics XGBoost: SHAP by Modality")
    savefig("fig3_shap_modality")


def plot_top_shap_features() -> None:
    path = ROOT / "multisource_multiomics_shap_final/multisource_inductive/shap_global.csv"
    if not path.exists():
        return
    data = pd.read_csv(path).head(20).sort_values("mean_abs_shap", ascending=True)
    labels = data["feature"].str.replace("_", " ", regex=False)
    plt.figure(figsize=(9, 6.5))
    plt.barh(labels, data["mean_abs_shap"], color="#2a9d8f")
    plt.xlabel("Mean absolute SHAP contribution")
    plt.title("Top 20 SHAP Features")
    savefig("fig4_top_shap_features")


def plot_predicted_vs_observed() -> None:
    path = ROOT / "multisource_multiomics_shap_final/multisource_inductive/predictions.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    if len(data) > 60000:
        data = data.sample(n=60000, random_state=42)
    plt.figure(figsize=(6, 5.5))
    plt.scatter(data["AUC"], data["prediction"], s=5, alpha=0.18, color="#1d3557", edgecolors="none")
    low = min(data["AUC"].min(), data["prediction"].min())
    high = max(data["AUC"].max(), data["prediction"].max())
    plt.plot([low, high], [low, high], color="#e76f51", linewidth=1.3)
    plt.xlabel("Observed harmonized AUC/resistance")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Observed: Final Multiomics XGBoost")
    savefig("fig5_predicted_vs_observed")


def plot_per_dataset_performance() -> None:
    sources = [
        ("Multiomics XGBoost", ROOT / "multisource_multiomics_shap_final/multisource_inductive/per_dataset_metrics.csv"),
        ("Full PRISM lazy two-tower", ROOT / "lazy_prism_full/lazy_interaction/per_dataset_metrics.csv"),
    ]
    rows = []
    for model, path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = model
        rows.append(df)
    if not rows:
        return
    data = pd.concat(rows, ignore_index=True)
    data["label"] = data["model"] + "\n" + data["dataset_source"].astype(str)
    x = np.arange(len(data))
    plt.figure(figsize=(9, 5))
    plt.bar(x, data["r2"], color="#8ab17d")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x, data["label"], rotation=30, ha="right")
    plt.ylabel("R2")
    plt.title("Per-Dataset Generalisation Performance")
    savefig("fig6_per_dataset_performance")


def main() -> None:
    ensure_out()
    plot_split_performance()
    plot_unseen_drug_comparison()
    plot_shap_modality()
    plot_top_shap_features()
    plot_predicted_vs_observed()
    plot_per_dataset_performance()
    print(f"Saved figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
