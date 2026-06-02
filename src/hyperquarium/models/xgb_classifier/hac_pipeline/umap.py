"""
umap.py — Standalone UMAP visualisation for HAC pipeline outputs.
------------------------------------------------------------------
Reads X_pca.npy and pixel_clusters.parquet saved by hac.py and
produces umap_k{K}.png for each K value. Run locally after hac.py
has completed on HPC.

Usage:
    python3 umap.py --spectra A
    python3 umap.py --spectra A --overwrite false
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd

from hac_pipeline.config.config import HACConfig
from hac_pipeline.utils.io import load_json
from hac_pipeline.utils.logger import setup_logger
from hac_pipeline.visualisations.plots import plot_umap

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone UMAP plots from saved HAC pipeline outputs."
    )
    p.add_argument("--spectra", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--hac_output", type=Path, default=Path("outputs/hac"))
    p.add_argument("--overwrite", default="true", choices=["true", "false"])
    return p.parse_args()


def should_run(output_path: Path, overwrite: bool) -> bool:
    if overwrite:
        return True
    if output_path.exists():
        logger.info(f"Checkpoint: skipping (exists) — {output_path.name}")
        return False
    return True


def main() -> None:
    args = parse_args()
    overwrite = args.overwrite.lower() == "true"

    cfg = HACConfig()
    cfg.output_dir = args.hac_output / f"spectra_{args.spectra}"
    out = cfg.output_dir

    setup_logger(out, spectra=f"{args.spectra}_umap")
    logger.info(f"=== UMAP — Spectra {args.spectra} ===")
    logger.info(f"Output dir: {out}")
    logger.info(f"Overwrite:  {overwrite}")

    # Load saved outputs
    logger.info("Loading X_pca.npy.")
    X_pca = np.load(out / "X_pca.npy")

    logger.info("Loading pixel_clusters.parquet.")
    pixel_df = pd.read_parquet(out / "pixel_clusters.parquet")
    roi_ids = (pixel_df[pixel_df["k"] == pixel_df["k"].iloc[0]]
               .sort_values("pixel_idx")["roi_ID"]
               .reset_index(drop=True))

    # Determine K values from saved silhouette scores
    silhouette_data = load_json(out / "silhouette_scores.json")
    best_k = silhouette_data["best_k"]
    k_rois = roi_ids.nunique()
    k_values = sorted(set(cfg.k_values + [best_k, k_rois]))
    logger.info(f"K values: {k_values}")

    # Plot UMAP for each K
    for k in k_values:
        if should_run(out / f"umap_k{k}.png", overwrite):
            plot_umap(
                X_pca=X_pca,
                roi_ids=roi_ids,
                pixel_df=pixel_df,
                k=k,
                output_dir=out,
            )

    logger.info(f"=== UMAP complete — Spectra {args.spectra} ===")


if __name__ == "__main__":
    main()
