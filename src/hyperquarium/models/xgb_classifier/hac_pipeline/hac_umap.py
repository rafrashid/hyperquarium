"""
hac_umap.py -- Standalone UMAP visualisation for HAC pipeline outputs.
------------------------------------------------------------------
Reads X_pca.npy (or scaled features for --no-pca branch) and
pixel_clusters.parquet saved by hac.py. Run locally after hac.py
has completed on HPC.

Usage:
    python3 hac_umap.py --spectra A
    python3 hac_umap.py --spectra A --overwrite false
    python3 hac_umap.py --spectra C --no-pca
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
    p.add_argument(
        "--no-pca", dest="no_pca", action="store_true", default=False,
        help="Load from spectra_C_no_pca output directory. "
             "Only permitted with --spectra C.",
    )
    return p.parse_args()


def should_run(output_path: Path, overwrite: bool) -> bool:
    if overwrite:
        return True
    if output_path.exists():
        logger.info(f"Checkpoint: skipping (exists) - {output_path.name}")
        return False
    return True


def main() -> None:
    args = parse_args()
    overwrite = args.overwrite.lower() == "true"
    no_pca = args.no_pca

    if no_pca and args.spectra != "C":
        raise ValueError(
            f"--no-pca is only permitted with --spectra C (got --spectra {args.spectra})."
        )

    cfg = HACConfig()
    if no_pca:
        cfg.output_dir = args.hac_output / "spectra_C_no_pca"
    else:
        cfg.output_dir = args.hac_output / f"spectra_{args.spectra}"
    out = cfg.output_dir

    setup_logger(out, spectra=f"{args.spectra}{'_no_pca' if no_pca else ''}_umap")
    logger.info(f"=== UMAP --- Spectra {args.spectra}{' (no PCA)' if no_pca else ''} ===")
    logger.info(f"Output dir: {out}")
    logger.info(f"Overwrite:  {overwrite}")

    # Load feature matrix — X_pca.npy for PCA branch, absent for no-pca branch
    # In no-pca branch, hac.py does not save the scaled matrix to disk.
    # UMAP is run on X_pca.npy regardless (it is the input to linkage in both cases).
    xpca_path = out / "X_pca.npy"
    if not xpca_path.exists():
        raise FileNotFoundError(
            f"X_pca.npy not found at {xpca_path}.\n"
            f"Run hac.py {'--spectra C --no-pca' if no_pca else f'--spectra {args.spectra}'} "
            f"first to generate pipeline outputs."
        )
    logger.info(f"Loading feature matrix: {xpca_path}")
    X_input = np.load(xpca_path)

    logger.info("Loading pixel_clusters.parquet.")
    pixel_df = pd.read_parquet(out / "pixel_clusters.parquet")
    roi_ids = (pixel_df[pixel_df["k"] == pixel_df["k"].iloc[0]]
               .sort_values("pixel_idx")["roi_ID"]
               .reset_index(drop=True))

    silhouette_data = load_json(out / "silhouette_scores.json")
    best_k = silhouette_data["best_k"]
    k_rois = roi_ids.nunique()
    k_values = sorted(set(cfg.k_values + [best_k, k_rois]))
    logger.info(f"K values: {k_values}")

    for k in k_values:
        if should_run(out / f"umap_k{k}.png", overwrite):
            plot_umap(
                X_pca=X_input,
                roi_ids=roi_ids,
                pixel_df=pixel_df,
                k=k,
                output_dir=out,
            )

    logger.info(f"=== UMAP complete --- Spectra {args.spectra}"
                f"{' (no PCA)' if no_pca else ''} ===")


if __name__ == "__main__":
    main()