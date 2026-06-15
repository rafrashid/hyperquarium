#!/bin/bash
#PBS -N bayes_aggregate
#PBS -l select=1:ncpus=4:mem=128gb
#PBS -l walltime=02:00:00
#PBS -j oe

# Aggregation stage only — the big-RAM step that reads the 2.6M/26M-row parquet.
# The 21 fits are light and run locally / on a modest node (no array job needed).
#
# Edit DATASET_PARQUET / MAPPING / OUTPUT per dataset run.

set -euo pipefail

module load python/3.12        # adjust to your HPC module name

export PYTHONPATH="${PBS_O_WORKDIR}:${PYTHONPATH:-}"
cd "${PBS_O_WORKDIR}"

# Activate the dedicated Bayesian venv (separate from xgb_pipeline env).
source /path/to/bayes_venv/bin/activate

DATASET_PARQUET="data/compiled_dataset.parquet"
MAPPING="data/labelset_mapping.csv"
OUTPUT="outputs/bayes"

python scripts/aggregate.py \
    --parquet "${DATASET_PARQUET}" \
    --mapping "${MAPPING}" \
    --output  "${OUTPUT}"
