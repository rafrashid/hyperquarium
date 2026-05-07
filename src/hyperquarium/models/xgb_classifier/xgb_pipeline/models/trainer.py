"""
models/trainer.py
XGBoost model training with early stopping, evals tracking, and model persistence.
Handles both weighted and unweighted runs for the Level 3 hypothesis baseline.
"""

import logging
from pathlib import Path

import xgboost as xgb
from config.config import XGBConfig, LevelConfig, XGB
from sklearn.preprocessing import LabelEncoder
from utils.io import save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter assembly
# ---------------------------------------------------------------------------

def build_params(xgb_cfg: XGBConfig, level_cfg: LevelConfig) -> dict:
    """
    Assembles XGBoost parameter dict from config dataclasses.

    Args:
        xgb_cfg:   Global XGBoost hyperparameter config.
        level_cfg: Level-specific config (objective, n_classes, eval_metric).

    Returns:
        Parameter dict ready to pass to xgb.train().
    """
    params = {
        "tree_method": xgb_cfg.tree_method,
        "device": xgb_cfg.device,
        "max_depth": xgb_cfg.max_depth,
        "eta": xgb_cfg.eta,
        "subsample": xgb_cfg.subsample,
        "colsample_bytree": xgb_cfg.colsample_bytree,
        "min_child_weight": xgb_cfg.min_child_weight,
        "gamma": xgb_cfg.gamma,
        "lambda": xgb_cfg.reg_lambda,
        "alpha": xgb_cfg.reg_alpha,
        "objective": level_cfg.objective,
        "eval_metric": level_cfg.eval_metric,  # str or list[str]
        "seed": xgb_cfg.seed,
    }
    if level_cfg.n_classes > 2:
        params["num_class"] = level_cfg.n_classes

    logger.info(f"XGBoost params: {params}")
    return params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
        dtrain: xgb.DMatrix | xgb.QuantileDMatrix,
        dval: xgb.DMatrix | xgb.QuantileDMatrix,
        params: dict,
        xgb_cfg: XGBConfig = XGB,
        run_label: str = "model",
) -> tuple[xgb.Booster, dict]:
    """
    Trains an XGBoost model with early stopping on the validation set.

    Args:
        dtrain:    Training DMatrix.
        dval:      Validation DMatrix (used for early stopping).
        params:    XGBoost parameter dict from build_params().
        xgb_cfg:   XGBConfig for num_boost_round and early_stopping_rounds.
        run_label: Label for logging (e.g. 'spectra_A_level3_weighted').

    Returns:
        (trained Booster, evals_result dict)
    """
    evals_result = {}
    evals = [(dtrain, "train"), (dval, "val")]

    logger.info(f"[{run_label}] Starting training — max rounds: {xgb_cfg.num_boost_round}")

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=xgb_cfg.num_boost_round,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=xgb_cfg.early_stopping_rounds,
        verbose_eval=xgb_cfg.verbose_eval,
    )

    logger.info(
        f"[{run_label}] Training complete — "
        f"best iteration: {booster.best_iteration} | "
        f"best score: {booster.best_score:.5f}"
    )

    return booster, evals_result


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_model(booster: xgb.Booster, out_dir: Path, filename: str = "model.json") -> None:
    """
    Saves a trained Booster to JSON format (portable, human-readable).

    Args:
        booster:  Trained XGBoost Booster.
        out_dir:  Output directory.
        filename: Output filename (default: model.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / filename
    booster.save_model(model_path)
    logger.info(f"Model saved: {model_path}")


def save_training_metadata(
        booster: xgb.Booster,
        evals_result: dict,
        params: dict,
        le: LabelEncoder,
        out_dir: Path,
        weighted: bool = True,
) -> None:
    """
    Saves training metadata (best iteration, best score, params, evals history) to JSON.

    Args:
        booster:      Trained Booster.
        evals_result: Dict of eval metrics per round from xgb.train().
        params:       Parameter dict used for training.
        le:           Fitted LabelEncoder (to record class mapping).
        out_dir:      Output directory.
        weighted:     Whether sample weights were used.
    """
    # If eval_metric is a list, best_score corresponds to the last metric (early stopping target)
    eval_metric = params.get("eval_metric", "unknown")
    primary_metric = eval_metric[-1] if isinstance(eval_metric, list) else eval_metric

    meta = {
        "best_iteration": booster.best_iteration,
        "best_score": booster.best_score,
        "primary_metric": primary_metric,
        "num_features": booster.num_features(),
        "weighted": weighted,
        "class_mapping": {str(i): cls for i, cls in enumerate(le.classes_)},
        "params": params,
    }
    save_json(meta, out_dir / "training_metadata.json")
    save_json(evals_result, out_dir / "evals_result.json")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(model_path: str | Path) -> xgb.Booster:
    """
    Loads a saved XGBoost model from JSON.

    Args:
        model_path: Path to the saved model.json file.

    Returns:
        Loaded Booster.
    """
    booster = xgb.Booster()
    booster.load_model(model_path)
    logger.info(f"Model loaded: {model_path}")
    return booster