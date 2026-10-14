"""
lr_search.py
------------
Búsqueda de learning rate óptimo para XGBoost.

Migrado de: b2_lr_search.py

Uso:
    from src.analysis import lr_search
    lr_search.run(cfg)
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.io_utils     import load_npz, save_json, ensure_dirs
from src.utils.log_utils    import get_logger

log = get_logger("lr_search")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run(cfg: dict, config_name: str = "4q_3l_angle") -> None:
    """
    Ejecuta la búsqueda de learning rate.

    Parameters
    ----------
    cfg         : configuración global
    config_name : nombre de la combinación
    """
    log.info("=" * 60)
    log.info("ANÁLISIS — Búsqueda de Learning Rate")
    log.info("=" * 60)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    res_dir  = Path("data/results/lr_search")
    ensure_dirs(res_dir)

    SEED = 42
    LEARNING_RATES = [0.001, 0.005, 0.01, 0.03, 0.05,
                      0.08, 0.10, 0.15, 0.20, 0.30]

    BASE_PARAMS = {
        "n_estimators":          500,
        "max_depth":             4,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "random_state":          SEED,
        "n_jobs":                -1,
        "early_stopping_rounds": 30,
    }

    # ── Carga ─────────────────────────────────────────────────────────────
    log.info(f"Cargando datos para: {config_name}")

    splits  = load_npz(proc_dir / "splits.npz")
    reduced = load_npz(proc_dir / "reduced.npz")
    qfeat   = load_npz(proc_dir / f"qfeatures_{config_name}.npz")

    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]

    X_train = np.hstack([reduced["X_train_r"], qfeat["X_train_q"]])
    X_val   = np.hstack([reduced["X_val_r"],   qfeat["X_val_q"]])
    X_test  = np.hstack([reduced["X_test_r"],  qfeat["X_test_q"]])

    # ── Loop ──────────────────────────────────────────────────────────────
    all_results = []
    loss_curves = {}

    log.info(f"\nEvaluando {len(LEARNING_RATES)} learning rates...\n")
    log.info(f"{'LR':>8} | {'Best iter':>9} | {'Val R²':>8} | {'Val MAE':>8} | {'Test R²':>8}")
    log.info("-" * 60)

    for lr in LEARNING_RATES:
        params = {**BASE_PARAMS, "learning_rate": lr}

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        best_iter    = model.best_iteration
        y_val_pred   = model.predict(X_val,  iteration_range=(0, best_iter + 1))
        y_test_pred  = model.predict(X_test, iteration_range=(0, best_iter + 1))

        m_val  = compute_metrics(y_val,  y_val_pred)
        m_test = compute_metrics(y_test, y_test_pred)

        evals_result = model.evals_result()
        loss_curves[lr] = evals_result.get("validation_0", {}).get("rmse", [])

        row = {
            "lr":         lr,
            "best_iter":  int(best_iter),
            "val_r2":     m_val["r2"],
            "val_mae":    m_val["mae"],
            "val_rmse":   m_val["rmse"],
            "test_r2":    m_test["r2"],
            "test_mae":   m_test["mae"],
            "test_rmse":  m_test["rmse"],
        }
        all_results.append(row)

        log.info(
            f"  {lr:>6.3f} | {best_iter:>9d} | "
            f"{m_val['r2']:>+8.4f} | {m_val['mae']:>8.2f} | "
            f"{m_test['r2']:>+8.4f}"
        )

    best = max(all_results, key=lambda x: x["val_r2"])
    log.info(f"\n✅ Mejor LR: {best['lr']} (val R²={best['val_r2']:+.4f})")

    # ── Guardar ───────────────────────────────────────────────────────────
    save_json(res_dir / f"lr_search_{config_name}.json", {
        "config":      config_name,
        "timestamp":   datetime.now().isoformat(),
        "best_lr":     best["lr"],
        "results":     all_results,
    })

    # ── Figuras (2 paneles) ───────────────────────────────────────────────
    lrs      = [r["lr"]       for r in all_results]
    val_r2   = [r["val_r2"]   for r in all_results]
    val_mae  = [r["val_mae"]  for r in all_results]
    val_rmse = [r["val_rmse"] for r in all_results]
    test_r2  = [r["test_r2"]  for r in all_results]
    test_mae = [r["test_mae"] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Búsqueda de Learning Rate — {config_name}", fontsize=13)

    for ax, val, test, ylabel, title in [
        (axes[0], val_r2,   test_r2,   "R²",          "R² vs LR"),
        (axes[1], val_mae,  test_mae,  "MAE [g/cm²]", "MAE vs LR"),
        (axes[2], val_rmse, None,      "RMSE [g/cm²]","RMSE vs LR"),
    ]:
        ax.semilogx(lrs, val, "o-", color="steelblue", label="Val")
        if test is not None:
            ax.semilogx(lrs, test, "s--", color="tomato", label="Test")
        ax.axvline(best["lr"], color="green", linestyle=":", lw=1.5,
                   label=f"Mejor LR={best['lr']}")
        ax.set_xlabel("Learning Rate (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(res_dir / f"lr_search_metrics_{config_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Curvas de pérdida
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis
    for (lr, history), color in zip(loss_curves.items(),
                                     [cmap(i / len(LEARNING_RATES))
                                      for i in range(len(LEARNING_RATES))]):
        if history:
            ax.plot(history, label=f"LR={lr}", color=color, lw=1.5)

    ax.set_xlabel("Iteración (n_estimators)")
    ax.set_ylabel("RMSE en validación [g/cm²]")
    ax.set_title(f"Curvas de pérdida — {config_name}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(res_dir / f"lr_loss_curves_{config_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Figuras guardadas en: {res_dir}")
    log.info("\nANÁLISIS completado ✓")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)