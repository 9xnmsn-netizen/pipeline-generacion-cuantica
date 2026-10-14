"""
learning_curve.py
-----------------
Curva de aprendizaje del modelo híbrido QML-XGBoost.

Migrado de: b1_learning_curve.py

Evalúa R², MAE y RMSE en train/val al variar el tamaño del
conjunto de entrenamiento. Usa la mejor configuración conocida.

Uso:
    from src.analysis import learning_curve
    learning_curve.run(cfg)
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

log = get_logger("learning_curve")


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict, config_name: str = "4q_3l_angle") -> None:
    """
    Ejecuta la curva de aprendizaje.

    Parameters
    ----------
    cfg         : configuración global
    config_name : nombre de la combinación (qfeatures_{config_name}.npz)
    """
    log.info("=" * 60)
    log.info("ANÁLISIS — Curva de aprendizaje")
    log.info("=" * 60)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    res_dir  = Path("data/results/learning_curve")
    ensure_dirs(res_dir)

    # ── Parámetros ────────────────────────────────────────────────────────
    FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 1.00]
    SEED      = 42

    xgb_params = {
        "n_estimators":     300,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     SEED,
        "n_jobs":           -1,
    }

    # ── Carga ─────────────────────────────────────────────────────────────
    log.info(f"Cargando datos para config: {config_name}")

    splits  = load_npz(proc_dir / "splits.npz")
    reduced = load_npz(proc_dir / "reduced.npz")
    qfeat   = load_npz(proc_dir / f"qfeatures_{config_name}.npz")

    y_train = splits["y_train"]
    y_val   = splits["y_val"]

    X_train_full = np.hstack([reduced["X_train_r"], qfeat["X_train_q"]])
    X_val_full   = np.hstack([reduced["X_val_r"],   qfeat["X_val_q"]])

    log.info(f"  X_train shape : {X_train_full.shape}")
    log.info(f"  X_val   shape : {X_val_full.shape}")

    # ── Loop ──────────────────────────────────────────────────────────────
    results = []
    rng = np.random.default_rng(SEED)

    log.info(f"\nEvaluando {len(FRACTIONS)} fracciones...\n")

    for frac in FRACTIONS:
        n_samples = max(10, int(len(X_train_full) * frac))
        idx = rng.choice(len(X_train_full), size=n_samples, replace=False)
        X_sub = X_train_full[idx]
        y_sub = y_train[idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_sub, y_sub, verbose=False)

        m_train = compute_metrics(y_sub, model.predict(X_sub))
        m_val   = compute_metrics(y_val, model.predict(X_val_full))

        row = {
            "fraction":   frac,
            "n_samples":  n_samples,
            "train_r2":   m_train["r2"],
            "train_mae":  m_train["mae"],
            "train_rmse": m_train["rmse"],
            "val_r2":     m_val["r2"],
            "val_mae":    m_val["mae"],
            "val_rmse":   m_val["rmse"],
        }
        results.append(row)

        log.info(
            f"  frac={frac:.2f} n={n_samples:5d} | "
            f"train R²={m_train['r2']:+.4f} val R²={m_val['r2']:+.4f}"
        )

    # ── Guardar JSON ──────────────────────────────────────────────────────
    out_json = res_dir / f"lc_{config_name}.json"
    save_json(out_json, {
        "config":     config_name,
        "timestamp":  datetime.now().isoformat(),
        "xgb_params": xgb_params,
        "results":    results,
    })
    log.info(f"Resultados: {out_json}")

    # ── Gráfica ───────────────────────────────────────────────────────────
    fracs      = [r["fraction"]   for r in results]
    n_samples  = [r["n_samples"]  for r in results]
    train_r2   = [r["train_r2"]   for r in results]
    val_r2     = [r["val_r2"]     for r in results]
    train_mae  = [r["train_mae"]  for r in results]
    val_mae    = [r["val_mae"]    for r in results]
    train_rmse = [r["train_rmse"] for r in results]
    val_rmse   = [r["val_rmse"]   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Curva de Aprendizaje — {config_name}\n"
        f"(XGBoost lr={xgb_params['learning_rate']}, "
        f"n_est={xgb_params['n_estimators']}, seed={SEED})",
        fontsize=13,
    )

    for ax, train_v, val_v, ylabel, title in [
        (axes[0], train_r2,   val_r2,   "R²",           "R² vs Tamaño"),
        (axes[1], train_mae,  val_mae,  "MAE [g/cm²]",  "MAE vs Tamaño"),
        (axes[2], train_rmse, val_rmse, "RMSE [g/cm²]", "RMSE vs Tamaño"),
    ]:
        ax.plot(n_samples, train_v, "o-",  color="steelblue", label="Train")
        ax.plot(n_samples, val_v,   "s--", color="tomato",    label="Val")
        ax.set_xlabel("N muestras de entrenamiento")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig = res_dir / f"lc_{config_name}.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura: {out_fig}")

    # ── Diagnóstico ───────────────────────────────────────────────────────
    gap_r2 = train_r2[-1] - val_r2[-1]
    log.info("\n" + "=" * 60)
    log.info("DIAGNÓSTICO")
    log.info("=" * 60)
    log.info(f"  R² train (100%): {train_r2[-1]:+.4f}")
    log.info(f"  R² val   (100%): {val_r2[-1]:+.4f}")
    log.info(f"  Gap R²         : {gap_r2:+.4f}")

    if gap_r2 > 0.15:
        log.warning("  ⚠️  Alto sobreajuste")
    elif val_r2[-1] < 0.10:
        log.warning("  ⚠️  Subajuste severo")
    elif abs(gap_r2) < 0.05 and val_r2[-1] > 0.30:
        log.info("  ✅ Buen ajuste")

    log.info("\nANÁLISIS completado ✓")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)