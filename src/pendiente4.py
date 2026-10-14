"""
pendiente4.py
-------------
Pendiente 4: Optimización del circuito con N filas (barrido COBYLA).

Pregunta: ¿cuántas filas de entrenamiento necesita el optimizador
para que θ sea útil en el test completo?

Flujo por cada N en N_rows:
  1. Tomar N filas del train (seed fija → reproducible)
  2. Optimizar θ con COBYLA (reutiliza optimize_theta de phase2b)
  3. Aplicar θ_opt al test COMPLETO → ⟨Z⟩
  4. Concatenar [X_4 | ⟨Z⟩_4] y evaluar con XGBoost
  5. Guardar métricas + curva de convergencia

Reutiliza:
  - phase2b_optimize.optimize_theta   ← ya implementado
  - quantum_utils.build_circuit, get_qfeatures
  - io_utils.save_json, save_npz, ensure_dirs
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils  import load_config, get_paths
from src.utils.io_utils      import load_npz, save_json, save_npz, ensure_dirs
from src.utils.log_utils     import get_logger
from src.utils.quantum_utils import build_circuit, get_qfeatures, init_theta
from src.phase2b_optimize    import optimize_theta   # ← reutilizar directamente

log = get_logger("pendiente4")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (idénticos a pendiente3 — no duplicar en módulo aparte)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2"  : float(r2_score(y_true, y_pred)),
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def select_first_n(X: np.ndarray, n: int) -> np.ndarray:
    return X[:, :n]


def subsample(X: np.ndarray, y: np.ndarray, N: int, seed: int):
    """Toma N filas aleatorias reproducibles de (X, y)."""
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(X), size=min(N, len(X)), replace=False)
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_one_N(
    N           : int,
    X_train     : np.ndarray,
    X_val       : np.ndarray,
    X_test      : np.ndarray,
    y_train     : np.ndarray,
    y_val       : np.ndarray,
    y_test      : np.ndarray,
    nq          : int,
    L           : int,
    enc         : str,
    seed        : int,
    max_iter    : int,
    results_dir : Path,
    xgb_params  : dict,
) -> dict:
    """Corre optimización + evaluación para un valor de N."""

    log.info(f"\n  ── N={N} filas ──")

    # 1. Tomar primeras nq columnas
    Xtr4 = select_first_n(X_train, nq)
    Xvl4 = select_first_n(X_val,   nq)
    Xte4 = select_first_n(X_test,  nq)

    # 2. Submuestra de N filas para optimización
    Xsub, ysub = subsample(Xtr4, y_train, N, seed)

    # 3. Normalizar a [-π, π] (fit sobre submuestra)
    X_min = Xsub.min(axis=0)
    X_max = Xsub.max(axis=0)
    rng_  = X_max - X_min
    rng_[rng_ == 0] = 1.0

    def norm(X):
        return (2.0 * np.pi * (X - X_min) / rng_ - np.pi).astype(np.float32)

    Xsub_n = norm(Xsub)
    Xvl4_n = norm(Xvl4)
    Xte4_n = norm(Xte4)
    Xtr4_n = norm(Xtr4)   # train completo normalizado (para XGBoost)

    # 4. Optimizar θ con COBYLA sobre N filas
    log.info(f"    Optimizando θ con {N} filas, max_iter={max_iter}...")
    theta_opt, history = optimize_theta(
        nq, L, enc,
        X_train_n = Xsub_n,
        y_train   = ysub,
        seed      = seed,
        max_iter  = max_iter,
    )

    # 5. Aplicar θ_opt al dataset completo → ⟨Z⟩
    circuit = build_circuit(nq, L, enc)

    log.info(f"    Generando features con θ_opt (train completo)...")
    Ztr = get_qfeatures(Xtr4_n, circuit, theta_opt, verbose=False)
    Zvl = get_qfeatures(Xvl4_n, circuit, theta_opt, verbose=False)
    Zte = get_qfeatures(Xte4_n, circuit, theta_opt, verbose=False)

    # 6. Concatenar [X_4 | ⟨Z⟩_4]
    Ftr = np.hstack([Xtr4, Ztr])
    Fvl = np.hstack([Xvl4, Zvl])
    Fte = np.hstack([Xte4, Zte])

    # 7. Entrenar XGBoost
    model = XGBRegressor(**xgb_params)
    model.fit(Ftr, y_train, eval_set=[(Fvl, y_val)], verbose=False)

    # 8. Métricas
    metrics = {
        "config": {
            "N_rows"  : N,
            "nq"      : nq,
            "L"       : L,
            "encoding": enc,
            "seed"    : seed,
            "max_iter": max_iter,
        },
        "train"      : compute_metrics(y_train, model.predict(Ftr)),
        "val"        : compute_metrics(y_val,   model.predict(Fvl)),
        "test"       : compute_metrics(y_test,  model.predict(Fte)),
        "n_iter_conv": len(history),
        "mse_final"  : float(history[-1]) if history else None,
    }

    log.info(
        f"    R² test={metrics['test']['r2']:.4f} | "
        f"MAE={metrics['test']['mae']:.2f} | "
        f"RMSE={metrics['test']['rmse']:.2f}"
    )

    # 9. Guardar métricas y curva de convergencia
    save_json(results_dir / f"metrics_N{N}.json", metrics)

    conv_dir = results_dir / "convergence_curves"
    ensure_dirs(conv_dir)
    save_npz(conv_dir / f"conv_N{N}.npz", history=np.array(history))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("PENDIENTE 4 — Optimización con N filas (barrido COBYLA)")
    log.info("=" * 60)

    p4       = cfg["pendiente4"]
    nq       = int(p4["nq"])
    L        = int(p4["layers"])
    enc      = str(p4["encoding"])
    N_rows   = p4["N_rows"]              # [10, 25, 50, 100, 200]
    max_iter = int(p4.get("max_iter", 100))
    seed     = int(p4.get("seed", 42))
    res_dir  = Path(p4["results_dir"])
    ensure_dirs(res_dir)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]

    # Cargar datos
    splits  = load_npz(proc_dir / "splits.npz")
    reduced = load_npz(proc_dir / "reduced.npz")

    # Usar features originales (sin PCA) para P4
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]

    # Verificar columnas suficientes
    assert X_train.shape[1] >= nq, (
        f"X_train tiene {X_train.shape[1]} columnas, se necesitan {nq}"
    )

    # Parámetros XGBoost
    xgb_cfg = cfg.get("xgboost", {})
    xgb_p   = {
        "n_estimators"    : int(xgb_cfg.get("n_estimators", 300)),
        "max_depth"       : int(xgb_cfg.get("max_depth", 6)),
        "learning_rate"   : float(xgb_cfg.get("learning_rate", 0.05)),
        "subsample"       : float(xgb_cfg.get("subsample", 0.8)),
        "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.8)),
        "random_state"    : 42,
        "verbosity"       : 0,
    }

    all_metrics = []

    for N in N_rows:
        m = run_one_N(
            N=N,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            nq=nq, L=L, enc=enc,
            seed=seed, max_iter=max_iter,
            results_dir=res_dir,
            xgb_params=xgb_p,
        )
        all_metrics.append(m)

    # Resumen global
    save_json(res_dir / "summary_p4.json", {"results": all_metrics})

    # Tabla resumen en consola
    log.info("\n── Resumen barrido N ──")
    log.info(f"{'N':>6} | {'R²_test':>8} | {'MAE_test':>9} | {'RMSE_test':>10}")
    log.info("-" * 42)
    for m in all_metrics:
        log.info(
            f"{m['config']['N_rows']:>6} | "
            f"{m['test']['r2']:>8.4f} | "
            f"{m['test']['mae']:>9.2f} | "
            f"{m['test']['rmse']:>10.2f}"
        )

    log.info(f"\nPENDIENTE 4 completado ✓ — {len(all_metrics)} valores de N")
    log.info(f"Resultados en: {res_dir}")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)