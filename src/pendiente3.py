"""
pendiente3.py
-------------
Pendiente 3: XGBoost con primeras 4 features + ⟨Z⟩ cuánticas.

Flujo:
  1. Cargar splits.npz y feature_cols.json
  2. Tomar las primeras n_features columnas (orden fijo del CSV)
  3. Para cada dataset (Golden, Golden_PCA):
     Para cada (L, encoding):
       a. Inicializar θ con seed fija
       b. Aplicar circuito → ⟨Z⟩  shape (n_samples, nq)
       c. Concatenar [X_orig_4 | ⟨Z⟩_4]  shape (n_samples, 8)
       d. Entrenar XGBoost
       e. Evaluar R², MAE, RMSE en train / val / test
       f. Guardar métricas en JSON

Reutiliza:
  - quantum_utils.build_circuit, get_qfeatures, init_theta
  - io_utils.save_json, load_npz, load_json, ensure_dirs
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.io_utils   import load_npz, load_json, save_json, ensure_dirs
from src.utils.log_utils  import get_logger
from src.utils.quantum_utils import build_circuit, get_qfeatures, init_theta

log = get_logger("pendiente3")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², MAE, RMSE para un conjunto."""
    return {
        "r2"  : float(r2_score(y_true, y_pred)),
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def normalize_pi(X: np.ndarray) -> np.ndarray:
    """Normaliza X a [-π, π] columna a columna."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    rng   = X_max - X_min
    rng[rng == 0] = 1.0
    return (2.0 * np.pi * (X - X_min) / rng - np.pi).astype(np.float32)


def get_xgb_params(cfg: dict) -> dict:
    """Lee hiperparámetros XGBoost del config o usa defaults."""
    xgb_cfg = cfg.get("xgboost", {})
    return {
        "n_estimators"     : int(xgb_cfg.get("n_estimators", 300)),
        "max_depth"        : int(xgb_cfg.get("max_depth", 6)),
        "learning_rate"    : float(xgb_cfg.get("learning_rate", 0.05)),
        "subsample"        : float(xgb_cfg.get("subsample", 0.8)),
        "colsample_bytree" : float(xgb_cfg.get("colsample_bytree", 0.8)),
        "random_state"     : 42,
        "verbosity"        : 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(proc_dir: Path) -> dict:
    """
    Carga splits.npz y reduced.npz.
    Devuelve dict con claves:
      X_train_r, X_val_r, X_test_r   ← features reducidas (PCA/UMAP/none)
      y_train, y_val, y_test
      X_train_raw, X_val_raw, X_test_raw  ← features originales (antes de reducción)
    """
    splits  = load_npz(proc_dir / "splits.npz")
    reduced = load_npz(proc_dir / "reduced.npz")

    return {
        # targets
        "y_train"   : splits["y_train"],
        "y_val"     : splits["y_val"],
        "y_test"    : splits["y_test"],
        # features reducidas (Golden_PCA)
        "X_train_r" : reduced["X_train_r"],
        "X_val_r"   : reduced["X_val_r"],
        "X_test_r"  : reduced["X_test_r"],
        # features originales (Golden) — guardadas en splits
        "X_train"   : splits["X_train"],
        "X_val"     : splits["X_val"],
        "X_test"    : splits["X_test"],
    }


def select_first_n(X: np.ndarray, n: int) -> np.ndarray:
    """Toma las primeras n columnas. Simple, sin módulo nuevo."""
    return X[:, :n]


# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    dataset_name : str,
    X_train      : np.ndarray,
    X_val        : np.ndarray,
    X_test       : np.ndarray,
    y_train      : np.ndarray,
    y_val        : np.ndarray,
    y_test       : np.ndarray,
    nq           : int,
    L            : int,
    enc          : str,
    seed         : int,
    results_dir  : Path,
    xgb_params   : dict,
) -> dict:
    """
    Corre una configuración completa (dataset × L × enc).
    Devuelve dict de métricas.
    """
    tag = f"{dataset_name}_{nq}q_{L}l_{enc}"
    log.info(f"  → {tag}")

    # 1. Tomar primeras nq columnas
    Xtr4 = select_first_n(X_train, nq)
    Xvl4 = select_first_n(X_val,   nq)
    Xte4 = select_first_n(X_test,  nq)

    # 2. Normalizar a [-π, π] (fit solo en train)
    X_min = Xtr4.min(axis=0)
    X_max = Xtr4.max(axis=0)
    rng_  = X_max - X_min
    rng_[rng_ == 0] = 1.0

    def norm(X):
        return (2.0 * np.pi * (X - X_min) / rng_ - np.pi).astype(np.float32)

    Xtr4_n = norm(Xtr4)
    Xvl4_n = norm(Xvl4)
    Xte4_n = norm(Xte4)

    # 3. Circuito cuántico → ⟨Z⟩
    circuit = build_circuit(nq, L, enc)
    theta   = init_theta(L, nq, seed=seed)

    log.info(f"    Generando features cuánticas (train)...")
    Ztr = get_qfeatures(Xtr4_n, circuit, theta, verbose=True)
    log.info(f"    Generando features cuánticas (val)...")
    Zvl = get_qfeatures(Xvl4_n, circuit, theta, verbose=False)
    log.info(f"    Generando features cuánticas (test)...")
    Zte = get_qfeatures(Xte4_n, circuit, theta, verbose=False)

    # 4. Concatenar [X_orig_4 | ⟨Z⟩_4]
    Ftr = np.hstack([Xtr4, Ztr])   # (n_train, 8)
    Fvl = np.hstack([Xvl4, Zvl])   # (n_val,   8)
    Fte = np.hstack([Xte4, Zte])   # (n_test,  8)

    log.info(f"    Feature matrix: train={Ftr.shape}, val={Fvl.shape}, test={Fte.shape}")

    # 5. Entrenar XGBoost
    model = XGBRegressor(**xgb_params)
    model.fit(Ftr, y_train, eval_set=[(Fvl, y_val)], verbose=False)

    # 6. Métricas
    metrics = {
        "config": {
            "dataset" : dataset_name,
            "nq"      : nq,
            "L"       : L,
            "encoding": enc,
            "seed"    : seed,
        },
        "train": compute_metrics(y_train, model.predict(Ftr)),
        "val"  : compute_metrics(y_val,   model.predict(Fvl)),
        "test" : compute_metrics(y_test,  model.predict(Fte)),
    }

    log.info(
        f"    R² test={metrics['test']['r2']:.4f} | "
        f"MAE={metrics['test']['mae']:.2f} | "
        f"RMSE={metrics['test']['rmse']:.2f}"
    )

    # 7. Guardar métricas
    out_path = results_dir / f"metrics_{tag}.json"
    save_json(out_path, metrics)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("PENDIENTE 3 — XGBoost con primeras 4 features + ⟨Z⟩")
    log.info("=" * 60)

    p3  = cfg["pendiente3"]
    nq  = int(p3["nq"])                    # = 4
    n_f = int(p3["n_features"])            # = 4
    assert nq == n_f, "nq debe ser igual a n_features en P3"

    layers    = p3["layers"]               # [1, 2, 3]
    encodings = p3["encoding"]             # [angle, zz]
    datasets  = p3["datasets"]             # [Golden, Golden_PCA]
    seed      = int(p3.get("seed", 42))
    res_dir   = Path(p3["results_dir"])
    ensure_dirs(res_dir)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    data     = load_splits(proc_dir)
    xgb_p    = get_xgb_params(cfg)

    # Mapeo dataset_name → arrays correctos
    dataset_map = {
        "Golden"    : (data["X_train"],   data["X_val"],   data["X_test"]),
        "Golden_PCA": (data["X_train_r"], data["X_val_r"], data["X_test_r"]),
    }

    all_metrics = []

    for ds_name in datasets:
        if ds_name not in dataset_map:
            log.warning(f"Dataset '{ds_name}' no reconocido, saltando.")
            continue

        X_tr, X_vl, X_te = dataset_map[ds_name]

        # Verificar que hay suficientes columnas
        if X_tr.shape[1] < nq:
            log.warning(
                f"Dataset '{ds_name}' tiene solo {X_tr.shape[1]} columnas, "
                f"se necesitan {nq}. Saltando."
            )
            continue

        for L in layers:
            for enc in encodings:
                m = run_one(
                    dataset_name = ds_name,
                    X_train=X_tr, X_val=X_vl, X_test=X_te,
                    y_train=data["y_train"],
                    y_val  =data["y_val"],
                    y_test =data["y_test"],
                    nq=nq, L=L, enc=enc, seed=seed,
                    results_dir=res_dir,
                    xgb_params=xgb_p,
                )
                all_metrics.append(m)

    # Resumen global
    save_json(res_dir / "summary_p3.json", {"results": all_metrics})
    log.info(f"\nPENDIENTE 3 completado ✓ — {len(all_metrics)} configuraciones")
    log.info(f"Resultados en: {res_dir}")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)