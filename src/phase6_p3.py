"""
phase6_p3.py
------------
Fase 6 (Pendiente 3): XGBoost con primeras 4 features + ⟨Z⟩ cuánticas.

Migrado de: pendiente3.py

Flujo:
  1. Cargar splits.npz y feature_cols.json
  2. Tomar las primeras n_features columnas (orden fijo del CSV)
  3. Para cada dataset (Golden, Golden_PCA):
     Para cada (L, encoding):
       a. Inicializar θ con seed fija
       b. Aplicar circuito → ⟨Z⟩  shape (n_samples, nq)
       c. Concatenar [X_orig_4 | ⟨Z⟩_4]
       d. Entrenar XGBoost
       e. Evaluar R², MAE, RMSE
       f. Guardar métricas en JSON

Reutiliza:
  - utils/quantum_utils.build_circuit, get_qfeatures, init_theta
  - utils/io_utils.save_json, load_npz, load_json, ensure_dirs
  - utils/feature_utils.normalize_pi

Uso:
    from src import phase6_p3
    phase6_p3.run(cfg)
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils  import load_config, get_paths
from src.utils.io_utils      import load_npz, load_json, save_json, ensure_dirs
from src.utils.log_utils     import get_logger
from src.utils.quantum_utils import build_circuit, get_qfeatures, init_theta
from src.utils.feature_utils import normalize_pi

log = get_logger("phase6_p3")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², MAE, RMSE para un conjunto."""
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def get_xgb_params(cfg: dict) -> dict:
    """Lee hiperparámetros XGBoost del config o usa defaults."""
    xgb_cfg = cfg.get("phase3", {})
    return {
        "n_estimators":     int(xgb_cfg.get("n_estimators", 300)),
        "max_depth":        int(xgb_cfg.get("max_depth", 6)),
        "learning_rate":    float(xgb_cfg.get("learning_rate", 0.05)),
        "subsample":        float(xgb_cfg.get("subsample", 0.8)),
        "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.8)),
        "random_state":     42,
        "verbosity":        0,
    }


def select_first_n(X: np.ndarray, n: int) -> np.ndarray:
    """Toma las primeras n columnas."""
    return X[:, :n]


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(proc_dir: Path) -> dict:
    """
    Carga splits.npz y reduced.npz.

    Returns
    -------
    dict con: y_train, y_val, y_test,
              X_train_r, X_val_r, X_test_r (reducidas),
              X_train, X_val, X_test (originales)
    """
    splits  = load_npz(proc_dir / "splits.npz")
    reduced = load_npz(proc_dir / "reduced.npz")

    return {
        "y_train":   splits["y_train"],
        "y_val":     splits["y_val"],
        "y_test":    splits["y_test"],
        "X_train_r": reduced["X_train_r"],
        "X_val_r":   reduced["X_val_r"],
        "X_test_r":  reduced["X_test_r"],
        "X_train":   splits["X_train"],
        "X_val":     splits["X_val"],
        "X_test":    splits["X_test"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_one(
    dataset_name: str,
    X_train:      np.ndarray,
    X_val:        np.ndarray,
    X_test:       np.ndarray,
    y_train:      np.ndarray,
    y_val:        np.ndarray,
    y_test:       np.ndarray,
    nq:           int,
    L:            int,
    enc:          str,
    seed:         int,
    results_dir:  Path,
    xgb_params:   dict,
) -> dict:
    """Corre una configuración (dataset × L × enc)."""
    tag = f"{dataset_name}_{nq}q_{L}l_{enc}"
    log.info(f"  → {tag}")

    # 1. Primeras nq columnas
    Xtr4 = select_first_n(X_train, nq)
    Xvl4 = select_first_n(X_val,   nq)
    Xte4 = select_first_n(X_test,  nq)

    # 2. Normalizar a [-π, π] (fit sobre train)
    Xtr4_n, X_min, X_max = normalize_pi(Xtr4)
    Xvl4_n, _, _         = normalize_pi(Xvl4, X_min, X_max)
    Xte4_n, _, _         = normalize_pi(Xte4, X_min, X_max)

    # 3. Circuito cuántico → ⟨Z⟩
    circuit = build_circuit(nq, L, enc)
    theta   = init_theta(L, nq, seed=seed)

    log.info(f"    Generando features cuánticas (train)...")
    Ztr = get_qfeatures(Xtr4_n, circuit, theta, verbose=False)
    Zvl = get_qfeatures(Xvl4_n, circuit, theta, verbose=False)
    Zte = get_qfeatures(Xte4_n, circuit, theta, verbose=False)

    # 4. Concatenar [X_orig_4 | ⟨Z⟩_4]
    Ftr = np.hstack([Xtr4, Ztr])
    Fvl = np.hstack([Xvl4, Zvl])
    Fte = np.hstack([Xte4, Zte])

    log.info(f"    Feature matrix: train={Ftr.shape}, val={Fvl.shape}, test={Fte.shape}")

    # 5. Entrenar
    model = XGBRegressor(**xgb_params)
    model.fit(Ftr, y_train, eval_set=[(Fvl, y_val)], verbose=False)

    # 6. Métricas
    metrics = {
        "config": {
            "dataset":  dataset_name,
            "nq":       nq,
            "L":        L,
            "encoding": enc,
            "seed":     seed,
        },
        "train": compute_metrics(y_train, model.predict(Ftr)),
        "val":   compute_metrics(y_val,   model.predict(Fvl)),
        "test":  compute_metrics(y_test,  model.predict(Fte)),
    }

    log.info(
        f"    R² test={metrics['test']['r2']:.4f} | "
        f"MAE={metrics['test']['mae']:.2f} | "
        f"RMSE={metrics['test']['rmse']:.2f}"
    )

    # 7. Guardar
    save_json(results_dir / f"metrics_{tag}.json", metrics)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("FASE 6 (P3) — XGBoost con primeras 4 features + ⟨Z⟩")
    log.info("=" * 60)

    p3  = cfg["pendiente3"]
    nq  = int(p3["nq"])
    n_f = int(p3["n_features"])
    assert nq == n_f, "nq debe ser igual a n_features en P3"

    layers    = p3["layers"]
    encodings = p3["encoding"]
    datasets  = p3["datasets"]
    seed      = int(p3.get("seed", 42))
    res_dir   = Path(p3["results_dir"])
    ensure_dirs(res_dir)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    data     = load_splits(proc_dir)
    xgb_p    = get_xgb_params(cfg)

    dataset_map = {
        "Golden":     (data["X_train"],   data["X_val"],   data["X_test"]),
        "Golden_PCA": (data["X_train_r"], data["X_val_r"], data["X_test_r"]),
    }

    all_metrics = []

    for ds_name in datasets:
        if ds_name not in dataset_map:
            log.warning(f"Dataset '{ds_name}' no reconocido, saltando.")
            continue

        X_tr, X_vl, X_te = dataset_map[ds_name]

        if X_tr.shape[1] < nq:
            log.warning(f"Dataset '{ds_name}' tiene {X_tr.shape[1]} columnas, necesita {nq}")
            continue

        for L in layers:
            for enc in encodings:
                m = run_one(
                    dataset_name=ds_name,
                    X_train=X_tr, X_val=X_vl, X_test=X_te,
                    y_train=data["y_train"],
                    y_val=data["y_val"],
                    y_test=data["y_test"],
                    nq=nq, L=L, enc=enc, seed=seed,
                    results_dir=res_dir,
                    xgb_params=xgb_p,
                )
                all_metrics.append(m)

    # Resumen global
    save_json(res_dir / "summary_p3.json", {"results": all_metrics})
    log.info(f"\nFASE 6 completada ✓ — {len(all_metrics)} configuraciones")
    log.info(f"Resultados en: {res_dir}")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)