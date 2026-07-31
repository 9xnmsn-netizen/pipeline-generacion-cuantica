"""
phase1_reduce.py
----------------
Fase 1: Reducción de dimensionalidad (PCA / UMAP / ninguna).

ENTRADA : data/processed/splits.npz  (X_train, X_val, X_test, y_*)
SALIDA  : data/processed/reduced.npz (X_train_r, X_val_r, X_test_r, n_components)
          data/processed/reducer.pkl  (objeto fitted, para fase 4)
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.io_utils     import load_npz, save_npz, ensure_dirs
from src.utils.log_utils    import get_logger

log = get_logger("phase1")


# ── Reducción ──────────────────────────────────────────────────────────────

# ✅ función fit_reducer corregida
def fit_reducer(X_train: np.ndarray, cfg: dict):
    """
    Ajusta el reductor SOLO sobre X_train.
    Para PCA: aplica StandardScaler primero para evitar overflow.
    Devuelve (reducer, scaler) donde scaler puede ser None.
    """
    method = cfg["phase1"].get("method", "pca").lower()
    nc = cfg["phase1"].get("n_components", 4)
    seed = cfg["phase1"].get("random_seed", 42)

    scaler = None  # solo se usa con PCA

    if method == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # ── PASO 1: Escalar primero (CRÍTICO para evitar overflow) ──────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        log.info(
            f"StandardScaler aplicado: "
            f"media≈{X_scaled.mean():.3f}, std≈{X_scaled.std():.3f}"
        )

        # ── PASO 2: PCA sobre datos escalados ───────────────────────────────
        reducer = PCA(n_components=nc, random_state=seed)
        log.info(
            f"Ajustando PCA con n_components={nc} "
            f"sobre X_train escalado {X_scaled.shape}"
        )
        reducer.fit(X_scaled)

        var = reducer.explained_variance_ratio_
        log.info(f"Varianza explicada por componente: {np.round(var, 4)}")
        log.info(f"Varianza acumulada total: {var.sum():.4f}")

        return reducer, scaler

    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "umap-learn no está instalado. Ejecuta: pip install umap-learn"
            )
        reducer = UMAP(
            n_components  = nc,
            random_state  = seed,
            n_neighbors   = cfg["phase1"].get("umap_n_neighbors", 15),
            min_dist      = cfg["phase1"].get("umap_min_dist", 0.1),
        )
        log.info(
            f"Ajustando UMAP con n_components={nc} "
            f"sobre X_train {X_train.shape}"
        )
        reducer.fit(X_train)
        return reducer, None

    elif method == "none":
        log.info("Método 'none': sin reducción.")
        return None, None

    else:
        raise ValueError(
            f"Método desconocido: '{method}'. Usa 'pca', 'umap' o 'none'."
        )


# ✅ REEMPLAZAR POR
def apply_reducer(
    reducer,
    scaler,
    X: np.ndarray,
    method: str,
) -> np.ndarray:
    """
    Aplica el reductor.
    Para PCA: primero escala con scaler, luego transforma con PCA.
    """
    if method == "none" or reducer is None:
        return X
    if method == "pca" and scaler is not None:
        X = scaler.transform(X)
    return reducer.transform(X).astype(np.float32)


# ── Pipeline ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("FASE 1 — Reducción de dimensionalidad")
    log.info("=" * 60)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    ensure_dirs(proc_dir)

    method = cfg["phase1"].get("method", "pca").lower()

    # ── Cargar splits ──────────────────────────────────────────────────
    splits_path = proc_dir / "splits.npz"
    if not splits_path.exists():
        raise FileNotFoundError(f"No existe {splits_path}. Ejecuta fase 0 primero.")

    data    = load_npz(splits_path)
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]

    log.info(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ── Fit solo en train ──────────────────────────────────────────────
    # ✅ REEMPLAZAR POR
    reducer, scaler = fit_reducer(X_train, cfg)
    X_train_r = apply_reducer(reducer, scaler, X_train, method)
    X_val_r   = apply_reducer(reducer, scaler, X_val,   method)
    X_test_r  = apply_reducer(reducer, scaler, X_test,  method)

    log.info(f"Shapes reducidos: train={X_train_r.shape}, val={X_val_r.shape}, test={X_test_r.shape}")

    # ── Guardar ────────────────────────────────────────────────────────
    nc = X_train_r.shape[1]
    save_npz(
        proc_dir / "reduced.npz",
        X_train_r = X_train_r,
        X_val_r   = X_val_r,
        X_test_r  = X_test_r,
        n_components = np.array(nc),
        method = np.array(method)
    )

    # ✅ REEMPLAZAR POR
    if reducer is not None:
        if method == "pca":
            # ── Guardar scaler (NUEVO — necesario para phase4) ───────────────
            with open(proc_dir / "pca_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open(proc_dir / "pca_model.pkl", "wb") as f:
                pickle.dump(reducer, f)
            with open(proc_dir / "reducer.pkl", "wb") as f:
                pickle.dump(reducer, f)
            log.info(
            "Guardados: pca_scaler.pkl + pca_model.pkl + reducer.pkl"
            )
        elif method == "umap":
            with open(proc_dir / "umap_model.pkl", "wb") as f:
                pickle.dump(reducer, f)
            with open(proc_dir / "reducer.pkl", "wb") as f:
                pickle.dump(reducer, f)
            log.info("Guardados: umap_model.pkl + reducer.pkl")
        elif method == "umap":
            with open(proc_dir / "umap_model.pkl", "wb") as f:
                pickle.dump(reducer, f)
            with open(proc_dir / "reducer.pkl", "wb") as f:
                pickle.dump(reducer, f)
            log.info(f"Reductor UMAP guardado: umap_model.pkl + reducer.pkl")
            
    log.info(f"FASE 1 completada ✓  →  {nc} componentes ({method})")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)