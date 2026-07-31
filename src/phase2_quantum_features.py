"""
phase2_quantum_features.py
--------------------------
Fase 2: Generación de características cuánticas.

Para cada combinación (nq, L, encoding) del barrido:
  1. Construye el circuito con quantum_utils.build_circuit()
  2. Inicializa θ con quantum_utils.init_theta()
  3. Evalúa el circuito en train / val / test
  4. Guarda qfeatures_{nq}q_{L}l_{enc}.npz en data/processed/

Contrato de salida (por archivo .npz):
  X_train_q, X_val_q, X_test_q : float32 shape (N, nq)
  y_train, y_val, y_test        : copiados sin cambio
  logE_train, logE_val, logE_test: copiados sin cambio
  theta                          : float32 shape (L, nq, 2)
"""

import sys
import json
import itertools
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils  import load_config, get_paths, get_combos
from src.utils.io_utils      import load_npz, save_npz, ensure_dirs
from src.utils.log_utils     import get_logger
from src.utils.quantum_utils import build_circuit, get_qfeatures, init_theta

log = get_logger("phase2")

# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def normalize_to_pi(X: np.ndarray) -> np.ndarray:
    """
    Normaliza cada columna de X al rango [-π, π].
    Necesario antes de pasarlo al circuito cuántico.

    Fórmula: x_norm = 2π · (x - x_min) / (x_max - x_min) - π

    Parameters
    ----------
    X : array shape (n_samples, n_features)

    Returns
    -------
    X_norm : array shape (n_samples, n_features) en [-π, π]
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    rng   = X_max - X_min
    # Evitar división por cero en columnas constantes
    rng[rng == 0] = 1.0
    X_norm = 2.0 * np.pi * (X - X_min) / rng - np.pi
    return X_norm.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# NOMBRE DE ARCHIVO
# ══════════════════════════════════════════════════════════════════════════════

def _make_filename(nq: int, L: int, enc: str) -> str:
    return f"qfeatures_{nq}q_{L}l_{enc}.npz"


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE POR COMBINACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def process_combo(
    nq       : int,
    L        : int,
    enc      : str,
    X_train  : np.ndarray,
    X_val    : np.ndarray,
    X_test   : np.ndarray,
    data     : dict,
    proc_dir : Path,
    seed     : int,
    batch_size: int,
    overwrite : bool,
) -> str:
    """
    Genera y guarda features cuánticas para una combinación (nq, L, enc).

    Returns
    -------
    str : ruta del archivo guardado
    """
    fname = _make_filename(nq, L, enc)
    fpath = proc_dir / fname

    # ── Skip si ya existe ─────────────────────────────────────────────────
    if fpath.exists() and not overwrite:
        log.info(f"  Ya existe y overwrite=False → saltando: {fname}")
        return str(fpath)

    log.info(f"  nq={nq} | L={L} | enc='{enc}'")

    # ── Normalizar entrada al rango [-π, π] ───────────────────────────────
    # IMPORTANTE: la normalización se ajusta sobre train y se aplica a todos
    X_min  = X_train.min(axis=0)
    X_max  = X_train.max(axis=0)
    rng    = X_max - X_min
    rng[rng == 0] = 1.0

    def _norm(X):
        return (2.0 * np.pi * (X - X_min) / rng - np.pi).astype(np.float32)

    X_train_n = _norm(X_train)
    X_val_n   = _norm(X_val)
    X_test_n  = _norm(X_test)

    # ── Inicializar θ ─────────────────────────────────────────────────────
    theta   = init_theta(L, nq, seed=seed)
    circuit = build_circuit(nq, L, enc)

    log.info(f"  θ shape: {theta.shape} | "
             f"rango: [{theta.min():.3f}, {theta.max():.3f}]")

    # ── Evaluar circuito ──────────────────────────────────────────────────
    log.info("  → Evaluando en train...")
    X_train_q = get_qfeatures(X_train_n, circuit, theta,
                               batch_size=batch_size, verbose=True)

    log.info("  → Evaluando en val...")
    X_val_q   = get_qfeatures(X_val_n,   circuit, theta,
                               batch_size=batch_size, verbose=True)

    log.info("  → Evaluando en test...")
    X_test_q  = get_qfeatures(X_test_n,  circuit, theta,
                               batch_size=batch_size, verbose=True)

    # ── Guardar .npz ──────────────────────────────────────────────────────
    cfg_combo = {"nq": nq, "L": L, "encoding": enc, "seed": seed}
    save_npz(
        fpath,
        X_train_q   = X_train_q,
        X_val_q     = X_val_q,
        X_test_q    = X_test_q,
        y_train     = data["y_train"],
        y_val       = data["y_val"],
        y_test      = data["y_test"],
        logE_train  = data["logE_train"],
        logE_val    = data["logE_val"],
        logE_test   = data["logE_test"],
        theta       = theta,
        # Guardar parámetros de normalización para usarlos en phase4
        norm_X_min  = X_min.astype(np.float32),
        norm_X_max  = X_max.astype(np.float32),
    )
    log.info(f"  Guardado: {fpath}")
    return str(fpath)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("FASE 2 — Generación de características cuánticas")
    log.info("=" * 60)

    paths    = get_paths(cfg)
    p2       = cfg["phase2"]
    proc_dir = paths["processed_dir"]
    ensure_dirs(proc_dir)

    # ── Cargar datos reducidos ────────────────────────────────────────────
    reduced = load_npz(
        proc_dir / "reduced.npz",
        required_keys=["X_train_r", "X_val_r", "X_test_r"],
        )
    splits = load_npz(
        proc_dir / "splits.npz",
        required_keys=["y_train", "y_val", "y_test",
                   "logE_train", "logE_val", "logE_test"],
                   )
    
    X_train = reduced["X_train_r"]
    X_val   = reduced["X_val_r"]
    X_test  = reduced["X_test_r"]

    # Pasar splits al process_combo como data
    data = splits   # ya tiene y_train, y_val, y_test, logE_*

    log.info(
        f"Datos reducidos → "
        f"train: {X_train.shape} | "
        f"val: {X_val.shape} | "
        f"test: {X_test.shape}"
    )

    # ── Leer parámetros del barrido ───────────────────────────────────────
    # Tomar de sweep (global) — phase2 no redefine el barrido
    combos     = get_combos(cfg)
    seed       = int(p2.get("random_seed", 42))
    batch_size = int(p2.get("batch_size", 64))
    overwrite  = bool(p2.get("overwrite", True))

    log.info(f"Total de combinaciones: {len(combos)}")

    # ── Barrido ───────────────────────────────────────────────────────────
    generated = []
    for idx, combo in enumerate(combos, start=1):

        nq, L, enc = combo

        log.info(f"\n[{idx}/{len(combos)}] nq={nq} | L={L} | enc='{enc}'")

        try:
            out = process_combo(
                nq        = nq,
                L         = L,
                enc       = enc,
                X_train   = X_train,
                X_val     = X_val,
                X_test    = X_test,
                data      = data,
                proc_dir  = proc_dir,
                seed      = seed,
                batch_size= batch_size,
                overwrite = overwrite,
            )
            generated.append(out)
        except Exception as e:
            log.error(f"  ERROR en nq={nq}, L={L}, enc={enc}: {e}")
            import traceback
            log.debug(traceback.format_exc())

    log.info(f"\nFASE 2 completada ✓ — {len(generated)} archivos generados")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)