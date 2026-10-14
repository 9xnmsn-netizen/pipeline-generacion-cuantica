"""
feature_utils.py
----------------
Funciones comunes para construcción y transformación de features.

Extrae las funciones repetidas en:
  - entrenar_lotes_slurm.py
  - generar_datasets_validacion.py
  - generar_datasets_test.py
  - phase4_predict.py
  - predecir_datos_reales_baseO.py
  - preparar_datos_lotes.py

Uso:
    from src.utils.feature_utils import (
        normalize_pi,
        apply_pca,
        load_pca_objects,
        prepare_X_config,
        CONFIGS_12,
        load_top30,
    )
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any

from src.utils.log_utils import get_logger

log = get_logger("feature_utils")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

# Las 12 configuraciones del pipeline original.
# Formato: (nombre, usa_top30, usa_pca, usa_cuanticas, usa_gps_quantum)
CONFIGS_12 = [
    ("Original",                        False, False, False, False),
    ("Original_GPS_Quantum",            False, False, False, True),
    ("Original_Cuanticas",              False, False, True,  False),
    ("Original_Cuanticas_GPS_Quantum",  False, False, True,  True),
    ("Top30",                           True,  False, False, False),
    ("Top30_GPS_Quantum",               True,  False, False, True),
    ("Top30_Cuanticas",                 True,  False, True,  False),
    ("Top30_Cuanticas_GPS_Quantum",     True,  False, True,  True),
    ("PCA",                             False, True,  False, False),
    ("PCA_GPS_Quantum",                 False, True,  False, True),
    ("PCA_Cuanticas",                   False, True,  True,  False),
    ("PCA_Cuanticas_GPS_Quantum",       False, True,  True,  True),
]

CONFIG_NAMES = [c[0] for c in CONFIGS_12]

# Columnas GPS del detector de superficie
GPS_COLS = [
    "SD_fSdGPSSecond",
    "SD_fSdGPSNanoSecond",
    "SD_fYYMMDD",
    "SD_fHHMMSS",
]


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def normalize_pi(
    X:      np.ndarray,
    X_min:  np.ndarray | None = None,
    X_max:  np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza X a [-π, π] columna a columna.

    Si X_min y X_max son None, se ajustan sobre X (entrenamiento).
    Si se proporcionan, se usan para transformar (validación/test).

    Parameters
    ----------
    X     : (N, D) array
    X_min : (D,) mínimos por columna (opcional)
    X_max : (D,) máximos por columna (opcional)

    Returns
    -------
    X_norm : (N, D) en [-π, π]
    X_min  : mínimos usados
    X_max  : máximos usados
    """
    if X_min is None:
        X_min = X.min(axis=0)
    if X_max is None:
        X_max = X.max(axis=0)

    rng = X_max - X_min
    rng[rng == 0] = 1.0   # evitar división por cero en columnas constantes

    X_norm = (2.0 * np.pi * (X - X_min) / rng - np.pi).astype(np.float32)
    return X_norm, X_min, X_max


# ══════════════════════════════════════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════════════════════════════════════

def load_pca_objects(pca_dir: Path) -> dict:
    """
    Carga los objetos PCA guardados en Fase 1:
      - pca_scaler.pkl   (StandardScaler)
      - pca_model.pkl    (PCA)

    Parameters
    ----------
    pca_dir : Path — data/processed/

    Returns
    -------
    dict con claves 'scaler' y 'pca'
    """
    import pickle

    scaler_path = Path(pca_dir) / "pca_scaler.pkl"
    pca_path    = Path(pca_dir) / "pca_model.pkl"

    if not scaler_path.exists() or not pca_path.exists():
        raise FileNotFoundError(
            f"Objetos PCA no encontrados en {pca_dir}.\n"
            f"  Buscados: {scaler_path.name}, {pca_path.name}\n"
            f"  Ejecuta la Fase 1 primero."
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    log.debug(f"PCA cargado: {pca.n_components_} componentes")
    return {"scaler": scaler, "pca": pca}


def apply_pca(
    X:      np.ndarray,
    scaler,
    pca,
) -> np.ndarray:
    """
    Aplica StandardScaler + PCA a X.

    Parameters
    ----------
    X      : (N, D) array
    scaler : StandardScaler fitted
    pca    : PCA fitted

    Returns
    -------
    X_pca : (N, n_components) float32
    """
    X_scaled = scaler.transform(X)
    X_pca    = pca.transform(X_scaled)
    return X_pca.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TOP30
# ══════════════════════════════════════════════════════════════════════════════

def load_top30(top30_path: Path) -> list[str]:
    """
    Carga la lista de las 30 columnas top (seleccionadas por PCA).

    Parameters
    ----------
    top30_path : Path — fase_pca/data/top30_columnas.txt

    Returns
    -------
    list[str] con 30 nombres de columnas
    """
    top30_path = Path(top30_path)
    if not top30_path.exists():
        raise FileNotFoundError(f"top30_columnas.txt no encontrado: {top30_path}")

    with open(top30_path) as f:
        cols = [line.strip() for line in f if line.strip()]

    log.debug(f"Top30 cargado: {len(cols)} columnas")
    return cols


# ══════════════════════════════════════════════════════════════════════════════
# PREPARACIÓN POR CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def prepare_X_config(
    X:              np.ndarray,
    config_name:    str,
    top30_cols:     list[str] | None = None,
    pca_objects:    dict | None = None,
    X_quantum:      np.ndarray | None = None,
    X_gps_quantum:  np.ndarray | None = None,
) -> np.ndarray:
    """
    Prepara la matriz X según la configuración (12 opciones).

    Mapeo de configuraciones:
      - Original                        → X tal cual
      - Original_GPS_Quantum            → X + GPS_quantum
      - Original_Cuanticas              → X + X_quantum
      - Original_Cuanticas_GPS_Quantum  → X + X_quantum + GPS_quantum
      - Top30                           → X[top30]
      - Top30_GPS_Quantum               → X[top30] + GPS_quantum
      - Top30_Cuanticas                 → X[top30] + X_quantum
      - Top30_Cuanticas_GPS_Quantum     → X[top30] + X_quantum + GPS_quantum
      - PCA                             → PCA(X)
      - PCA_GPS_Quantum                 → PCA(X) + GPS_quantum
      - PCA_Cuanticas                   → PCA(X) + X_quantum
      - PCA_Cuanticas_GPS_Quantum       → PCA(X) + X_quantum + GPS_quantum

    Parameters
    ----------
    X              : (N, D) matriz base (columnas SD + logE)
    config_name    : nombre de la configuración
    top30_cols     : lista de columnas Top30 (si aplica)
    pca_objects    : dict con 'scaler' y 'pca' (si aplica)
    X_quantum      : (N, nq_total) features cuánticas (si aplica)
    X_gps_quantum  : (N, 4) features GPS cuánticas (si aplica)

    Returns
    -------
    X_prepared : np.ndarray (N, D')
    """
    # ── Validar configuración ──────────────────────────────────────────────
    configs_dict = {c[0]: c for c in CONFIGS_12}
    if config_name not in configs_dict:
        raise ValueError(
            f"Configuración '{config_name}' no reconocida.\n"
            f"Disponibles: {list(configs_dict.keys())}"
        )

    _, uses_top30, uses_pca, uses_q, uses_gps = configs_dict[config_name]

    # ── Paso 1: transformación base ────────────────────────────────────────
    X_out = X.copy()
    if hasattr(X_out, "values"):
        X_out = X_out.values
    X_out = np.asarray(X_out, dtype=np.float32)

    # ── Paso 2: Top30 ──────────────────────────────────────────────────────
    if uses_top30:
        if top30_cols is None:
            raise ValueError(f"Config '{config_name}' requiere top30_cols")
        # Necesitamos los índices de top30 en las columnas originales.
        # Si X tiene nombres, se usa búsqueda; si no, se asume orden.
        if hasattr(X, "columns"):
            idx = [X.columns.get_loc(c) for c in top30_cols if c in X.columns]
        else:
            idx = list(range(min(30, X.shape[1])))   # fallback
        X_out = X_out[:, idx]

    # ── Paso 3: PCA ────────────────────────────────────────────────────────
    if uses_pca:
        if pca_objects is None:
            raise ValueError(f"Config '{config_name}' requiere pca_objects")
        X_out = apply_pca(X_out, pca_objects["scaler"], pca_objects["pca"])

    # ── Paso 4: Cuánticas ──────────────────────────────────────────────────
    if uses_q:
        if X_quantum is None:
            raise ValueError(f"Config '{config_name}' requiere X_quantum")
        X_out = np.hstack([X_out, X_quantum])

    # ── Paso 5: GPS Quantum ────────────────────────────────────────────────
    if uses_gps:
        if X_gps_quantum is None:
            raise ValueError(f"Config '{config_name}' requiere X_gps_quantum")
        X_out = np.hstack([X_out, X_gps_quantum])

    return X_out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE FEATURES POR LOTE
# ══════════════════════════════════════════════════════════════════════════════

def load_lote_features(
    lote:         str,
    features_dir: Path,
    kind:         str = "quantum",
) -> np.ndarray | None:
    """
    Carga features cuánticas o GPS cuánticas de un lote.

    Parameters
    ----------
    lote         : 'bajo', 'medio', 'alto', 'todo'
    features_dir : Path — directorio que contiene los .npy
    kind         : 'quantum' → X_quantum_{lote}.npy
                   'gps'     → GPS_quantum_{lote}.npy
                   'gps_val' → GPS_quantum_{lote}_val.npy
                   'quantum_val' → X_quantum_{lote}_val.npy

    Returns
    -------
    np.ndarray o None si no existe
    """
    if kind == "quantum":
        fname = f"X_quantum_{lote}.npy"
    elif kind == "gps":
        fname = f"GPS_quantum_{lote}.npy"
    elif kind == "gps_val":
        fname = f"GPS_quantum_{lote}_val.npy"
    elif kind == "quantum_val":
        fname = f"X_quantum_{lote}_val.npy"
    else:
        raise ValueError(f"kind='{kind}' no reconocido")

    fpath = Path(features_dir) / fname
    if not fpath.exists():
        log.warning(f"No encontrado: {fpath}")
        return None

    return np.load(fpath)


# ══════════════════════════════════════════════════════════════════════════════
# FILTROS POR LOTE
# ══════════════════════════════════════════════════════════════════════════════

def get_lote_mask(
    logE:      np.ndarray,
    lote:      str,
    cfg:       dict,
) -> np.ndarray:
    """
    Devuelve máscara booleana para filtrar por lote de energía.

    Parameters
    ----------
    logE : array de log10(E/eV)
    lote : 'bajo', 'medio', 'alto', 'todo'
    cfg  : configuración global (usa cfg['energy_bins'])

    Returns
    -------
    mask : (N,) bool
    """
    bins = cfg.get("energy_bins", {})
    if lote not in bins:
        raise ValueError(f"Lote '{lote}' no reconocido. Opciones: {list(bins.keys())}")

    info = bins[lote]
    lim_inf = info["lim_inf"]
    lim_sup = info["lim_sup"]

    if lote == "todo":
        return np.ones(len(logE), dtype=bool)

    return (logE >= lim_inf) & (logE < lim_sup)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def get_config_flags(config_name: str) -> dict:
    """
    Devuelve las banderas (top30, pca, cuanticas, gps) de una configuración.

    Returns
    -------
    dict con claves 'top30', 'pca', 'cuanticas', 'gps'
    """
    configs_dict = {c[0]: c for c in CONFIGS_12}
    if config_name not in configs_dict:
        raise ValueError(f"Config '{config_name}' no reconocida")

    _, top30, pca, cuanticas, gps = configs_dict[config_name]
    return {
        "top30":     top30,
        "pca":       pca,
        "cuanticas": cuanticas,
        "gps":       gps,
    }


def list_configs() -> list[str]:
    """Devuelve la lista de las 12 configuraciones."""
    return list(CONFIG_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "CONFIGS_12",
    "CONFIG_NAMES",
    "GPS_COLS",
    "normalize_pi",
    "load_pca_objects",
    "apply_pca",
    "load_top30",
    "prepare_X_config",
    "load_lote_features",
    "get_lote_mask",
    "get_config_flags",
    "list_configs",
]