"""
io_utils.py
-----------
Lectura y escritura de archivos del pipeline.

Funciones:
    save_npz      → guarda arrays en .npz
    load_npz      → carga .npz con validación de claves
    save_json     → guarda dict en .json
    load_json     → carga .json
    save_csv      → guarda DataFrame en .csv
    load_csv      → carga .csv con validación de columnas
    ensure_dirs   → crea directorios necesarios
"""

import json
import numpy  as np
import pandas as pd
from pathlib import Path
from typing  import Any

from src.utils.log_utils import get_logger

log = get_logger("io_utils")


# ══════════════════════════════════════════════════════════════════════════════
# NPZ
# ══════════════════════════════════════════════════════════════════════════════

def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """
    Guarda arrays numpy en formato .npz comprimido.

    Parameters
    ----------
    path   : ruta de salida (se crea el directorio si no existe)
    arrays : keyword arguments → nombre=array

    Ejemplo
    -------
    save_npz("data/processed/train.npz", X=X_train, y=y_train)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)
    sizes = {k: v.shape for k, v in arrays.items()}
    log.debug(f"  NPZ guardado: {path.name} | shapes={sizes}")


def load_npz(
    path:           str | Path,
    required_keys:  list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Carga un archivo .npz y devuelve un dict de arrays.

    Parameters
    ----------
    path          : ruta al archivo .npz
    required_keys : lista de claves que deben existir (opcional)

    Returns
    -------
    dict[str, np.ndarray]

    Raises
    ------
    FileNotFoundError si el archivo no existe
    KeyError          si faltan claves requeridas
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ no encontrado: {path}")

    data = dict(np.load(str(path), allow_pickle=False))

    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise KeyError(
                f"Claves faltantes en {path.name}: {missing}. "
                f"Claves disponibles: {list(data.keys())}"
            )

    log.debug(
        f"  NPZ cargado: {path.name} | "
        f"keys={list(data.keys())}"
    )
    return data


# ══════════════════════════════════════════════════════════════════════════════
# JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """
    Guarda un diccionario en formato JSON con indentación.

    Parameters
    ----------
    path : ruta de salida
    data : diccionario a guardar
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convertir tipos numpy a Python nativos
    data_clean = _numpy_to_python(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_clean, f, indent=2, ensure_ascii=False)

    log.debug(f"  JSON guardado: {path.name}")


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Carga un archivo JSON.

    Raises
    ------
    FileNotFoundError si el archivo no existe
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON no encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log.debug(f"  JSON cargado: {path.name}")
    return data


def _numpy_to_python(obj: Any) -> Any:
    """
    Convierte recursivamente tipos numpy a tipos Python nativos
    para serialización JSON.
    """
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None   # JSON no soporta NaN/Inf
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(
    path: str | Path,
    df:   pd.DataFrame,
    **kwargs,
) -> None:
    """
    Guarda un DataFrame en CSV.

    Parameters
    ----------
    path   : ruta de salida
    df     : DataFrame a guardar
    kwargs : argumentos adicionales para df.to_csv()
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False, **kwargs)
    log.debug(f"  CSV guardado: {path.name} | shape={df.shape}")


def load_csv(
    path:             str | Path,
    required_columns: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Carga un CSV con validación de columnas requeridas.

    Parameters
    ----------
    path             : ruta al CSV
    required_columns : columnas que deben existir (opcional)
    kwargs           : argumentos adicionales para pd.read_csv()

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError si el archivo no existe
    ValueError        si faltan columnas requeridas
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {path}")

    df = pd.read_csv(str(path), **kwargs)

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columnas faltantes en {path.name}: {missing}. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    log.debug(
        f"  CSV cargado: {path.name} | "
        f"shape={df.shape}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTORIOS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs(*paths: str | Path) -> None:
    """
    Crea los directorios indicados si no existen.

    Ejemplo
    -------
    ensure_dirs("data/processed", "data/results", "logs")
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        log.debug(f"  Directorio asegurado: {p}")