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

import json # Hacer un ejercicio de ejemplo de json
import numpy  as np # Hacer un ejercicio de ejemplo de numpy, con los temas que aquí se usan: np.savez_compressed, np.load, np.isnan, np.isinf
import pandas as pd # Hacer un ejercicio de ejemplo de pandas, con los temas que aquí se usan: pd.DataFrame, df.to_csv, pd.read_csv
from pathlib import Path # Hacer un ejercicio de ejemplo de pathlib, con los temas que aquí se usan: Path, path.parent.mkdir, path.exists
from typing  import Any # Hacer un ejercicio de ejemplo de typing, con los temas que aquí se usan: Any, dict[str, Any], list[str], str | Path, np.ndarray

from src.utils.log_utils import get_logger # ¿Cómo es que se permite importar una función de otro archivo dentro del mismo paquete? Hacer un ejemplo de importación relativa y absoluta.

log = get_logger("io_utils") # ¿Por qué parametro "io_utils"?


# ══════════════════════════════════════════════════════════════════════════════
# NPZ
# ════════════════════════════════════

def save_npz(path: str | Path, **arrays: np.ndarray) -> None: # ¿por qué usar |?
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
    # Convertir a Path y crear directorio si no existe
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays) # ¿Por qué se usa .savez_compressed y no np.savez? ¿Cuál es la diferencia?
    sizes = {k: v.shape for k, v in arrays.items()} # ¿por qué se usa un dict comprehension? Hacer un ejemplo de dict comprehension.
    log.debug(f"  NPZ guardado: {path.name} | shapes={sizes}") # ¿Qué es .debug? Hacer un ejemplo de logging con debug, info, warning, error y critical.

# Una función que carga un archivo .npz y devuelve un diccionario de arrays, con validación de claves requeridas.
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

    data = dict(np.load(str(path), allow_pickle=False)) # Por qué allow_pickle=False? Hacer un ejemplo de np.load con allow_pickle=True y allow_pickle=False.

    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise KeyError(
                f"Claves faltantes en {path.name}: {missing}. "
                f"Claves disponibles: {list(data.keys())}"
            )

    log.debug(
        f"  NPZ cargado: {path.name} | " # De nuevo, .debug. Hacer un ejemplo de logging con debug, info, warning, error y critical.
        f"keys={list(data.keys())}" #¿Por qué se usa list(data.keys()) y no data.keys()? Hacer un ejemplo de dict.keys() y list(dict.keys()).
    )
    return data


# ══════════════════════════════════════════════════════════════════════════════
# JSON
# ══════════════════════════════════════════════════════════════════════════════

# Una función que guarda un diccionario en un archivo JSON con indentación y convierte tipos numpy a tipos Python nativos para serialización.
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
    data_clean = _numpy_to_python(data) # ¿por qué convertir tipos numpy a Python nativos? Hacer un ejemplo de conversión de tipos numpy a tipos Python nativos. ¿Qué diferencia hay entre np.int64 y int, np.float64 y float, np.ndarray y list?

    with open(path, "w", encoding="utf-8") as f: # ¿Qué es "w"? ¿Qué es "r"? ¿Qué es "a"?
        json.dump(data_clean, f, indent=2, ensure_ascii=False) # ¿Qué es indent=2? ¿Qué es ensure_ascii=False? Hacer un ejemplo de json.dump con indent y ensure_ascii.

    log.debug(f"  JSON guardado: {path.name}") # ¿Qué es path.name? Hacer un ejemplo de Path.name, Path.stem, Path.suffix, Path.parent, Path.parts, Path.is_file(), Path.is_dir(), Path.exists(), Path.mkdir(), Path.rmdir(), Path.unlink(), Path.rename(), Path.glob(), Path.iterdir(), Path.read_text(), Path.write_text(), Path.read_bytes(), Path.write_bytes().

# Una función que carga un archivo JSON y devuelve un diccionario, con validación de existencia del archivo.
def load_json(path: str | Path) -> dict[str, Any]:
    """
    Carga un archivo JSON.

    Raises
    ------
    FileNotFoundError si el archivo no existe
    """
    path = Path(path)
    if not path.exists(): # path.exists() devuelve True si el archivo o directorio existe, False si no. Hacer un ejemplo de Path.exists() con un archivo que existe y uno que no existe.
        raise FileNotFoundError(f"JSON no encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log.debug(f"  JSON cargado: {path.name}")
    return data


def _numpy_to_python(obj: Any) -> Any:
    """
    Convierte recursivamente tipos numpy a tipos Python nativos # ¿Qué son los Python nativos? Hacer un ejemplo de tipos Python nativos: int, float, str, bool, list, dict, tuple, set, None. Hacer un ejemplo de tipos numpy: np.int64, np.float64, np.ndarray, np.bool_, np.str_, np.object_.
    para serialización JSON.
    """
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()} # ¿Por qué se usa esta forma: v in obj.items()? Hacer un ejemplo de dict.items() y dict comprehension.
    if isinstance(obj, list):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist() # ¿Qué es .tolist()? Hacer un ejemplo de np.ndarray.tolist() y np.ndarray.flatten(). ¿Cuál es la diferencia entre np.ndarray.tolist() y np.ndarray.flatten().tolist()?
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None   # JSON no soporta NaN/Inf
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# CSV
# ══════════════════════════════════════════════════════════════════════════════

# Una función que guarda un DataFrame en un archivo CSV, creando el directorio si no existe y permitiendo argumentos adicionales para df.to_csv().
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
    df.to_csv(str(path), index=False, **kwargs) #¿Qué hace **kwargs aquí? Hacer un ejemplo de **kwargs en una función. 
    # ¿Qué hace index=False? Hacer un ejemplo de df.to_csv() con index=True y index=False.
    log.debug(f"  CSV guardado: {path.name} | shape={df.shape}")

# Una función que carga un archivo CSV y devuelve un DataFrame, con validación de existencia del archivo y de columnas requeridas.
def load_csv(
    path:             str | Path,
    required_columns: list[str] | None = None,
    **kwargs, # ¿Qué hace **kwargs aquí? Hacer un ejemplo de **kwargs en una función.
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
# Una función que crea los directorios indicados si no existen, con logging de depuración. ¿Qué es logging de depuración? Hacer un ejemplo de logging con debug, info, warning, error y critical.
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