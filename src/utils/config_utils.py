# src/utils/config_utils.py

from pathlib import Path
import yaml
import os
from itertools import product


def load_config(path="config.yaml"):
    """Carga el archivo de configuración YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_paths(cfg):
    """
    Devuelve el diccionario de rutas desde el config,
    convirtiendo todos los valores a pathlib.Path.
    """
    raw = cfg.get("paths", {})
    return {key: Path(val) for key, val in raw.items()}


def get_combos(cfg):
    """
    Genera todas las combinaciones (nq, L, encoding)
    definidas en cfg['sweep'].
    """
    sweep = cfg.get("sweep", {})
    nq_list  = sweep.get("nq",       [2])
    L_list   = sweep.get("L",        [1])
    enc_list = sweep.get("encoding", ["angle"])

    return list(product(nq_list, L_list, enc_list))


def get_sweep_combinations(cfg):
    """Alias de get_combos para compatibilidad."""
    return get_combos(cfg)


def combo_name(nq, L, encoding):
    """Nombre canónico de una combinación."""
    return f"nq{nq}_L{L}_{encoding}"


def ensure_dirs(cfg):
    """Crea todos los directorios definidos en cfg['paths'] si no existen."""
    paths = get_paths(cfg)
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)# src/utils/config_utils.py

from pathlib import Path
import yaml
import os
from itertools import product


def load_config(path="config.yaml"):
    """Carga el archivo de configuración YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_paths(cfg):
    """
    Devuelve el diccionario de rutas desde el config,
    convirtiendo todos los valores a pathlib.Path.
    """
    raw = cfg.get("paths", {})
    return {key: Path(val) for key, val in raw.items()}


def get_combos(cfg):
    """
    Genera todas las combinaciones (nq, L, encoding)
    definidas en cfg['sweep'].
    """
    sweep = cfg.get("sweep", {})
    nq_list  = sweep.get("nq",       [2])
    L_list   = sweep.get("L",        [1])
    enc_list = sweep.get("encoding", ["angle"])

    return list(product(nq_list, L_list, enc_list))


def get_sweep_combinations(cfg):
    """Alias de get_combos para compatibilidad."""
    return get_combos(cfg)


def combo_name(nq, L, encoding):
    """Nombre canónico de una combinación."""
    return f"nq{nq}_L{L}_{encoding}"


def ensure_dirs(cfg):
    """Crea todos los directorios definidos en cfg['paths'] si no existen."""
    paths = get_paths(cfg)
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)