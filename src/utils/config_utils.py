# src/utils/config_utils.py

from pathlib import Path #Investigar pathlib, hacer un ejercicio de ejemplo
import yaml #nvestigar yaml, hacer un ejercicio de ejemplo
import os # Investigar os, hacer un ejercicio de ejemplo
from itertools import product # Investigar itertools, hacer un ejercicio de ejemplo

# Una función que carga un archivo de configuración YAML. 
# El parámetro path tiene un valor predeterminado de "config.yaml", 
# lo que significa que si no se proporciona un argumento al llamar a la función,
# se intentará cargar un archivo llamado "config.yaml" en el directorio actual.
def load_config(path="config.yaml"): 
    """Carga el archivo de configuración YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Una función que toma un diccionario de configuración (cfg) y devuelve otro diccionario que contiene rutas de archivos o directorios.
def get_paths(cfg): 
    """
    Devuelve el diccionario de rutas desde el config,
    convirtiendo todos los valores a pathlib.Path.
    """
    raw = cfg.get("paths", {})
    return {key: Path(val) for key, val in raw.items()}

# Una función que genera todas las combinaciones posibles de tres parámetros: nq, L y encoding, 
# definidos en el diccionario de configuración (cfg) bajo la clave 'sweep'.
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

# Una función que actúa como un alias de get_combos para compatibilidad con versiones anteriores del código.
def get_sweep_combinations(cfg): 
    """Alias de get_combos para compatibilidad."""
    return get_combos(cfg)

# Una función que genera un nombre canónico para una combinación específica de parámetros nq, L y encoding.
def combo_name(nq, L, encoding): 
    """Nombre canónico de una combinación."""
    return f"nq{nq}_L{L}_{encoding}"

# Una función que crea todos los directorios definidos en el diccionario de configuración (cfg) bajo la clave 'paths' si no existen.
def ensure_dirs(cfg):
    """Crea todos los directorios definidos en cfg['paths'] si no existen."""
    paths = get_paths(cfg)
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)