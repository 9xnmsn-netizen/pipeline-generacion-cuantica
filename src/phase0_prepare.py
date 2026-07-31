"""
phase0_prepare.py  (versión corregida)
--------------------------------------
Cambios respecto a la versión anterior:
  ✅ Lee claves desde cfg["phase0"] (compatible con config.yaml corregido)
  ✅ Guarda feature_cols.json (necesario para phase4)
  ✅ Guarda fd_real.npz con logE y xmax del conjunto completo (para phase5)
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.io_utils     import save_npz, save_json, ensure_dirs
from src.utils.log_utils    import get_logger

log = logging.getLogger("phase0")

def _read_csv_auto(filepath: Path) -> pd.DataFrame:
    """
    Lee un CSV detectando automáticamente si el separador es ',' o '\\t'.
    """
    # Leer primeras líneas para detectar separador
    with open(filepath, "r") as f:
        first_line = f.readline()

    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    log.info(f"  Leyendo {filepath.name} con sep={repr(sep)}")
    return pd.read_csv(filepath, sep=sep, low_memory=False)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y MERGE
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data(cfg: dict) -> pd.DataFrame:
    p0       = cfg["phase0"]
    paths    = get_paths(cfg)
    merge_on = p0.get("merge_on", "index")

    fd_path = Path(p0["fd_file"])
    sd_path = Path(p0["sd_file"])

    if not fd_path.exists():
        raise FileNotFoundError(f"No se encontró FD: {fd_path}")
    if not sd_path.exists():
        raise FileNotFoundError(f"No se encontró SD: {sd_path}")

    df_fd = pd.read_csv(fd_path, sep="\t")
    df_sd = pd.read_csv(sd_path, sep="\t")
    log.info(f"FD: {df_fd.shape} | SD: {df_sd.shape}")

    if merge_on == "index":
        if len(df_fd) != len(df_sd):
            n = min(len(df_fd), len(df_sd))
            log.warning(f"Filas distintas → usando las primeras {n} de cada uno")
            df_fd = df_fd.iloc[:n].reset_index(drop=True)
            df_sd = df_sd.iloc[:n].reset_index(drop=True)

        # Eliminar columnas duplicadas (conservar versión FD)
        dup = set(df_fd.columns) & set(df_sd.columns)
        if dup:
            log.warning(f"Columnas duplicadas eliminadas de SD: {dup}")
            df_sd = df_sd.drop(columns=list(dup))

        df = pd.concat([df_fd, df_sd], axis=1)
    else:
        df = pd.merge(df_fd, df_sd, on=merge_on, how="inner")

    if df.empty:
        raise ValueError("El DataFrame fusionado está vacío.")

    log.info(f"DataFrame fusionado: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FILTRO DE ENERGÍA
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_energy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p0         = cfg["phase0"]
    energy_col = p0["energy_col"]
    E_min      = 10 **float(p0["E_min"]) # convierte 18.5 → 10^18.5 eV
    E_max      = 10 **float(p0["E_max"]) # convierte 20.0 → 10^20.0 eV

    if energy_col not in df.columns:
        raise KeyError(
            f"Columna de energía '{energy_col}' no encontrada. "
            f"Disponibles: {list(df.columns)}"
        )

    mask = (df[energy_col] >= E_min) & (df[energy_col] <= E_max)
    df_f = df[mask].reset_index(drop=True)
    log.info(
        f"Filtro energía [{E_min}, {E_max}] en '{energy_col}': "
        f"{mask.sum()} / {len(df)} filas"
    )
    if df_f.empty:
        raise ValueError(
            f"Filtro de energía dejó 0 filas. "
            f"Rango en datos: [{df[energy_col].min():.3f}, "
            f"{df[energy_col].max():.3f}]"
        )
    return df_f


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES Y TARGET
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_target(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    p0         = cfg["phase0"]
    target_col = p0["target_col"]
    energy_col = p0["energy_col"]
    correction = p0.get("target_correction", None)
    exclude    = set(p0.get("exclude_cols", []))

    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' no encontrado en el DataFrame.")

    # Selección de features
    feature_cols = p0.get("feature_cols", [])
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"feature_cols no encontradas: {missing}")
        log.info(f"Features explícitas ({len(feature_cols)}): {feature_cols}")
    else:
        always_exclude = {target_col, energy_col} | exclude
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in always_exclude
        ]
        log.info(f"Features auto-detectadas ({len(feature_cols)}): {feature_cols}")

    if not feature_cols:
        raise ValueError("No se encontraron columnas de features.")

    X    = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)
    logE  = df[energy_col].values.astype(np.float32)

    # Corrección al target
    y = y_raw + float(correction) if correction is not None else y_raw.copy()
    if correction is not None:
        log.info(f"Corrección al target: +{correction}")

    # Imputar NaN en features
    nan_X = np.isnan(X).sum()
    if nan_X > 0:
        log.warning(f"NaN en features: {nan_X} → imputando con mediana")
        medians = np.nanmedian(X, axis=0)
        idx     = np.where(np.isnan(X))
        X[idx]  = np.take(medians, idx[1])

    # Eliminar filas con NaN en target o energía
    mask = ~(np.isnan(y) | np.isnan(logE))
    if (~mask).sum() > 0:
        log.warning(f"Eliminando {(~mask).sum()} filas con NaN en y o logE")
    X, y, logE = X[mask], y[mask], logE[mask]

    log.info(
        f"X: {X.shape} | "
        f"y: [{y.min():.1f}, {y.max():.1f}] | "
        f"logE: [{logE.min():.2f}, {logE.max():.2f}]"
    )
    return X, y, logE, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_data(
    X    : np.ndarray,
    y    : np.ndarray,
    logE : np.ndarray,
    cfg  : dict,
) -> dict[str, np.ndarray]:
    p0      = cfg["phase0"]
    splits  = p0["splits"]
    seed    = int(p0.get("random_seed", 42))
    f_train = float(splits["train"])
    f_val   = float(splits["val"])
    f_test  = float(splits["test"])

    f_val_rel = f_val / (f_train + f_val)

    X_tv, X_test, y_tv, y_test, logE_tv, logE_test = train_test_split(
        X, y, logE, test_size=f_test, random_state=seed
    )
    X_train, X_val, y_train, y_val, logE_train, logE_val = train_test_split(
        X_tv, y_tv, logE_tv, test_size=f_val_rel, random_state=seed
    )

    log.info(
        f"Split → train: {len(X_train)} | "
        f"val: {len(X_val)} | "
        f"test: {len(X_test)}"
    )
    return {
        "X_train"    : X_train,   "X_val"    : X_val,   "X_test"    : X_test,
        "y_train"    : y_train,   "y_val"    : y_val,   "y_test"    : y_test,
        "logE_train" : logE_train,"logE_val" : logE_val,"logE_test" : logE_test,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("FASE 0 — Preparación del dataset")
    log.info("=" * 60)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    ensure_dirs(proc_dir, paths["logs_dir"])

    # Pasos 1-4
    df                         = load_raw_data(cfg)
    df                         = filter_by_energy(df, cfg)
    X, y, logE, feature_names  = extract_features_target(df, cfg)
    splits                     = split_data(X, y, logE, cfg)

    # ── Guardar splits.npz ────────────────────────────────────────────────
    save_npz(proc_dir / "splits.npz", **splits)
    log.info(f"splits.npz guardado")

    # ── ✅ Guardar feature_cols.json (necesario para phase4) ──────────────
    save_json(proc_dir / "feature_cols.json", feature_names)
    log.info(f"feature_cols.json guardado: {feature_names}")

    # ── ✅ Guardar fd_real.npz (necesario para phase5) ────────────────────
    # Usamos el conjunto completo (antes del split) para tener más estadística
    save_npz(
        proc_dir / "fd_real.npz",
        logE = logE.astype(np.float32),
        xmax = y.astype(np.float32),    # y ya tiene la corrección aplicada
    )
    log.info(f"fd_real.npz guardado: {len(logE)} eventos")

    # ── Guardar meta.json ─────────────────────────────────────────────────
    p0   = cfg["phase0"]
    meta = {
        "n_features"   : int(X.shape[1]),
        "feature_names": feature_names,
        "n_train"      : int(len(splits["X_train"])),
        "n_val"        : int(len(splits["X_val"])),
        "n_test"       : int(len(splits["X_test"])),
        "target_col"   : p0["target_col"],
        "energy_col"   : p0["energy_col"],
        "E_min"        : float(p0["E_min"]),
        "E_max"        : float(p0["E_max"]),
        "correction"   : p0.get("target_correction", None),
    }
    save_json(proc_dir / "meta.json", meta)
    log.info("meta.json guardado")

    log.info("FASE 0 completada ✓")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)