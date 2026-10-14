"""
phase0_prepare.py  (versión corregida)
--------------------------------------
Cambios respecto a la versión anterior:
  ✅ Lee claves desde cfg["phase0"] (compatible con config.yaml corregido)
  ✅ Guarda feature_cols.json (necesario para phase4)
  ✅ Guarda fd_real.npz con logE y xmax del conjunto completo (para phase5)
  ✅ X contiene SOLO columnas SD + log10(SD_RS_fEnergy)
  ✅ Corrección al target ELIMINADA de esta fase (se aplica solo al graficar)
  ✅ Bloque include_logE ELIMINADO (innecesario)
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

    target_col    = p0.get("target_col", cfg.get("target_col", "FD_RS_fXmax"))
    energy_col    = p0.get("energy_col", cfg.get("energy_col", "FD_RS_logE"))
    fd_cols_keep  = p0.get("fd_cols_keep", [target_col, energy_col])

    fd_path = Path(p0["fd_file"])
    sd_path = Path(p0["sd_file"])

    if not fd_path.exists():
        raise FileNotFoundError(f"No se encontró FD: {fd_path}")
    if not sd_path.exists():
        raise FileNotFoundError(f"No se encontró SD: {sd_path}")

    df_fd = _read_csv_auto(fd_path)
    df_sd = _read_csv_auto(sd_path)
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

        fd_cols_present = [c for c in fd_cols_keep if c in df_fd.columns]
        missing_fd = [c for c in fd_cols_keep if c not in df_fd.columns]
        if missing_fd:
            log.warning(f"Columnas de FD no encontradas y se omiten: {missing_fd}")
        df_fd = df_fd[fd_cols_present]

        df = pd.concat([df_sd, df_fd], axis=1)
    else:
        df = pd.merge(df_sd, df_fd, on=merge_on, how="inner")

    if df.empty:
        raise ValueError("El DataFrame fusionado está vacío.")

    log.info(f"DataFrame fusionado: {df.shape}")
    print(df.columns.tolist())

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FILTRO DE ENERGÍA
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_energy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p0 = cfg["phase0"]
    energy_col = p0["energy_col"]
    E_min = 10 ** float(p0["E_min"])
    E_max = 10 ** float(p0["E_max"])

    if energy_col not in df.columns:
        raise KeyError(
            f"Columna de energía '{energy_col}' no encontrada.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    # ── DIAGNÓSTICO ──────────────────────────────────────────────────
    col_vals = df[energy_col].dropna()
    print(f"\n>>> Columna energía: {energy_col}")
    print(f">>> Rango en datos : [{col_vals.min():.3e}, {col_vals.max():.3e}]")
    print(f">>> Filtro aplicado: [{E_min:.3e}, {E_max:.3e}]")
    print(f">>> NaN en columna : {df[energy_col].isna().sum()}")
    # ─────────────────────────────────────────────────────────────────

    mask = (df[energy_col] >= E_min) & (df[energy_col] <= E_max)
    df_f = df[mask].reset_index(drop=True)

    print(f">>> Filas tras filtro: {len(df_f)} / {len(df)}")

    if df_f.empty:
        raise ValueError(
            f"Filtro de energía dejó 0 filas.\n"
            f"Rango en datos: [{col_vals.min():.3e}, {col_vals.max():.3e}]\n"
            f"Rango en config (E_min/E_max): [{E_min:.3e}, {E_max:.3e}]\n"
            f"Verifica las unidades: ¿eV o log10(eV/eV)?"
        )

    log.info(
        f"Filtro energía [{E_min:.3e}, {E_max:.3e}] en '{energy_col}': "
        f"{mask.sum()} / {len(df)} filas"
    )
    return df_f


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES Y TARGET
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_target(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:

    p0 = cfg["phase0"]
    target_col = p0["target_col"]
    energy_col = p0["energy_col"]
    exclude = set(p0.get("exclude_cols", []))

    if target_col not in df.columns:
        raise KeyError(
            f"Target '{target_col}' no encontrado.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

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

    # ── Construir X ────────────────────────────────────────────────────
    X_sd = df[feature_cols].values.astype(np.float32)
    logE_raw = df[energy_col].values.astype(np.float32)
    logE_feat = np.log10(logE_raw).reshape(-1, 1).astype(np.float32)
    X = np.hstack([X_sd, logE_feat])
    feature_cols = feature_cols + [f"log10_{energy_col}"]

    print(f"\n>>> X construida: {X.shape} | "
          f"features SD: {X_sd.shape[1]} | +1 log10(E)")

    # ── Imputar NaN en features (vectorizado) ──────────────────────────
    nan_X = np.isnan(X).sum()
    if nan_X > 0:
        nan_por_col = np.isnan(X).sum(axis=0)
        cols_con_nan = [(feature_cols[i], int(nan_por_col[i]))
                        for i in range(len(feature_cols))
                        if nan_por_col[i] > 0]
        print(f"\n>>> NaN por columna ({nan_X} total):")
        for col_name, n_nan in cols_con_nan:
            print(f"    {col_name}: {n_nan} NaN")

        log.warning(f"NaN en features: {nan_X} → imputando con mediana")
        medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X = np.where(nan_mask, medians[np.newaxis, :], X)   # ✅ vectorizado
        print(f">>> Imputación OK. NaN restantes: {np.isnan(X).sum()}")

    # ── Target ─────────────────────────────────────────────────────────
    y = df[target_col].values.astype(np.float32)
    logE = logE_raw.copy()

    # ── Eliminar filas con NaN en y o logE ────────────────────────────
    mask = ~(np.isnan(y) | np.isnan(logE))
    if (~mask).sum() > 0:
        log.warning(f"Eliminando {(~mask).sum()} filas con NaN en y o logE")
        X, y, logE = X[mask], y[mask], logE[mask]

    print(f"\n>>> Final → X: {X.shape} | "
          f"y: [{y.min():.1f}, {y.max():.1f}] g/cm² | "
          f"logE: [{np.log10(logE.min()):.2f}, {np.log10(logE.max()):.2f}]")

    log.info(f"Resultado final → X: {X.shape}")
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

    # Fracción relativa de val dentro de (train + val)
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
    
    # ── DIAGNÓSTICO RÁPIDO ────────────────────────────────────────────
    from src.utils.config_utils import load_config, get_paths
    p0 = cfg["phase0"]
    import pandas as pd, numpy as np
    sd_path = Path(p0["sd_file"])
    df_sd_check = pd.read_csv(sd_path, nrows=5)
    energy_col = p0["energy_col"]
    print(f"\n>>> energy_col en config: {energy_col}")
    if energy_col in df_sd_check.columns:
        print(f">>> Primeros valores de {energy_col}:")
        print(df_sd_check[energy_col].values)
    else:
        print(f">>> ⚠️  {energy_col} NO está en SD")
    print()
    # ── FIN DIAGNÓSTICO ───────────────────────────────────────────────

    paths = get_paths(cfg)

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
        xmax = y.astype(np.float32),    # ✅ SIN corrección
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