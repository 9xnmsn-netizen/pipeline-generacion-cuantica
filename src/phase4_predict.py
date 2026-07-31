"""
phase4_predict.py
-----------------
Fase 4: Predicción sobre datos nuevos (SDnewversion.csv).

Para cada modelo entrenado en data/results/model_{combo}.json:
  1. Carga SDnewversion.csv
  2. Selecciona las mismas columnas que en fase 0 (feature_cols.json)
  3. Aplica la misma normalización PCA que en fase 1 (reduced.npz → scaler)
  4. Aplica el mismo circuito cuántico con el mismo θ (qfeatures_{combo}.npz)
  5. Aplica la misma normalización [-π, π] (norm_X_min/max del .npz)
  6. Predice con el modelo XGBoost guardado
  7. Guarda predictions_{combo}.npz en data/results/

IMPORTANTE:
  - No se reajusta nada (ni PCA, ni θ, ni XGBoost)
  - Todo lo que se ajustó en train se carga desde archivos
  - La energía (logE) se extrae de SDnewversion.csv para las gráficas de fase 5
"""

from pyexpat import model
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils  import load_config, get_paths, get_combos, combo_name
from src.utils.io_utils      import load_npz, save_npz, load_json, ensure_dirs
from src.utils.log_utils     import get_logger
from src.utils.quantum_utils import build_circuit, get_qfeatures

log = get_logger("phase4")

# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — Carga y validación de SDnewversion.csv
# ══════════════════════════════════════════════════════════════════════════════

# ✅ REEMPLAZAR POR
def load_new_data(cfg: dict) -> pd.DataFrame:
    """
    Carga SDnewversion.csv detectando automáticamente el separador.
    """
    p4 = cfg["phase4"]
    new_path = Path(p4["new_data_path"])

    if not new_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos nuevos: {new_path}\n"
            f"Verifica la clave 'phase4.new_data_path' en config.yaml"
        )

    # ── Detectar separador automáticamente ───────────────────────────────
    with open(new_path, "r") as f:
        first_line = f.readline()
    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    log.info(f"SDnewversion.csv: separador detectado = {repr(sep)}")

    df = pd.read_csv(new_path, sep=sep, low_memory=False)

    if df.empty:
        raise ValueError(f"SDnewversion.csv está vacío: {new_path}")

    log.info(
        f"SDnewversion.csv cargado: {df.shape[0]} filas, "
        f"{df.shape[1]} columnas"
    )
    log.info(f"Columnas disponibles ({len(df.columns)}): {list(df.columns[:10])}...")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — Selección de features y extracción de logE
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_new(
    df           : pd.DataFrame,
    feature_cols : list[str],
    cfg          : dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae la matriz de features X_new y el vector logE_new
    de SDnewversion.csv.

    Los features se seleccionan en el MISMO orden que en fase 0
    (leído de feature_cols.json).

    Parameters
    ----------
    df           : DataFrame de SDnewversion.csv
    feature_cols : lista de nombres de columnas (de feature_cols.json)
    cfg          : configuración global

    Returns
    -------
    X_new    : float32 array (N, n_features)
    logE_new : float32 array (N,)
    """
    p0         = cfg["phase0"]
    energy_col = p0["energy_col"]

    # ── Verificar columnas de features ───────────────────────────────────
    # ✅ REEMPLAZAR POR — imputar columnas faltantes con 0
    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        log.warning(
            f"SDnewversion.csv no tiene {len(missing_feat)} columnas de features.\n"
            f"  Faltantes (primeras 5): {missing_feat[:5]}...\n"
            f"  → Imputando con 0.0 (columnas FD no disponibles en SD-only)"
        )
        # Añadir columnas faltantes con valor 0
        for col in missing_feat:
            df[col] = 0.0

    log.info(
        f"Features disponibles: {len(feature_cols) - len(missing_feat)}/{len(feature_cols)} "
        f"| Imputadas con 0: {len(missing_feat)}"
    )

    # ── Verificar columna de energía ──────────────────────────────────────
    if energy_col not in df.columns:
        log.warning(
            f"Columna de energía '{energy_col}' no encontrada en "
            f"SDnewversion.csv. logE_new será array de NaN."
        )
        logE_new = np.full(len(df), np.nan, dtype=np.float32)
    else:
        logE_new = df[energy_col].values.astype(np.float32)

    # ── Extraer features en orden correcto ────────────────────────────────
    X_new = df[feature_cols].values.astype(np.float32)

    # ── Imputar NaN con 0 (placeholder) ──────────────────────────────────
    # En producción real se debería usar la mediana de train
    nan_count = np.isnan(X_new).sum()
    if nan_count > 0:
        log.warning(
            f"NaN en X_new: {nan_count} valores → imputando con 0.0\n"
            f"  (Para mayor precisión, usar la mediana de train)"
        )
        X_new = np.nan_to_num(X_new, nan=0.0)

    # ── Eliminar filas con logE NaN si hay columna de energía ─────────────
    valid = ~np.isnan(logE_new)
    if (~valid).sum() > 0:
        log.warning(
            f"Eliminando {(~valid).sum()} filas con logE NaN"
        )
        X_new    = X_new[valid]
        logE_new = logE_new[valid]

    log.info(
        f"X_new: {X_new.shape} | "
        f"logE_new: [{np.nanmin(logE_new):.2f}, {np.nanmax(logE_new):.2f}]"
    )

    # ── AÑADIR DESPUÉS de extraer logE_new ───────────────────────────────────
    # Filtrar por rango de energía físicamente válido
    p0 = cfg["phase0"]
    E_min_raw = float(p0.get("E_min", 18.5))  # en log10 si la energía es log10
    E_max_raw = float(p0.get("E_max", 20.0))

    # Detectar si logE_new es log10(E) o E directamente
    # Del log vemos: logE_new: [0.00, 144115276036786094080.00]
    # → los datos están en eV (no en log10), igual que en fase 0
    E_min_filter = 10 ** E_min_raw   # 10^18.5 eV
    E_max_filter = 10 ** E_max_raw   # 10^20.0 eV

    n_before = len(logE_new)
    valid_energy = (logE_new >= E_min_filter) & (logE_new <= E_max_filter)
    X_new    = X_new[valid_energy]
    logE_new = logE_new[valid_energy]

    log.info(
        f"Filtro energía: {valid_energy.sum()} / {n_before} filas válidas "
        f"({100*valid_energy.mean():.1f}%)"
    )

    if len(logE_new) == 0:
        log.warning("Filtro de energía dejó 0 filas → saltando combinación")
        return None, None   # ← necesitas manejar este caso en predict_combo

    return X_new, logE_new


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — Aplicar reducción PCA (cargada de fase 1)
# ══════════════════════════════════════════════════════════════════════════════

def apply_pca_reduction(
    X_new    : np.ndarray,
    proc_dir : Path,
    cfg      : dict,
) -> np.ndarray:
    """
    Aplica la reducción de dimensionalidad ajustada en fase 1.

    Soporta:
      - method='pca'  → carga pca_scaler.pkl + pca_model.pkl
      - method='umap' → carga umap_model.pkl
      - method='none' → devuelve X_new sin cambios

    Parameters
    ----------
    X_new    : float32 array (N, n_features)
    proc_dir : carpeta data/processed/
    cfg      : configuración global

    Returns
    -------
    X_new_r : float32 array (N, n_components)
    """
    p1     = cfg["phase1"]
    method = p1.get("method", "pca").lower()

    if method == "none":
        log.info("Reducción: 'none' → sin transformación")
        return X_new

    if method == "pca":
        scaler_path = proc_dir / "pca_scaler.pkl"
        pca_path    = proc_dir / "pca_model.pkl"

        # ── Verificar existencia ──────────────────────────────────────────
        for p in [scaler_path, pca_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Archivo de fase 1 no encontrado: {p}\n"
                    f"Asegúrate de haber ejecutado la fase 1 primero."
                )

        # ── Cargar scaler y PCA ───────────────────────────────────────────
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        # ── Transformar (sin reajustar) ───────────────────────────────────
        X_scaled = scaler.transform(X_new)
        X_new_r  = pca.transform(X_scaled).astype(np.float32)

        log.info(
            f"PCA aplicado: {X_new.shape} → {X_new_r.shape} "
            f"(varianza acumulada: {pca.explained_variance_ratio_.sum():.3f})"
        )
        return X_new_r

    if method == "umap":
        umap_path = proc_dir / "umap_model.pkl"
        if not umap_path.exists():
            raise FileNotFoundError(
                f"Modelo UMAP no encontrado: {umap_path}\n"
                f"Asegúrate de haber ejecutado la fase 1 primero."
            )
        with open(umap_path, "rb") as f:
            umap_model = pickle.load(f)

        X_new_r = umap_model.transform(X_new).astype(np.float32)
        log.info(f"UMAP aplicado: {X_new.shape} → {X_new_r.shape}")
        return X_new_r

    raise ValueError(
        f"Método de reducción '{method}' no reconocido. "
        f"Opciones: 'pca', 'umap', 'none'"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — Normalización [-π, π] y circuito cuántico
# ══════════════════════════════════════════════════════════════════════════════

def apply_quantum_encoding(
    X_new_r  : np.ndarray,
    nq       : int,
    L        : int,
    enc      : str,
    npz_data : dict,
    cfg      : dict,
) -> np.ndarray:
    """
    Aplica la misma normalización [-π, π] y el mismo circuito cuántico
    que en fase 2, usando los parámetros guardados en el .npz.

    Parameters
    ----------
    X_new_r  : float32 array (N, n_components) — datos reducidos
    nq       : número de qubits
    L        : número de capas
    enc      : encoding 'angle' | 'zz'
    npz_data : contenido del qfeatures_{combo}.npz
               debe tener: norm_X_min, norm_X_max, theta
    cfg      : configuración global

    Returns
    -------
    X_new_q : float32 array (N, nq) — features cuánticas
    """
    p2         = cfg["phase2"]
    batch_size = int(p2.get("batch_size", 64))

    # ── Cargar parámetros de normalización guardados en fase 2 ────────────
    if "norm_X_min" not in npz_data or "norm_X_max" not in npz_data:
        raise KeyError(
            "El archivo .npz no contiene 'norm_X_min' / 'norm_X_max'.\n"
            "Regenera la fase 2 con la versión corregida de phase2."
        )

    X_min = npz_data["norm_X_min"]   # shape (n_components,)
    X_max = npz_data["norm_X_max"]   # shape (n_components,)
    theta = npz_data["theta"]        # shape (L, nq, 2)

    # ── Validar dimensiones ───────────────────────────────────────────────
    n_comp = X_new_r.shape[1]
    if X_min.shape[0] != n_comp:
        raise ValueError(
            f"Dimensión de X_new_r ({n_comp}) no coincide con "
            f"norm_X_min ({X_min.shape[0]}).\n"
            f"Verifica que la fase 1 se ejecutó con los mismos parámetros."
        )

    # ── Aplicar normalización [-π, π] ─────────────────────────────────────
    rng = X_max - X_min
    rng[rng == 0] = 1.0   # evitar división por cero

    X_new_n = (2.0 * np.pi * (X_new_r - X_min) / rng - np.pi).astype(
        np.float32
    )

    log.info(
        f"Normalización [-π, π] aplicada: "
        f"rango resultante [{X_new_n.min():.3f}, {X_new_n.max():.3f}]"
    )

    # ── Construir circuito y evaluar ──────────────────────────────────────
    circuit = build_circuit(nq, L, enc)

    log.info(f"Evaluando circuito (nq={nq}, L={L}, enc={enc}) "
             f"en {X_new_r.shape[0]} muestras...")

    X_new_q = get_qfeatures(
        X         = X_new_n,
        circuit   = circuit,
        theta     = theta,
        batch_size= batch_size,
        verbose   = True,
    )

    return X_new_q


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — Predicción con XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_model(
    X_new_q   : np.ndarray,
    model_path : Path,
) -> np.ndarray:
    """
    Carga el modelo XGBoost y predice sobre X_new_q.

    Parameters
    ----------
    X_new_q    : float32 array (N, nq)
    model_path : ruta al modelo .pkl guardado en fase 3

    Returns
    -------
    xmax_pred : float32 array (N,)
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"Asegúrate de haber ejecutado la fase 3 primero."
        )

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    xmax_pred = model.predict(X_new_q).astype(np.float32)

    log.info(
        f"Predicción completada: {len(xmax_pred)} muestras | "
        f"rango Xmax: [{xmax_pred.min():.1f}, {xmax_pred.max():.1f}] g/cm²"
    )
    return xmax_pred


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE POR COMBINACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def predict_combo(
    nq          : int,
    L           : int,
    enc         : str,
    df_new      : pd.DataFrame,
    feature_cols: list[str],
    proc_dir    : Path,
    results_dir : Path,
    cfg         : dict,
) -> str | None:
    """
    Ejecuta el pipeline completo de predicción para una combinación.

    Returns
    -------
    str  : ruta del archivo de predicciones guardado
    None : si hubo error
    """
    cname      = f"{nq}q_{L}l_{enc}"
    npz_fname  = f"qfeatures_{cname}.npz"
    out_fname  = f"predictions_{cname}.npz"

    npz_path   = proc_dir / npz_fname
    model_path = results_dir / cname / "model.json"   # ✅ ruta real
    out_path   = results_dir / cname / out_fname       # ✅ guardar junto al modelo

    log.info(f"\n── Combinación: {cname} ──────────────────────────────────")

    # ── Verificar que existen los archivos necesarios ─────────────────────
    missing = []
    if not npz_path.exists():
        missing.append(str(npz_path))
    if not model_path.exists():
        missing.append(str(model_path))

    if missing:
        log.warning(
            f"Archivos faltantes para {cname} → saltando:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
        return None

    # ── Cargar .npz de fase 2 ─────────────────────────────────────────────
    npz_data = load_npz(npz_path)

    # ── Paso 2: Features y logE ───────────────────────────────────────────
    X_new, logE_new = extract_features_new(df_new, feature_cols, cfg)

    # Convertir energía a log10 para que sea compatible con fase 5
    # logE_new está en eV → convertir a log10(E/eV)
    logE_new_log10 = np.where(
        logE_new > 0,
        np.log10(logE_new.astype(np.float64)).astype(np.float32),
        np.nan
    )


    # ── Paso 3: Reducción PCA ─────────────────────────────────────────────
    X_new_r = apply_pca_reduction(X_new, proc_dir, cfg)

    # ── Paso 4: Circuito cuántico ─────────────────────────────────────────
    X_new_q = apply_quantum_encoding(
        X_new_r  = X_new_r,
        nq       = nq,
        L        = L,
        enc      = enc,
        npz_data = npz_data,
        cfg      = cfg,
    )

    # ── Paso 5: Predicción ────────────────────────────────────────────────
    xmax_pred = predict_with_model(X_new_q, model_path)

    # ── Guardar predicciones ──────────────────────────────────────────────
    save_npz(
        out_path,
        logE_new   = logE_new_log10,  # log10(E/eV) para fase 5
        xmax_pred  = xmax_pred,
        combo_name = np.array(cname),   # guardado como array escalar
    )
    log.info(f"Predicciones guardadas: {out_path}")
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    log.info("=" * 60)
    log.info("FASE 4 — Predicción sobre datos nuevos")
    log.info("=" * 60)

    paths       = get_paths(cfg)
    proc_dir    = paths["processed_dir"]
    results_dir = paths["results_dir"]
    ensure_dirs(results_dir)

    # ── Cargar feature_cols.json ──────────────────────────────────────────
    feat_path = proc_dir / "feature_cols.json"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"No se encontró feature_cols.json en {proc_dir}\n"
            f"Asegúrate de haber ejecutado la fase 0 primero."
        )
    feature_cols = load_json(feat_path)
    log.info(f"Features cargadas ({len(feature_cols)}): {feature_cols}")

    # ── Cargar SDnewversion.csv ───────────────────────────────────────────
    df_new = load_new_data(cfg)

    # ── Iterar sobre combinaciones ────────────────────────────────────────
    combos = get_combos(cfg)
    log.info(f"Combinaciones a predecir: {len(combos)}")

    saved  = []
    errors = []

    for idx, c in enumerate(combos, start=1):
        nq, L, enc = c

        log.info(f"\n[{idx}/{len(combos)}] nq={nq}, L={L}, enc='{enc}'")

        try:
            out = predict_combo(
                nq           = nq,
                L            = L,
                enc          = enc,
                df_new       = df_new,
                feature_cols = feature_cols,
                proc_dir     = proc_dir,
                results_dir  = results_dir,
                cfg          = cfg,
            )
            if out:
                saved.append(out)
            else:
                errors.append(f"{nq}q_{L}l_{enc} (archivos faltantes)")

        except Exception as e:
            import traceback
            log.error(f"ERROR en {nq}q_{L}l_{enc}: {e}")
            log.debug(traceback.format_exc())
            errors.append(f"{nq}q_{L}l_{enc} ({type(e).__name__})")

    # ── Resumen ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"FASE 4 completada ✓")
    log.info(f"  Predicciones guardadas : {len(saved)}")
    log.info(f"  Combinaciones con error: {len(errors)}")
    if errors:
        log.warning("  Errores en: " + ", ".join(errors))
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)