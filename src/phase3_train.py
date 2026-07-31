"""
phase3_train.py
---------------
Fase 3: Entrenamiento del modelo híbrido cuántico-clásico.

Para cada archivo qfeatures_*.npz generado en Fase 2:
  1. Carga las features cuánticas (X_train_q, X_val_q, X_test_q)
  2. Entrena un regresor XGBoost
  3. Evalúa en train / val / test
  4. Guarda el modelo y las métricas en data/results/{combo}/

Contrato de salida (por combinación):
  data/results/{nq}q_{L}l_{enc}/
      model.json       ← modelo XGBoost serializado
      metrics.json     ← R², MAE, RMSE para train/val/test
      predictions.npz  ← y_pred_train, y_pred_val, y_pred_test
                          + y_true_*, logE_*
"""

import sys
import json
import glob
import numpy as np
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.io_utils      import load_npz, save_npz, ensure_dirs
from src.utils.log_utils     import get_logger

log = get_logger("phase3")


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split:  str,
) -> dict:
    """
    Calcula R², MAE y RMSE para un split dado.

    Parameters
    ----------
    y_true : valores reales
    y_pred : predicciones del modelo
    split  : 'train' | 'val' | 'test'

    Returns
    -------
    dict con claves f"{split}_r2", f"{split}_mae", f"{split}_rmse"
    """
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    log.info(
        f"  [{split:>5}] R²={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}"
    )

    return {
        f"{split}_r2":   r2,
        f"{split}_mae":  mae,
        f"{split}_rmse": rmse,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def build_xgb_model(p3: dict) -> xgb.XGBRegressor:
    """
    Construye el modelo XGBoost con hiperparámetros del config.

    Hiperparámetros soportados en config.yaml → phase3:
      n_estimators, max_depth, learning_rate,
      subsample, colsample_bytree, reg_lambda, reg_alpha,
      min_child_weight, random_state
    """
    params = {
        "n_estimators":     int(p3.get("n_estimators",    300)),
        "max_depth":        int(p3.get("max_depth",        6)),
        "learning_rate":    float(p3.get("learning_rate",  0.05)),
        "subsample":        float(p3.get("subsample",      0.8)),
        "colsample_bytree": float(p3.get("colsample_bytree", 0.8)),
        "reg_lambda":       float(p3.get("reg_lambda",     1.0)),
        "reg_alpha":        float(p3.get("reg_alpha",      0.0)),
        "min_child_weight": int(p3.get("min_child_weight", 1)),
        "random_state":     int(p3.get("random_seed",      42)),
        "n_jobs":           int(p3.get("n_jobs",           -1)),
        "verbosity":        0,
        "objective":        "reg:squarederror",
    }

    log.info(f"  Hiperparámetros XGBoost: {params}")
    return xgb.XGBRegressor(**params)


# ✅ REEMPLAZAR POR
def train_model(X_train, y_train, X_val, y_val, p3):
    """
    Entrena XGBoost.
    Detecta la versión de XGBoost automáticamente:
    - >= 2.0 : early_stopping_rounds va en el constructor
    - <  2.0 : no soporta early_stopping_rounds → entrena todas las iteraciones
    """
    import xgboost as xgb

    early_stopping = int(p3.get("early_stopping_rounds", 30))

    # ── Detectar versión ──────────────────────────────────────────────────
    xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    log.info(f" XGBoost versión: {xgb.__version__}")

    params = {
        "n_estimators"   : int(p3.get("n_estimators", 300)),
        "max_depth"      : int(p3.get("max_depth", 6)),
        "learning_rate"  : float(p3.get("learning_rate", 0.05)),
        "subsample"      : float(p3.get("subsample", 0.8)),
        "colsample_bytree": float(p3.get("colsample_bytree", 0.8)),
        "reg_lambda"     : float(p3.get("reg_lambda", 1.0)),
        "reg_alpha"      : float(p3.get("reg_alpha", 0.0)),
        "min_child_weight": int(p3.get("min_child_weight", 1)),
        "random_state"   : int(p3.get("random_seed", 42)),
        "n_jobs"         : int(p3.get("n_jobs", -1)),
        "verbosity"      : 0,
        "objective"      : "reg:squarederror",
    }

    if xgb_version >= (2, 0):
        # ── XGBoost >= 2.0: early_stopping en el constructor ─────────────
        params["early_stopping_rounds"] = early_stopping
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            verbose  = False,
        )
    else:
        # ── XGBoost < 2.0: sin early_stopping (entrena n_estimators fijo) ─
        log.warning(
            f"XGBoost {xgb.__version__} < 2.0: "
            "early_stopping desactivado. "
            "Se entrenarán todas las iteraciones."
        )
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            verbose  = False,
        )

    best_iter = getattr(model, "best_iteration", "N/A")
    log.info(f" Mejor iteración: {best_iter}")
    
    return model


# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO
# ══════════════════════════════════════════════════════════════════════════════

def _combo_dir(results_dir: Path, nq: int, L: int, enc: str) -> Path:
    """Devuelve el directorio de resultados para una combinación."""
    return results_dir / f"{nq}q_{L}l_{enc}"


def save_results(
    combo_dir:    Path,
    model:        xgb.XGBRegressor,
    metrics:      dict,
    y_pred_train: np.ndarray,
    y_pred_val:   np.ndarray,
    y_pred_test:  np.ndarray,
    y_true_train: np.ndarray,
    y_true_val:   np.ndarray,
    y_true_test:  np.ndarray,
    logE_train:   np.ndarray,
    logE_val:     np.ndarray,
    logE_test:    np.ndarray,
    cfg_combo:    dict,
) -> None:
    """
    Guarda modelo, métricas y predicciones en combo_dir.

    Estructura:
      combo_dir/
        model.json        ← XGBoost nativo
        metrics.json      ← dict con R²/MAE/RMSE por split
        predictions.npz   ← predicciones y valores reales
    """
    ensure_dirs(combo_dir)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model_path = combo_dir / "model.json"
    model.save_model(str(model_path))
    log.info(f"  Modelo guardado: {model_path}")

    # ── Métricas ──────────────────────────────────────────────────────────────
    metrics["config"] = cfg_combo
    metrics_path = combo_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Métricas guardadas: {metrics_path}")

    # ── Predicciones ──────────────────────────────────────────────────────────
    pred_path = combo_dir / "predictions.npz"
    save_npz(
        pred_path,
        y_pred_train = y_pred_train,
        y_pred_val   = y_pred_val,
        y_pred_test  = y_pred_test,
        y_true_train = y_true_train,
        y_true_val   = y_true_val,
        y_true_test  = y_true_test,
        logE_train   = logE_train,
        logE_val     = logE_val,
        logE_test    = logE_test,
    )
    log.info(f"  Predicciones guardadas: {pred_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE POR COMBINACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def process_combo(
    qf_path:     Path,
    results_dir: Path,
    p3:          dict,
) -> dict:
    """
    Ejecuta el pipeline completo para una combinación (nq, L, enc).

    Parameters
    ----------
    qf_path     : ruta al archivo qfeatures_*.npz
    results_dir : directorio raíz de resultados
    p3          : sección phase3 del config

    Returns
    -------
    metrics : dict con métricas de train/val/test
    """
    # ── Parsear nombre de archivo ─────────────────────────────────────────────
    # Formato: qfeatures_{nq}q_{L}l_{enc}.npz
    stem   = qf_path.stem                     # "qfeatures_4q_3l_angle"
    rest   = stem[len("qfeatures_"):]            # "4q_3l_angle"
    nq_str, L_str, enc = rest.split("_", 2)     # split máximo 2 veces
    nq = int(nq_str.replace("q", ""))
    L  = int(L_str.replace("l", ""))            # enc queda como "angle" o "zz"


    log.info(f"\n{'─'*55}")
    log.info(f"Procesando: nq={nq} | L={L} | encoding='{enc}'")
    log.info(f"{'─'*55}")

    # ── Verificar si ya existe (overwrite) ────────────────────────────────────
    combo_dir = _combo_dir(results_dir, nq, L, enc)
    if (combo_dir / "metrics.json").exists() and not p3.get("overwrite", True):
        log.info("  Ya existe y overwrite=False → saltando.")
        with open(combo_dir / "metrics.json") as f:
            return json.load(f)

    # ── Cargar features cuánticas ─────────────────────────────────────────────
    data = load_npz(qf_path)

    X_train = data["X_train_q"]
    X_val   = data["X_val_q"]
    X_test  = data["X_test_q"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    logE_train = data["logE_train"]
    logE_val   = data["logE_val"]
    logE_test  = data["logE_test"]

    log.info(
        f"  Shapes → "
        f"X_train: {X_train.shape} | "
        f"X_val: {X_val.shape} | "
        f"X_test: {X_test.shape}"
    )

    # ── Entrenar ──────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train, X_val, y_val, p3)

    # ── Predecir ──────────────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    # ── Métricas ──────────────────────────────────────────────────────────────
    metrics = {}
    metrics.update(compute_metrics(y_train, y_pred_train, "train"))
    metrics.update(compute_metrics(y_val,   y_pred_val,   "val"))
    metrics.update(compute_metrics(y_test,  y_pred_test,  "test"))

    # ── Guardar ───────────────────────────────────────────────────────────────
    cfg_combo = {"nq": nq, "L": L, "encoding": enc}
    save_results(
        combo_dir    = combo_dir,
        model        = model,
        metrics      = metrics,
        y_pred_train = y_pred_train,
        y_pred_val   = y_pred_val,
        y_pred_test  = y_pred_test,
        y_true_train = y_train,
        y_true_val   = y_val,
        y_true_test  = y_test,
        logE_train   = logE_train,
        logE_val     = logE_val,
        logE_test    = logE_test,
        cfg_combo    = cfg_combo,
    )

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(results_dir: Path, all_metrics: list[dict]) -> None:
    """
    Guarda un resumen global de todas las combinaciones en:
      data/results/summary.json

    Ordenado por R² de test (descendente).
    """
    # Ordenar por test_r2
    all_metrics_sorted = sorted(
        all_metrics,
        key=lambda m: m.get("test_r2", -999),
        reverse=True,
    )

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics_sorted, f, indent=2)

    log.info(f"\nResumen global guardado: {summary_path}")
    log.info(f"{'─'*55}")
    log.info(f"{'Config':30s} {'R²_train':>8} {'R²_val':>8} {'R²_test':>8}")
    log.info(f"{'─'*55}")

    for m in all_metrics_sorted:
        cfg = m.get("config", {})
        tag = f"nq={cfg.get('nq','?')} L={cfg.get('L','?')} {cfg.get('encoding','?')}"
        log.info(
            f"{tag:30s} "
            f"{m.get('train_r2', float('nan')):8.4f} "
            f"{m.get('val_r2',   float('nan')):8.4f} "
            f"{m.get('test_r2',  float('nan')):8.4f}"
        )

    log.info(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    """
    Ejecuta la Fase 3 completa.

    Busca todos los archivos qfeatures_*.npz en data/processed/
    y entrena un modelo XGBoost por cada combinación.
    """
    log.info("=" * 60)
    log.info("FASE 3 — Entrenamiento híbrido cuántico-clásico")
    log.info("=" * 60)

    paths       = get_paths(cfg)
    p3          = cfg["phase3"]
    proc_dir    = paths["processed_dir"]
    results_dir = paths["results_dir"]

    ensure_dirs(results_dir)

    # ── Descubrir archivos de features ────────────────────────────────────────
    pattern    = str(proc_dir / "qfeatures_*.npz")
    qf_paths   = sorted(Path(p) for p in glob.glob(pattern))

    if not qf_paths:
        log.error(
            f"No se encontraron archivos qfeatures_*.npz en {proc_dir}. "
            "Ejecuta primero la Fase 2."
        )
        return

    log.info(f"Archivos de features encontrados: {len(qf_paths)}")
    for p in qf_paths:
        log.info(f"  {p.name}")

    # ── Procesar cada combinación ─────────────────────────────────────────────
    all_metrics = []

    for idx, qf_path in enumerate(qf_paths, start=1):
        log.info(f"\n[{idx}/{len(qf_paths)}] {qf_path.name}")
        try:
            metrics = process_combo(qf_path, results_dir, p3)
            all_metrics.append(metrics)
        except Exception as e:
            log.error(f"  ERROR en {qf_path.name}: {e}")
            import traceback
            log.error(traceback.format_exc())

    # ── Resumen global ────────────────────────────────────────────────────────
    if all_metrics:
        save_summary(results_dir, all_metrics)

    log.info(f"\nFASE 3 completada ✓ — {len(all_metrics)} modelos entrenados")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)