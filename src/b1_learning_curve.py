# scripts/b1_learning_curve.py
"""
Curva de aprendizaje del modelo híbrido QML-XGBoost.
Evalúa R², MAE y RMSE en train/val al variar el tamaño del conjunto de entrenamiento.
Usa la mejor configuración conocida: 4q_3l_angle
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results", "learning_curve")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG_NAME = "4q_3l_angle"   # mejor configuración conocida

# ── Fracciones del conjunto de entrenamiento a evaluar ───────────────────────
FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
SEED       = 42

# OJO: Cambiar Pipeline Actual (22/08/2026, 12:29 pm)
# ── Hiperparámetros XGBoost (mismos que pipeline actual) ────────────────────
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     SEED,
    "n_jobs":           -1,
}

# ── Funciones auxiliares ─────────────────────────────────────────────────────

def load_npz(path: str) -> dict:
    """Carga un archivo .npz y retorna diccionario con arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula R², MAE y RMSE."""
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_feature_matrix(splits: dict, qfeat: dict, split: str) -> np.ndarray:
    """
    Construye la matriz de features combinando PCA + cuánticas.
    split: 'train', 'val' o 'test'
    """
    # Features clásicas reducidas por PCA
    X_pca = splits.get(f"X_{split}_r")
    if X_pca is None:
        # Intentar desde reduced.npz si no están en splits
        reduced = load_npz(os.path.join(PROC_DIR, "reduced.npz"))
        X_pca = reduced[f"X_{split}_r"]

    # Features cuánticas
    X_q = qfeat[f"X_{split}_q"]

    return np.hstack([X_pca, X_q])


# ── Carga de datos ───────────────────────────────────────────────────────────

print(f"[B-1] Cargando datos para configuración: {CONFIG_NAME}")

splits  = load_npz(os.path.join(PROC_DIR, "splits.npz"))
reduced = load_npz(os.path.join(PROC_DIR, "reduced.npz"))
qfeat   = load_npz(os.path.join(PROC_DIR, f"qfeatures_{CONFIG_NAME}.npz"))

# Etiquetas
y_train = splits["y_train"]
y_val   = splits["y_val"]

# Matrices de features
X_train_pca = reduced["X_train_r"]
X_val_pca   = reduced["X_val_r"]
X_train_q   = qfeat["X_train_q"]
X_val_q     = qfeat["X_val_q"]

X_train_full = np.hstack([X_train_pca, X_train_q])
X_val_full   = np.hstack([X_val_pca,   X_val_q])

print(f"  X_train shape : {X_train_full.shape}")
print(f"  X_val   shape : {X_val_full.shape}")
print(f"  y_train shape : {y_train.shape}")

# ── Curva de aprendizaje ─────────────────────────────────────────────────────

results = []
rng = np.random.default_rng(SEED)

print(f"\n[B-1] Evaluando {len(FRACTIONS)} fracciones de entrenamiento...\n")

for frac in FRACTIONS:
    n_samples = max(10, int(len(X_train_full) * frac))

    # Submuestreo aleatorio reproducible
    idx = rng.choice(len(X_train_full), size=n_samples, replace=False)
    X_sub = X_train_full[idx]
    y_sub = y_train[idx]

    # Entrenar modelo
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_sub, y_sub, verbose=False)

    # Evaluar
    m_train = compute_metrics(y_sub,       model.predict(X_sub))
    m_val   = compute_metrics(y_val,       model.predict(X_val_full))

    row = {
        "fraction":    frac,
        "n_samples":   n_samples,
        "train_r2":    m_train["r2"],
        "train_mae":   m_train["mae"],
        "train_rmse":  m_train["rmse"],
        "val_r2":      m_val["r2"],
        "val_mae":     m_val["mae"],
        "val_rmse":    m_val["rmse"],
    }
    results.append(row)

    print(f"  frac={frac:.2f}  n={n_samples:5d} | "
          f"train R²={m_train['r2']:+.4f}  val R²={m_val['r2']:+.4f} | "
          f"train MAE={m_train['mae']:.2f}  val MAE={m_val['mae']:.2f}")

# ── Guardar JSON ─────────────────────────────────────────────────────────────

out_json = os.path.join(RESULTS_DIR, f"lc_{CONFIG_NAME}.json")
with open(out_json, "w") as f:
    json.dump({
        "config":    CONFIG_NAME,
        "timestamp": datetime.now().isoformat(),
        "xgb_params": XGB_PARAMS,
        "results":   results,
    }, f, indent=2)
print(f"\n[B-1] Resultados guardados en: {out_json}")

# ── Gráficas ─────────────────────────────────────────────────────────────────

fracs      = [r["fraction"]   for r in results]
n_samples  = [r["n_samples"]  for r in results]
train_r2   = [r["train_r2"]   for r in results]
val_r2     = [r["val_r2"]     for r in results]
train_mae  = [r["train_mae"]  for r in results]
val_mae    = [r["val_mae"]    for r in results]
train_rmse = [r["train_rmse"] for r in results]
val_rmse   = [r["val_rmse"]   for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Curva de Aprendizaje — {CONFIG_NAME}\n"
    f"(XGBoost lr={XGB_PARAMS['learning_rate']}, "
    f"n_est={XGB_PARAMS['n_estimators']}, seed={SEED})",
    fontsize=13
)

# ── Panel 1: R² ──
ax = axes[0]
ax.plot(n_samples, train_r2, "o-", color="steelblue",  label="Train R²")
ax.plot(n_samples, val_r2,   "s--", color="tomato",    label="Val R²")
ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel("Número de muestras de entrenamiento")
ax.set_ylabel("R²")
ax.set_title("R² vs Tamaño de entrenamiento")
ax.legend()
ax.grid(True, alpha=0.3)

# ── Panel 2: MAE ──
ax = axes[1]
ax.plot(n_samples, train_mae, "o-",  color="steelblue", label="Train MAE")
ax.plot(n_samples, val_mae,   "s--", color="tomato",    label="Val MAE")
ax.set_xlabel("Número de muestras de entrenamiento")
ax.set_ylabel("MAE [g/cm²]")
ax.set_title("MAE vs Tamaño de entrenamiento")
ax.legend()
ax.grid(True, alpha=0.3)

# ── Panel 3: RMSE ──
ax = axes[2]
ax.plot(n_samples, train_rmse, "o-",  color="steelblue", label="Train RMSE")
ax.plot(n_samples, val_rmse,   "s--", color="tomato",    label="Val RMSE")
ax.set_xlabel("Número de muestras de entrenamiento")
ax.set_ylabel("RMSE [g/cm²]")
ax.set_title("RMSE vs Tamaño de entrenamiento")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_fig = os.path.join(RESULTS_DIR, f"lc_{CONFIG_NAME}.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"[B-1] Figura guardada en: {out_fig}")

# ── Diagnóstico final ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DIAGNÓSTICO DE LA CURVA DE APRENDIZAJE")
print("="*60)

gap_r2  = train_r2[-1]  - val_r2[-1]
gap_mae = val_mae[-1]   - train_mae[-1]

print(f"\n  R² train (100%): {train_r2[-1]:+.4f}")
print(f"  R² val   (100%): {val_r2[-1]:+.4f}")
print(f"  Gap R²         : {gap_r2:+.4f}")

print(f"\n  MAE train (100%): {train_mae[-1]:.2f} g/cm²")
print(f"  MAE val   (100%): {val_mae[-1]:.2f} g/cm²")
print(f"  Gap MAE         : {gap_mae:.2f} g/cm²")

# Interpretación automática
print("\n  ── Interpretación ──")
if gap_r2 > 0.15:
    print("  ⚠️  Alto sobreajuste: el modelo memoriza train pero no generaliza.")
    print("     → Reducir complejidad, aumentar regularización o más datos.")
elif val_r2[-1] < 0.10:
    print("  ⚠️  Subajuste severo: el modelo no aprende de los datos.")
    print("     → Revisar features, aumentar complejidad del circuito,")
    print("        o ajustar hiperparámetros de XGBoost.")
elif abs(gap_r2) < 0.05 and val_r2[-1] > 0.30:
    print("  ✅  Buen ajuste: train y val convergen con R² razonable.")
else:
    print("  🔶  Ajuste moderado: hay margen de mejora.")
    print("     → Probar B-2 (búsqueda de learning rate).")

# Tendencia: ¿mejora con más datos?
delta_val_r2 = val_r2[-1] - val_r2[0]
if delta_val_r2 > 0.02:
    print(f"  📈  El modelo mejora con más datos (Δval_R²={delta_val_r2:+.4f}).")
    print("     → Considerar aumentar el dataset.")
else:
    print(f"  📉  Más datos NO mejoran significativamente (Δval_R²={delta_val_r2:+.4f}).")
    print("     → El cuello de botella es la calidad de features, no la cantidad.")

print("\n[B-1] Script completado.\n")