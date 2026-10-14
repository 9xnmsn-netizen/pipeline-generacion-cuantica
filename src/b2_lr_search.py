# scripts/b2_lr_search.py
"""
Búsqueda de learning rate óptimo para XGBoost en el modelo híbrido QML.
Evalúa un rango de learning rates con validación cruzada temporal y
guarda curvas de pérdida por n_estimators para cada LR.
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
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results", "lr_search")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG_NAME = "4q_3l_angle"
SEED        = 42

# ── Rango de learning rates a explorar ───────────────────────────────────────
LEARNING_RATES = [0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

# ── Hiperparámetros base (todo fijo excepto lr) ───────────────────────────────
BASE_PARAMS = {
    "n_estimators":     500,   # alto para ver convergencia
    "max_depth":        4,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     SEED,
    "n_jobs":           -1,
    "early_stopping_rounds": 30,  # detener si no mejora en 30 rondas
}

# ── Funciones auxiliares ─────────────────────────────────────────────────────

def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

# ── Carga de datos ───────────────────────────────────────────────────────────

print(f"[B-2] Cargando datos para configuración: {CONFIG_NAME}")

splits  = load_npz(os.path.join(PROC_DIR, "splits.npz"))
reduced = load_npz(os.path.join(PROC_DIR, "reduced.npz"))
qfeat   = load_npz(os.path.join(PROC_DIR, f"qfeatures_{CONFIG_NAME}.npz"))

y_train = splits["y_train"]
y_val   = splits["y_val"]
y_test  = splits["y_test"]

X_train = np.hstack([reduced["X_train_r"], qfeat["X_train_q"]])
X_val   = np.hstack([reduced["X_val_r"],   qfeat["X_val_q"]])
X_test  = np.hstack([reduced["X_test_r"],  qfeat["X_test_q"]])

print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# ── Búsqueda de LR ───────────────────────────────────────────────────────────

all_results   = []
loss_curves   = {}   # lr → historial de val MAE por n_estimators

print(f"\n[B-2] Evaluando {len(LEARNING_RATES)} learning rates...\n")
print(f"{'LR':>8} | {'Best iter':>9} | {'Val R²':>8} | {'Val MAE':>8} | "
      f"{'Val RMSE':>9} | {'Test R²':>8} | {'Test MAE':>8}")
print("-" * 72)

for lr in LEARNING_RATES:
    params = {**BASE_PARAMS, "learning_rate": lr}

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    y_val_pred  = model.predict(X_val,  iteration_range=(0, best_iter + 1))
    y_test_pred = model.predict(X_test, iteration_range=(0, best_iter + 1))

    m_val  = compute_metrics(y_val,  y_val_pred)
    m_test = compute_metrics(y_test, y_test_pred)

    # Historial de pérdida (MAE en val por ronda)
    evals_result = model.evals_result()
    val_history  = evals_result.get("validation_0", {}).get("rmse", [])
    loss_curves[lr] = val_history

    row = {
        "lr":          lr,
        "best_iter":   int(best_iter),
        "val_r2":      m_val["r2"],
        "val_mae":     m_val["mae"],
        "val_rmse":    m_val["rmse"],
        "test_r2":     m_test["r2"],
        "test_mae":    m_test["mae"],
        "test_rmse":   m_test["rmse"],
    }
    all_results.append(row)

    print(f"  {lr:>6.3f} | {best_iter:>9d} | {m_val['r2']:>+8.4f} | "
          f"{m_val['mae']:>8.2f} | {m_val['rmse']:>9.2f} | "
          f"{m_test['r2']:>+8.4f} | {m_test['mae']:>8.2f}")

# ── Mejor LR ─────────────────────────────────────────────────────────────────

best = max(all_results, key=lambda x: x["val_r2"])
print(f"\n  ✅ Mejor LR por val R²: {best['lr']}  "
      f"(val R²={best['val_r2']:+.4f}, val MAE={best['val_mae']:.2f})")

# ── Guardar JSON ─────────────────────────────────────────────────────────────

out_json = os.path.join(RESULTS_DIR, f"lr_search_{CONFIG_NAME}.json")
with open(out_json, "w") as f:
    json.dump({
        "config":      CONFIG_NAME,
        "timestamp":   datetime.now().isoformat(),
        "base_params": {k: v for k, v in BASE_PARAMS.items()
                        if k != "early_stopping_rounds"},
        "best_lr":     best["lr"],
        "results":     all_results,
    }, f, indent=2)
print(f"[B-2] Resultados guardados en: {out_json}")

# ── Gráfica 1: Métricas vs LR ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Búsqueda de Learning Rate — {CONFIG_NAME}\n"
    f"(n_estimators={BASE_PARAMS['n_estimators']}, early_stop=30)",
    fontsize=13
)

lrs      = [r["lr"]       for r in all_results]
val_r2   = [r["val_r2"]   for r in all_results]
val_mae  = [r["val_mae"]  for r in all_results]
val_rmse = [r["val_rmse"] for r in all_results]
test_r2  = [r["test_r2"]  for r in all_results]
test_mae = [r["test_mae"] for r in all_results]

# R²
ax = axes[0]
ax.semilogx(lrs, val_r2,  "o-",  color="steelblue", label="Val R²")
ax.semilogx(lrs, test_r2, "s--", color="tomato",    label="Test R²")
ax.axvline(best["lr"], color="green", linestyle=":", linewidth=1.5,
           label=f"Mejor LR={best['lr']}")
ax.set_xlabel("Learning Rate (escala log)")
ax.set_ylabel("R²")
ax.set_title("R² vs Learning Rate")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

# MAE
ax = axes[1]
ax.semilogx(lrs, val_mae,  "o-",  color="steelblue", label="Val MAE")
ax.semilogx(lrs, test_mae, "s--", color="tomato",    label="Test MAE")
ax.axvline(best["lr"], color="green", linestyle=":", linewidth=1.5,
           label=f"Mejor LR={best['lr']}")
ax.set_xlabel("Learning Rate (escala log)")
ax.set_ylabel("MAE [g/cm²]")
ax.set_title("MAE vs Learning Rate")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

# RMSE
ax = axes[2]
ax.semilogx(lrs, val_rmse, "o-", color="steelblue", label="Val RMSE")
ax.axvline(best["lr"], color="green", linestyle=":", linewidth=1.5,
           label=f"Mejor LR={best['lr']}")
ax.set_xlabel("Learning Rate (escala log)")
ax.set_ylabel("RMSE [g/cm²]")
ax.set_title("RMSE vs Learning Rate")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out_fig1 = os.path.join(RESULTS_DIR, f"lr_search_metrics_{CONFIG_NAME}.png")
plt.savefig(out_fig1, dpi=150, bbox_inches="tight")
plt.close()
print(f"[B-2] Figura 1 guardada en: {out_fig1}")

# ── Gráfica 2: Curvas de pérdida por LR ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.cm.viridis
colors = [cmap(i / len(LEARNING_RATES)) for i in range(len(LEARNING_RATES))]

for (lr, history), color in zip(loss_curves.items(), colors):
    if history:
        ax.plot(history, label=f"LR={lr}", color=color, linewidth=1.5)

ax.set_xlabel("Iteración (n_estimators)")
ax.set_ylabel("RMSE en validación [g/cm²]")
ax.set_title(
    f"Curvas de pérdida por Learning Rate — {CONFIG_NAME}\n"
    "(early stopping activo, se muestra hasta mejor iteración)"
)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_fig2 = os.path.join(RESULTS_DIR, f"lr_loss_curves_{CONFIG_NAME}.png")
plt.savefig(out_fig2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[B-2] Figura 2 guardada en: {out_fig2}")

# ── Resumen final ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESUMEN — BÚSQUEDA DE LEARNING RATE")
print("="*60)
print(f"\n  Configuración : {CONFIG_NAME}")
print(f"  Mejor LR      : {best['lr']}")
print(f"  Best iteration: {best['best_iter']}")
print(f"  Val  R²       : {best['val_r2']:+.4f}")
print(f"  Val  MAE      : {best['val_mae']:.2f} g/cm²")
print(f"  Val  RMSE     : {best['val_rmse']:.2f} g/cm²")
print(f"  Test R²       : {best['test_r2']:+.4f}")
print(f"  Test MAE      : {best['test_mae']:.2f} g/cm²")
print(f"\n  Archivos generados:")
print(f"    {out_json}")
print(f"    {out_fig1}")
print(f"    {out_fig2}")
print("\n[B-2] Script completado.\n")