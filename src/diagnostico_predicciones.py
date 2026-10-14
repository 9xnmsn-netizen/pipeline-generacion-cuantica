"""
Diagnóstico rápido del estado actual de predicciones.
Responde: ¿qué rango tiene xmax_pred? ¿hay correlación con logE?
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ── Cargar predicciones ──────────────────────────────────────
pred_dir = Path("/home2/DSMaster/evarela/Simon/qml_xmax/data/results/4q_3l_angle")
# Buscar el mejor config: nq=4, L=3, angle
pred_file = pred_dir / "predictions_4q_3l_angle.npz"
data = np.load(pred_file)

y_pred  = data["y_pred"]    # predicciones en test
y_true  = data["y_true"]    # valores reales en test
logE    = data["logE_test"] # energías en test

print("=== DIAGNÓSTICO DE PREDICCIONES ===")
print(f"y_pred rango:  [{y_pred.min():.1f}, {y_pred.max():.1f}] g/cm²")
print(f"y_true rango:  [{y_true.min():.1f}, {y_true.max():.1f}] g/cm²")
print(f"y_pred media:  {y_pred.mean():.1f} ± {y_pred.std():.1f} g/cm²")
print(f"y_true media:  {y_true.mean():.1f} ± {y_true.std():.1f} g/cm²")
print(f"logE  rango:   [{logE.min():.2f}, {logE.max():.2f}]")
print(f"Correlación(logE, y_pred): {np.corrcoef(logE, y_pred)[0,1]:.4f}")
print(f"Correlación(logE, y_true): {np.corrcoef(logE, y_true)[0,1]:.4f}")

# ── Figura diagnóstico ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: y_pred vs y_true
axes[0].scatter(y_true, y_pred, alpha=0.3, s=5)
axes[0].plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--')
axes[0].set_xlabel("Xmax real [g/cm²]")
axes[0].set_ylabel("Xmax predicho [g/cm²]")
axes[0].set_title("Predicción vs Real")

# Panel 2: y_pred vs logE
axes[1].scatter(logE, y_pred, alpha=0.3, s=5, label="pred")
axes[1].scatter(logE, y_true, alpha=0.1, s=5, label="real", color='orange')
axes[1].set_xlabel("log₁₀(E/eV)")
axes[1].set_ylabel("Xmax [g/cm²]")
axes[1].set_title("Xmax vs logE")
axes[1].legend()

# Panel 3: histograma de residuos
residuos = y_pred - y_true
axes[2].hist(residuos, bins=50, edgecolor='black')
axes[2].axvline(0, color='r', linestyle='--')
axes[2].set_xlabel("Residuo [g/cm²]")
axes[2].set_title(f"Residuos: μ={residuos.mean():.1f}, σ={residuos.std():.1f}")

plt.tight_layout()
plt.savefig("data/results/figures/diagnostico_predicciones_6q_1l_angle.png", dpi=150)
print("Figura guardada.")