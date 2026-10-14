"""
Compara predicciones del modelo con datos de:
- PRD 111, 022003 (2025): ⟨Xmax⟩ y σ(Xmax)
- PRD 90,  122005 (2014): ⟨Xmax⟩ y σ(Xmax)
Genera figura publicable y métricas de acuerdo.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ════════════════════════════════════════════════════════════
# DATOS DE PAPERS (digitalizados exactos)
# ════════════════════════════════════════════════════════════

# ── PRD 111, 022003 (2025) ───────────────────────────────────
PRD111 = {
    "logE":  [18.55, 18.65, 18.75, 18.85, 18.95,
              19.05, 19.15, 19.25, 19.35, 19.45,
              19.55, 19.64, 19.74, 19.85, 20.00],
    "xmax":  [757.0, 758.8, 759.5, 761.8, 765.7,
              770.0, 769.9, 774.0, 774.7, 775.3,
              778.6, 783.2, 787.2, 794.5, 793.9],
    "xmax_stat": [0.5, 0.5, 0.5, 0.5, 0.5,
                  0.6, 0.6, 0.8, 0.95, 1.0,
                  1.3, 1.4, 2.0, 3.6, 4.5],
    "xmax_syst": [7.7, 3.0, 7.9, 8.0, 8.5,
                  9.0, 7.0, 7.5, 0.0, 10.0,
                  19.5, 11.0, 11.6, 12.2, 13.1],
    "sigma":  [48.8, 43.4, 39.7, 38.4, 46.3,
               38.8, 34.0, 33.2, 30.7, 27.3,
               26.3, 24.2, 20.7, 25.1, 21.8],
    "sigma_stat": [0.6, 0.7, 0.6, 0.7, 0.6,
                   1.0, 0.8, 0.9, 1.26, 1.6,
                   1.3, 1.8, 1.4, 3.9, 5.4],
}

# ── PRD 90, 122005 (2014) ────────────────────────────────────
PRD90 = {
    "logE":  [17.85, 17.95, 18.05, 18.15, 18.25,
              18.35, 18.45, 18.55, 18.65, 18.75,
              18.85, 18.95, 19.05, 19.14, 19.25,
              19.34, 19.45, 19.62],
    "xmax":  [709.9, 719.9, 725.2, 736.9, 744.5,
              748.0, 752.2, 754.5, 756.1, 757.4,
              763.6, 764.6, 766.4, 767.0, 779.5,
              773.1, 787.9, 779.8],
    "xmax_stat": [1.2, 1.4, 1.5, 1.8, 2.0,
                  2.0, 2.1, 2.2, 2.7, 2.8,
                  2.9, 3.2, 3.3, 3.6, 5.1,
                  5.0, 9.6, 5.0],
    "sigma":  [59.6, 62.4, 59.5, 64.3, 66.4,
               60.2, 53.3, 53.5, 54.5, 45.8,
               42.8, 43.4, 39.0, 36.7, 46.4,
               40.1, 53.2, 26.5],
    "sigma_stat": [1.7, 2.1, 2.0, 2.6, 2.6,
                   2.8, 2.9, 3.0, 3.5, 3.4,
                   3.6, 4.1, 3.8, 3.6, 6.2,
                   4.8, 12.7, 4.8],
}

# ════════════════════════════════════════════════════════════
# CARGAR PREDICCIONES DEL MODELO
# ════════════════════════════════════════════════════════════

def calcular_momentos_por_bin(logE, xmax_pred, xmax_true,
                               bin_edges=None):
    """
    Calcula ⟨Xmax⟩ y σ(Xmax) por bin de energía.
    Devuelve: logE_centro, xmax_mean_pred, sigma_pred,
                           xmax_mean_true, sigma_true, n_eventos
    """
    if bin_edges is None:
        bin_edges = np.arange(17.8, 20.2, 0.1)

    logE_centro, xmax_pred_bin, sigma_pred_bin = [], [], []
    xmax_true_bin, sigma_true_bin, n_bin = [], [], []

    for i in range(len(bin_edges) - 1):
        mask = (logE >= bin_edges[i]) & (logE < bin_edges[i+1])
        n = mask.sum()
        if n >= 10:   # mínimo 10 eventos por bin
            logE_centro.append(0.5*(bin_edges[i] + bin_edges[i+1]))
            xmax_pred_bin.append(xmax_pred[mask].mean())
            sigma_pred_bin.append(xmax_pred[mask].std())
            xmax_true_bin.append(xmax_true[mask].mean())
            sigma_true_bin.append(xmax_true[mask].std())
            n_bin.append(n)

    return (np.array(logE_centro),
            np.array(xmax_pred_bin), np.array(sigma_pred_bin),
            np.array(xmax_true_bin), np.array(sigma_true_bin),
            np.array(n_bin))

def chi2_reducido(xmodel, xpaper, err_paper):
    """χ² reducido entre modelo y paper en bins comunes."""
    chi2 = np.sum(((xmodel - xpaper) / err_paper)**2)
    return chi2 / len(xmodel)

def graficar_comparacion(pred_file: Path, out_dir: Path,
                          label_config: str):
    """
    Genera figura de 2 paneles: ⟨Xmax⟩ y σ(Xmax) vs logE
    comparando modelo con PRD 111 y PRD 90.
    """
    data = np.load(pred_file)
    logE    = data["logE_test"]
    y_pred  = data["y_pred"]
    y_true  = data["y_true"]

    (logE_c, xp, sp,
     xt, st, n) = calcular_momentos_por_bin(logE, y_pred, y_true)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Comparación con Auger — {label_config}",
                 fontsize=13)

    # ── Panel izquierdo: ⟨Xmax⟩ ────────────────────────────
    ax = axes[0]

    # PRD 90 (2014)
    ax.errorbar(PRD90["logE"], PRD90["xmax"], PRD90["xmax_stat"],
                fmt='s', color='steelblue', ms=4, capsize=3,
                label="Auger PRD 90 (2014)", alpha=0.7)

    # PRD 111 (2025)
    ax.errorbar(PRD111["logE"], PRD111["xmax"], PRD111["xmax_stat"],
                fmt='o', color='navy', ms=5, capsize=3,
                label="Auger PRD 111 (2025)")

    # Modelo — predicciones
    ax.errorbar(logE_c, xp, sp/np.sqrt(n),
                fmt='D', color='crimson', ms=5, capsize=3,
                label=f"Modelo ({label_config})")

    # Modelo — valores reales en test
    ax.errorbar(logE_c, xt, st/np.sqrt(n),
                fmt='^', color='darkorange', ms=5, capsize=3,
                label="Real (test set)", alpha=0.7)

    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$", fontsize=12)
    ax.set_ylabel(r"$\langle X_\mathrm{max} \rangle$ [g/cm²]",
                  fontsize=12)
    ax.set_xlim(17.7, 20.2)
    ax.set_ylim(680, 820)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel derecho: σ(Xmax) ──────────────────────────────
    ax = axes[1]

    ax.errorbar(PRD90["logE"], PRD90["sigma"], PRD90["sigma_stat"],
                fmt='s', color='steelblue', ms=4, capsize=3,
                label="Auger PRD 90 (2014)", alpha=0.7)

    ax.errorbar(PRD111["logE"], PRD111["sigma"], PRD111["sigma_stat"],
                fmt='o', color='navy', ms=5, capsize=3,
                label="Auger PRD 111 (2025)")

    ax.errorbar(logE_c, sp, sp*0.1,   # error aprox del σ
                fmt='D', color='crimson', ms=5, capsize=3,
                label=f"Modelo ({label_config})")

    ax.errorbar(logE_c, st, st*0.1,
                fmt='^', color='darkorange', ms=5, capsize=3,
                label="Real (test set)", alpha=0.7)

    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$", fontsize=12)
    ax.set_ylabel(r"$\sigma(X_\mathrm{max})$ [g/cm²]", fontsize=12)
    ax.set_xlim(17.7, 20.2)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = out_dir / f"comparacion_papers_{label_config}.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figura guardada: {out_file}")

    # ── Métricas de acuerdo ─────────────────────────────────
    # Interpolar modelo en logE de PRD111
    if len(logE_c) >= 3:
        xp_interp = np.interp(PRD111["logE"], logE_c, xp)
        sp_interp = np.interp(PRD111["logE"], logE_c, sp)
        desv_xmax = np.abs(xp_interp - np.array(PRD111["xmax"]))
        desv_sigma = np.abs(sp_interp - np.array(PRD111["sigma"]))

        print(f"\n  Desviación media ⟨Xmax⟩  vs PRD111: "
              f"{desv_xmax.mean():.1f} g/cm²")
        print(f"  Desviación media σ(Xmax) vs PRD111: "
              f"{desv_sigma.mean():.1f} g/cm²")

    return logE_c, xp, sp, xt, st

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pred_dir = Path("data/results/predictions")
    fig_dir  = Path("data/results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Correr para el mejor config
    configs_a_comparar = [
        ("nq4_L3_angle_predictions.npz", "nq4_L3_angle"),
        ("nq6_L2_angle_predictions.npz", "nq6_L2_angle"),
    ]

    for fname, label in configs_a_comparar:
        pred_file = pred_dir / fname
        if pred_file.exists():
            print(f"\nProcesando: {label}")
            graficar_comparacion(pred_file, fig_dir, label)
        else:
            print(f"  [SKIP] No encontrado: {pred_file}")
