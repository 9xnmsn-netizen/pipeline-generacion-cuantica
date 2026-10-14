#!/usr/bin/env python3
"""
Gráfica del segundo momento <Xmax> y sigma(Xmax) estilo Auger
para las 12 configuraciones del proyecto QML-Xmax
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
RUTA_NPZ = "/home2/DSMaster/evarela/Simon/datos_pca_cuanticos_backup/pred/predicciones/validacion/todo/" # ¿Para que era necesario ir a otro proyecto?
RUTA_SALIDA = "graficas_segundo_momento/" # ¿Con qué modelo salen generadas las gráficas resultantes? ¿En qué etapa de este pipeline, dichos modelos fueron entrenados? 
os.makedirs(RUTA_SALIDA, exist_ok=True)

# Bins de energía (log10 E/eV)
BINS_ENERGIA = np.arange(18.5, 20.1, 0.1)
BIN_CENTROS  = 0.5 * (BINS_ENERGIA[:-1] + BINS_ENERGIA[1:])

# ─────────────────────────────────────────────
# DATOS DE REFERENCIA AUGER (publicados)
# ─────────────────────────────────────────────
# <Xmax> Auger 2014 (aprox.) — g/cm²
auger_logE   = np.array([18.55, 18.65, 18.75, 18.85, 18.95,
                          19.05, 19.15, 19.25, 19.35, 19.55])
auger_xmax   = np.array([754.0, 757.0, 762.0, 766.0, 770.0,
                          774.0, 778.0, 782.0, 784.0, 786.0])
auger_xmax_err = np.array([3.0]*10)

# sigma(Xmax) Auger 2014 (aprox.) — g/cm²
auger_sigma  = np.array([26.0, 25.0, 24.5, 24.0, 23.0,
                          22.5, 21.5, 21.0, 20.5, 20.0])
auger_sigma_err = np.array([2.0]*10)

# ─────────────────────────────────────────────
# PREDICCIONES DE MODELOS HADRÓNICOS (constantes)
# ─────────────────────────────────────────────
def xmax_hadronico(logE, modelo="EPOS-LHC", composicion="p"):
    """Aproximación lineal de <Xmax> para modelos hadrónicos."""
    params = {
        ("EPOS-LHC", "p"):    (762.0, 56.0),
        ("EPOS-LHC", "Fe"):   (682.0, 56.0),
        ("QGSJet-II-04", "p"):  (758.0, 54.0),
        ("QGSJet-II-04", "Fe"): (676.0, 54.0),
        ("Sibyll2.3", "p"):   (755.0, 55.0),
        ("Sibyll2.3", "Fe"):  (673.0, 55.0),
    }
    X0, D = params.get((modelo, composicion), (760.0, 55.0))
    return X0 + D * (logE - 19.0)

# ─────────────────────────────────────────────
# CONFIGURACIONES A GRAFICAR
# ─────────────────────────────────────────────
CONFIGS = {
    "Original":                      ("tab:blue",   "-",  "o"),
    "Original_GPS_Quantum":          ("tab:cyan",   "--", "s"),
    "Original_Cuanticas":            ("tab:green",  "-",  "^"),
    "Original_Cuanticas_GPS_Quantum":("tab:olive",  "--", "D"),
    "Top30":                         ("tab:red",    "-",  "o"),
    "Top30_GPS_Quantum":             ("tab:orange", "--", "s"),
    "Top30_Cuanticas":               ("tab:pink",   "-",  "^"),
    "Top30_Cuanticas_GPS_Quantum":   ("tab:brown",  "--", "D"),
    "PCA":                           ("tab:purple", "-",  "o"),
    "PCA_GPS_Quantum":               ("tab:gray",   "--", "s"),
    "PCA_Cuanticas":                 ("navy",       "-",  "^"),
    "PCA_Cuanticas_GPS_Quantum":     ("darkgreen",  "--", "D"),
}

# ─────────────────────────────────────────────
# FUNCIÓN: calcular momentos por bins
# ─────────────────────────────────────────────
def calcular_momentos(logE, y_pred, bins):
    """Calcula <Xmax> y sigma(Xmax) en bins de energía."""
    centros, medias, sigmas, conteos = [], [], [], []
    for i in range(len(bins) - 1):
        mask = (logE >= bins[i]) & (logE < bins[i+1])
        n = mask.sum()
        if n >= 10:  # mínimo 10 eventos por bin
            vals = y_pred[mask]
            centros.append(0.5 * (bins[i] + bins[i+1]))
            medias.append(np.mean(vals))
            sigmas.append(np.std(vals))
            conteos.append(n)
    return (np.array(centros), np.array(medias),
            np.array(sigmas), np.array(conteos))

# ─────────────────────────────────────────────
# FUNCIÓN: graficar una configuración individual
# ─────────────────────────────────────────────
def graficar_config(nombre, color, ls, marker):
    ruta = os.path.join(RUTA_NPZ, f"{nombre}.npz")
    if not os.path.exists(ruta):
        print(f"  ⚠️  No encontrado: {ruta}")
        return None

    data   = np.load(ruta)
    logE   = data['logE_val']
    y_pred = data['y_pred']

    cE, mX, sX, nX = calcular_momentos(logE, y_pred, BINS_ENERGIA)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"Segundo Momento — {nombre.replace('_', ' ')}",
                 fontsize=14, fontweight='bold')

    # ── Panel superior: <Xmax> ──────────────────
    ax1 = axes[0]

    # Modelos hadrónicos
    logE_range = np.linspace(18.5, 19.8, 100)
    for modelo, comp, lc, ll in [
        ("EPOS-LHC",     "p",  "gold",      "EPOS-LHC p"),
        ("EPOS-LHC",     "Fe", "goldenrod", "EPOS-LHC Fe"),
        ("QGSJet-II-04", "p",  "tomato",    "QGSJet p"),
        ("QGSJet-II-04", "Fe", "firebrick", "QGSJet Fe"),
    ]:
        ax1.plot(logE_range,
                 xmax_hadronico(logE_range, modelo, comp),
                 color=lc, lw=1.2, ls=":", label=ll, alpha=0.7)

    # Referencia Auger
    ax1.errorbar(auger_logE, auger_xmax, yerr=auger_xmax_err,
                 fmt='k^', ms=6, capsize=3,
                 label="Auger 2014", zorder=5)

    # Predicciones del modelo
    ax1.plot(cE, mX, color=color, ls=ls, marker=marker,
             ms=7, lw=1.8, label=nombre.replace('_', ' '))

    ax1.set_ylabel(r"$\langle X_{\max} \rangle$ [g/cm²]", fontsize=12)
    ax1.set_ylim(640, 840)
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel inferior: sigma(Xmax) ─────────────
    ax2 = axes[1]

    ax2.errorbar(auger_logE, auger_sigma, yerr=auger_sigma_err,
                 fmt='k^', ms=6, capsize=3,
                 label="Auger 2014", zorder=5)

    ax2.plot(cE, sX, color=color, ls=ls, marker=marker,
             ms=7, lw=1.8, label=nombre.replace('_', ' '))

    ax2.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$", fontsize=12)
    ax2.set_ylabel(r"$\sigma(X_{\max})$ [g/cm²]", fontsize=12)
    ax2.set_ylim(0, 80)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta_fig = os.path.join(RUTA_SALIDA, f"segundo_momento_{nombre}.png")
    plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Guardado: {ruta_fig}")

    return {"config": nombre, "bins": cE.tolist(),
            "xmax_mean": mX.tolist(), "xmax_sigma": sX.tolist(),
            "n_eventos": nX.tolist()}

# ─────────────────────────────────────────────
# FUNCIÓN: gráfica comparativa TODAS las configs
# ─────────────────────────────────────────────
def graficar_comparativa(resultados):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Comparativa — Segundo Momento (todas las configuraciones)",
                 fontsize=13, fontweight='bold')

    ax1, ax2 = axes

    # Referencia Auger
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)

    ax1.errorbar(auger_logE, auger_xmax, yerr=auger_xmax_err,
                 fmt='k^', ms=7, capsize=3, lw=2,
                 label="Auger 2014", zorder=10)
    ax2.errorbar(auger_logE, auger_sigma, yerr=auger_sigma_err,
                 fmt='k^', ms=7, capsize=3, lw=2,
                 label="Auger 2014", zorder=10)

    # Modelos hadrónicos en panel superior
    logE_range = np.linspace(18.5, 19.8, 100)
    for modelo, comp, lc, ll in [
        ("EPOS-LHC",     "p",  "gold",      "EPOS-LHC p"),
        ("EPOS-LHC",     "Fe", "goldenrod", "EPOS-LHC Fe"),
    ]:
        ax1.plot(logE_range,
                 xmax_hadronico(logE_range, modelo, comp),
                 color=lc, lw=1.0, ls=":", label=ll, alpha=0.6)

    for res in resultados:
        nombre = res["config"]
        color, ls, marker = CONFIGS[nombre]
        cE = np.array(res["bins"])
        mX = np.array(res["xmax_mean"])
        sX = np.array(res["xmax_sigma"])
        label = nombre.replace('_', ' ')

        ax1.plot(cE, mX, color=color, ls=ls, marker=marker,
                 ms=5, lw=1.5, label=label, alpha=0.85)
        ax2.plot(cE, sX, color=color, ls=ls, marker=marker,
                 ms=5, lw=1.5, label=label, alpha=0.85)

    ax1.set_ylabel(r"$\langle X_{\max} \rangle$ [g/cm²]", fontsize=12)
    ax1.set_ylim(640, 840)
    ax1.legend(fontsize=6.5, ncol=3, loc="upper left")

    ax2.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$", fontsize=12)
    ax2.set_ylabel(r"$\sigma(X_{\max})$ [g/cm²]", fontsize=12)
    ax2.set_ylim(0, 80)
    ax2.legend(fontsize=6.5, ncol=3, loc="upper right")

    plt.tight_layout()
    ruta_fig = os.path.join(RUTA_SALIDA, "segundo_momento_COMPARATIVA.png")
    plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Comparativa guardada: {ruta_fig}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Generando gráficas de segundo momento...")
    print("=" * 55)

    resultados = []
    for nombre, (color, ls, marker) in CONFIGS.items():
        print(f"\n→ Procesando: {nombre}")
        res = graficar_config(nombre, color, ls, marker)
        if res is not None:
            resultados.append(res)

    print(f"\n→ Generando comparativa con {len(resultados)} configuraciones...")
    graficar_comparativa(resultados)

    print("\n" + "=" * 55)
    print(f"  ✅ Completado. Figuras en: {RUTA_SALIDA}")
    print(f"  📊 Configuraciones procesadas: {len(resultados)}/12")
    print("=" * 55)