"""
phase5_analysis.py
------------------
Fase 5: Análisis y visualizaciones.

Genera:
  1. Gráfica <Xmax> vs log10(E)          → figures/xmax_mean_{combo}.png
  2. Gráfica σ(Xmax) vs log10(E)         → figures/xmax_sigma_{combo}.png
  3. Heatmap R² por (nq, L, encoding)    → figures/heatmap_r2.png
  4. Comparación clásico vs híbrido      → figures/comparison_r2.png
  5. Resumen global                      → results/summary.csv

Contrato de entrada (por combinación):
  data/results/{nq}q_{L}l_{enc}/
      predictions_new.npz   ← y_pred_new, logE_new
      metrics.json          ← r2_train, r2_val, r2_test, mae_test, rmse_test

Contrato de salida:
  data/figures/
      xmax_mean_{combo}.png
      xmax_sigma_{combo}.png
      heatmap_r2_{enc}.png
      comparison_r2.png
  data/results/
      summary.csv
"""

import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.log_utils    import get_logger

log = get_logger("phase5")

# ── Estilo global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 12,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "figure.dpi"       : 150,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
})

# ── Modelos hadrónicos de referencia (constantes de literatura) ───────────────
# Valores de <Xmax> [g/cm²] y σ(Xmax) [g/cm²] por bin de log10(E)
# Fuente: Auger / compilaciones de literatura
HADRONIC_MODELS = {
    "EPOS-LHC": {
        "color"    : "royalblue",
        "ls"       : "--",
        "xmax_mean": lambda logE: 750.0 + 58.0 * (logE - 18.5),
        "xmax_sig" : lambda logE: 22.0  - 1.5  * (logE - 18.5),
    },
    "Sibyll2.3d": {
        "color"    : "tomato",
        "ls"       : "-.",
        "xmax_mean": lambda logE: 745.0 + 56.0 * (logE - 18.5),
        "xmax_sig" : lambda logE: 24.0  - 1.8  * (logE - 18.5),
    },
    "QGSJet-II": {
        "color"    : "seagreen",
        "ls"       : ":",
        "xmax_mean": lambda logE: 740.0 + 55.0 * (logE - 18.5),
        "xmax_sig" : lambda logE: 26.0  - 2.0  * (logE - 18.5),
    },
}

# Bins de energía estándar Auger
ENERGY_BINS = np.arange(18.5, 20.05, 0.1)   # bordes
BIN_CENTERS = 0.5 * (ENERGY_BINS[:-1] + ENERGY_BINS[1:])


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE BINS
# ══════════════════════════════════════════════════════════════════════════════

def bin_statistics(
    logE:    np.ndarray,
    xmax:    np.ndarray,
    bins:    np.ndarray,
    min_n:   int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula <Xmax> y σ(Xmax) en bins de log10(E).

    Parameters
    ----------
    logE   : array log10(E) por evento
    xmax   : array Xmax por evento
    bins   : bordes de los bins
    min_n  : mínimo de eventos por bin para incluirlo

    Returns
    -------
    centers : centros de bins con suficientes eventos
    means   : <Xmax> por bin
    sigmas  : σ(Xmax) por bin
    counts  : N eventos por bin
    """
    centers, means, sigmas, counts = [], [], [], []

    for i in range(len(bins) - 1):
        mask = (logE >= bins[i]) & (logE < bins[i + 1])
        n    = mask.sum()
        if n < min_n:
            continue
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        means.append(xmax[mask].mean())
        sigmas.append(xmax[mask].std(ddof=1))
        counts.append(n)

    return (
        np.array(centers),
        np.array(means),
        np.array(sigmas),
        np.array(counts),
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — <Xmax> vs log10(E)
# ══════════════════════════════════════════════════════════════════════════════

def plot_xmax_mean(
    logE_pred:  np.ndarray,
    xmax_pred:  np.ndarray,
    logE_real:  np.ndarray | None,
    xmax_real:  np.ndarray | None,
    combo_name: str,
    out_path:   Path,
) -> None:
    """
    Gráfica de <Xmax> vs log10(E).

    Incluye:
      - Predicciones del modelo híbrido (puntos con barras de error)
      - Datos reales del FD (si se proporcionan)
      - Curvas de modelos hadrónicos
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # ── Modelos hadrónicos ────────────────────────────────────────────────────
    logE_ref = np.linspace(18.5, 20.0, 200)
    for name, props in HADRONIC_MODELS.items():
        ax.plot(
            logE_ref,
            props["xmax_mean"](logE_ref),
            color = props["color"],
            ls    = props["ls"],
            lw    = 1.5,
            label = name,
            alpha = 0.8,
        )

    # ── Datos reales (FD) ─────────────────────────────────────────────────────
    if logE_real is not None and xmax_real is not None:
        c_r, m_r, s_r, _ = bin_statistics(logE_real, xmax_real, ENERGY_BINS)
        ax.errorbar(
            c_r, m_r,
            yerr      = s_r / np.sqrt(np.maximum(1, _)),
            fmt       = "ks",
            ms        = 6,
            capsize   = 4,
            label     = "Datos FD reales",
            zorder    = 5,
        )

    # ── Predicciones del modelo ───────────────────────────────────────────────
    c_p, m_p, s_p, n_p = bin_statistics(logE_pred, xmax_pred, ENERGY_BINS)
    ax.errorbar(
        c_p, m_p,
        yerr      = s_p / np.sqrt(np.maximum(1, n_p)),
        fmt       = "o",
        color     = "darkorange",
        ms        = 7,
        capsize   = 4,
        label     = f"Modelo híbrido ({combo_name})",
        zorder    = 6,
    )

    # ── Formato ───────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel(r"$\langle X_{\max} \rangle$ [g/cm²]")
    ax.set_title(
        r"$\langle X_{\max} \rangle$ vs $\log_{10}(E)$"
        f"\n{combo_name}"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Guardada: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — σ(Xmax) vs log10(E)  [SEGUNDO MOMENTO]
# ══════════════════════════════════════════════════════════════════════════════

def plot_xmax_sigma(
    logE_pred:  np.ndarray,
    xmax_pred:  np.ndarray,
    logE_real:  np.ndarray | None,
    xmax_real:  np.ndarray | None,
    combo_name: str,
    out_path:   Path,
) -> None:
    """
    Gráfica del segundo momento σ(Xmax) vs log10(E).

    El segundo momento refleja la dispersión de la composición
    de rayos cósmicos: mezclas de núcleos → σ grande;
    composición pura → σ pequeño.

    Incluye:
      - σ(Xmax) del modelo híbrido
      - σ(Xmax) de datos reales FD (si se proporcionan)
      - Bandas de modelos hadrónicos (protón puro / hierro puro)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    logE_ref = np.linspace(18.5, 20.0, 200)

    # ── Modelos hadrónicos ────────────────────────────────────────────────────
    for name, props in HADRONIC_MODELS.items():
        ax.plot(
            logE_ref,
            props["xmax_sig"](logE_ref),
            color = props["color"],
            ls    = props["ls"],
            lw    = 1.5,
            label = name,
            alpha = 0.8,
        )

    # ── Datos reales (FD) ─────────────────────────────────────────────────────
    if logE_real is not None and xmax_real is not None:
        c_r, _, s_r, n_r = bin_statistics(logE_real, xmax_real, ENERGY_BINS)
        # Error en σ: Var(S) ≈ σ²/(2(n-1))  →  ΔS ≈ σ/√(2(n-1))
        err_r = s_r / np.sqrt(2 * np.maximum(1, n_r - 1))
        ax.errorbar(
            c_r, s_r,
            yerr      = err_r,
            fmt       = "ks",
            ms        = 6,
            capsize   = 4,
            label     = "Datos FD reales",
            zorder    = 5,
        )

    # ── Predicciones del modelo ───────────────────────────────────────────────
    c_p, _, s_p, n_p = bin_statistics(logE_pred, xmax_pred, ENERGY_BINS)
    err_p = s_p / np.sqrt(2 * np.maximum(1, n_p - 1))
    ax.errorbar(
        c_p, s_p,
        yerr      = err_p,
        fmt       = "o",
        color     = "darkorange",
        ms        = 7,
        capsize   = 4,
        label     = f"Modelo híbrido ({combo_name})",
        zorder    = 6,
    )

    # ── Formato ───────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel(r"$\sigma(X_{\max})$ [g/cm²]")
    ax.set_title(
        r"Segundo momento: $\sigma(X_{\max})$ vs $\log_{10}(E)$"
        f"\n{combo_name}"
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Guardada: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Heatmap R² por (nq, L)
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap_r2(
    records:  list[dict],
    enc:      str,
    out_path: Path,
) -> None:
    """
    Heatmap de R²_test en función de (nq, L) para un encoding dado.

    Parameters
    ----------
    records  : lista de dicts con keys nq, L, enc, r2_test
    enc      : encoding a filtrar ('angle' o 'zz')
    out_path : ruta de salida
    """
    sub = [r for r in records if r["enc"] == enc]
    if not sub:
        log.warning(f"  No hay registros para encoding='{enc}' → saltando heatmap.")
        return

    df   = pd.DataFrame(sub)
    nqs  = sorted(df["nq"].unique())
    Ls   = sorted(df["L"].unique())

    # Construir matriz
    matrix = np.full((len(Ls), len(nqs)), np.nan)
    for _, row in df.iterrows():
        i = Ls.index(row["L"])
        j = nqs.index(row["nq"])
        matrix[i, j] = row["r2_test"]

    fig, ax = plt.subplots(figsize=(max(5, len(nqs) + 1), max(4, len(Ls) + 1)))

    im = ax.imshow(
        matrix,
        cmap   = "RdYlGn",
        aspect = "auto",
        vmin   = max(0, np.nanmin(matrix) - 0.02),
        vmax   = min(1, np.nanmax(matrix) + 0.02),
    )

    # Anotaciones
    for i in range(len(Ls)):
        for j in range(len(nqs)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i,
                    f"{val:.3f}",
                    ha         = "center",
                    va         = "center",
                    fontsize   = 10,
                    fontweight = "bold",
                    color      = "black" if 0.3 < val < 0.85 else "white",
                )

    ax.set_xticks(range(len(nqs)))
    ax.set_xticklabels([f"nq={q}" for q in nqs])
    ax.set_yticks(range(len(Ls)))
    ax.set_yticklabels([f"L={l}" for l in Ls])
    ax.set_title(f"R² (test) — encoding: {enc}")
    ax.set_xlabel("Número de qubits (nq)")
    ax.set_ylabel("Capas del circuito (L)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("R²")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Guardada: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Comparación clásico vs híbrido
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(
    records:        list[dict],
    baseline_r2:    float,
    out_path:       Path,
) -> None:
    """
    Gráfica de barras comparando R²_test de todas las combinaciones
    vs el baseline clásico.

    Parameters
    ----------
    records     : lista de dicts con keys combo_name, r2_test
    baseline_r2 : R² del modelo clásico de referencia
    out_path    : ruta de salida
    """
    if not records:
        log.warning("  No hay registros para comparación → saltando.")
        return

    df = pd.DataFrame(records).sort_values("r2_test", ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6 + 2), 5))

    colors = [
        "steelblue" if r < baseline_r2 else "darkorange"
        for r in df["r2_test"]
    ]

    bars = ax.bar(
        range(len(df)),
        df["r2_test"],
        color  = colors,
        edgecolor = "black",
        linewidth = 0.6,
    )

    # Línea baseline
    ax.axhline(
        baseline_r2,
        color     = "crimson",
        ls        = "--",
        lw        = 2,
        label     = f"Baseline clásico R²={baseline_r2:.4f}",
        zorder    = 5,
    )

    # Etiquetas en barras
    for bar, val in zip(bars, df["r2_test"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha       = "center",
            va       = "bottom",
            fontsize = 8,
            rotation = 45,
        )

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["combo_name"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R² (test)")
    ax.set_title("Comparación: Clásico vs Modelos Híbridos Cuánticos")
    ax.legend()
    ax.set_ylim(
        max(0, df["r2_test"].min() - 0.05),
        min(1, df["r2_test"].max() + 0.05),
    )

    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="darkorange", edgecolor="black",
              label="Supera baseline"),
        Patch(facecolor="steelblue",  edgecolor="black",
              label="Por debajo de baseline"),
    ]
    ax.legend(handles=legend_elems + ax.get_legend_handles_labels()[0],
              loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Guardada: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS REALES (FD)
# ══════════════════════════════════════════════════════════════════════════════

def load_real_fd_data(cfg: dict, paths: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Carga los datos reales del FD para comparación.

    Busca en data/processed/fd_real.npz (generado en Fase 0).
    Si no existe, devuelve (None, None).

    Returns
    -------
    logE_real : np.ndarray o None
    xmax_real : np.ndarray o None
    """
    proc_dir  = paths["processed_dir"]
    real_path = proc_dir / "fd_real.npz"

    if not real_path.exists():
        log.warning(
            f"  No se encontró {real_path} → "
            "gráficas sin datos reales FD."
        )
        return None, None

    data      = np.load(str(real_path))
    logE_real = data["logE"]
    xmax_real = data["xmax"]
    log.info(
        f"  Datos FD reales cargados: {len(xmax_real)} eventos"
    )
    return logE_real, xmax_real

# ══════════════════════════════════════════════════════════════════════════════
# PENDIENTE 5 — Métricas por bins de energía
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics_by_bin(
    logE    : np.ndarray,
    y_true  : np.ndarray,
    y_pred  : np.ndarray,
    bins    : np.ndarray,
    min_n   : int = 5,
) -> list[dict]:
    """
    Calcula R², MAE, RMSE, bias y N por bin de log10(E).

    Parámetros
    ----------
    logE   : array de log10(E/eV) por evento
    y_true : array de Xmax real por evento
    y_pred : array de Xmax predicho por evento
    bins   : bordes de los bins (ej: np.arange(18.5, 20.1, 0.1))
    min_n  : mínimo de eventos por bin para incluirlo

    Devuelve
    --------
    lista de dicts con: bin_center, n, r2, mae, rmse, bias, std_residual
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    results = []

    for i in range(len(bins) - 1):
        mask = (logE >= bins[i]) & (logE < bins[i + 1])
        n = int(mask.sum())

        if n < min_n:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        # Métricas estándar
        r2   = float(r2_score(yt, yp))        if n > 1 else np.nan
        mae  = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        bias = float(np.mean(yp - yt))          # positivo = sobreestima
        std_res = float(np.std(yp - yt, ddof=1)) if n > 1 else np.nan

        results.append({
            "bin_lo"      : float(bins[i]),
            "bin_hi"      : float(bins[i + 1]),
            "bin_center"  : float(0.5 * (bins[i] + bins[i + 1])),
            "n"           : n,
            "r2"          : r2,
            "mae"         : mae,
            "rmse"        : rmse,
            "bias"        : bias,
            "std_residual": std_res,
        })

    return results


def plot_metrics_by_bin(
    bin_metrics : list[dict],
    combo_name  : str,
    out_dir     : Path,
) -> None:
    """
    Genera 3 gráficas de métricas por bin de energía:
      1. R² por bin
      2. MAE y RMSE por bin
      3. Bias (sesgo) por bin
    """
    if not bin_metrics:
        log.warning(f"  {combo_name}: sin métricas por bin → saltando gráfica")
        return

    centers = np.array([b["bin_center"]   for b in bin_metrics])
    n_vals  = np.array([b["n"]            for b in bin_metrics])
    r2_vals = np.array([b["r2"]           for b in bin_metrics])
    mae_vals= np.array([b["mae"]          for b in bin_metrics])
    rmse_v  = np.array([b["rmse"]         for b in bin_metrics])
    bias_v  = np.array([b["bias"]         for b in bin_metrics])

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(
        f"Métricas por bin de energía\n{combo_name}",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: R² por bin ───────────────────────────────────────────────
    ax = axes[0]
    colors = ["tomato" if r < 0 else "steelblue" for r in r2_vals]
    bars = ax.bar(centers, r2_vals, width=0.08, color=colors,
                  edgecolor="black", linewidth=0.5, align="center")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("R²")
    ax.set_title("R² por bin de energía")

    # Anotar N eventos en cada barra
    for bar, n in zip(bars, n_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"N={n}",
            ha="center", va="bottom", fontsize=7, rotation=45
        )

    # ── Panel 2: MAE y RMSE por bin ───────────────────────────────────────
    ax = axes[1]
    ax.plot(centers, mae_vals,  "o-", color="darkorange",
            lw=2, ms=6, label="MAE [g/cm²]")
    ax.plot(centers, rmse_v,    "s--", color="royalblue",
            lw=2, ms=6, label="RMSE [g/cm²]")
    ax.set_ylabel("Error [g/cm²]")
    ax.set_title("MAE y RMSE por bin de energía")
    ax.legend(loc="upper right")

    # ── Panel 3: Bias por bin ─────────────────────────────────────────────
    ax = axes[2]
    ax.bar(centers, bias_v, width=0.08,
           color=["tomato" if b > 0 else "seagreen" for b in bias_v],
           edgecolor="black", linewidth=0.5, align="center")
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel("Sesgo [g/cm²]")
    ax.set_title("Sesgo (bias = ŷ − y) por bin de energía")

    # Texto explicativo
    ax.text(
        0.02, 0.95,
        "Rojo = sobreestima Xmax\nVerde = subestima Xmax",
        transform=ax.transAxes,
        va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    for ax in axes:
        ax.set_xlim(18.45, 20.05)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    out_path = out_dir / f"metrics_by_bin_{combo_name}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Guardada: {out_path.name}")


def save_metrics_by_bin_csv(
    all_bin_metrics : dict[str, list[dict]],
    results_dir     : Path,
) -> None:
    """
    Guarda todas las métricas por bin en un CSV único.

    Columnas: combo_name, bin_center, bin_lo, bin_hi, n, r2, mae, rmse, bias
    """
    rows = []
    for combo_name, bin_list in all_bin_metrics.items():
        for b in bin_list:
            rows.append({"combo_name": combo_name, **b})

    if not rows:
        log.warning("  No hay métricas por bin para guardar.")
        return

    df = pd.DataFrame(rows)
    out_path = results_dir / "metrics_by_bin.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    log.info(f"  CSV de métricas por bin guardado: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    """
    Ejecuta la Fase 5 completa.
    """
    log.info("=" * 60)
    log.info("FASE 5 — Análisis y visualizaciones")
    log.info("=" * 60)

    paths       = get_paths(cfg)
    results_dir = paths["results_dir"]
    figures_dir = paths["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    p5           = cfg.get("phase5", {})
    baseline_r2  = p5.get("baseline_r2", 0.0)

    # ── Cargar datos reales FD ────────────────────────────────────────────────
    logE_real, xmax_real = load_real_fd_data(cfg, paths)

    # ── Descubrir combinaciones ───────────────────────────────────────────────
    pattern     = str(results_dir / "*q_*l_*/metrics.json")
    metric_paths = sorted(Path(p) for p in glob.glob(pattern))

    if not metric_paths:
        log.error(
            f"No se encontraron metrics.json en {results_dir}. "
            "Ejecuta Fase 3 primero."
        )
        return

    log.info(f"\nCombinaciones encontradas: {len(metric_paths)}")

    records = []   # para heatmap y comparación
    # Guardar CSV global de métricas por bin
    all_bin_metrics = {}   # ← inicializar ANTES del bucle for
    # ... (el bucle lo llena)

    for mp in metric_paths:
        combo_name = mp.parent.name          # "4q_3l_angle"
        parts      = combo_name.split("_")
        nq         = int(parts[0].replace("q", ""))
        L          = int(parts[1].replace("l", ""))
        enc        = parts[2]

        # ── Dentro del bucle for mp in metric_paths: ──────────────────────────────

        with open(mp) as f:
            metrics = json.load(f)

        # ✅ Claves corregidas: phase3 guarda "{split}_{metric}"
        records.append({
            "combo_name" : combo_name,
            "nq"         : nq,
            "L"          : L,
            "enc"        : enc,
            "r2_train"   : metrics.get("train_r2",  np.nan),   # ✅ era "r2_train"
            "r2_val"     : metrics.get("val_r2",    np.nan),   # ✅ era "r2_val"
            "r2_test"    : metrics.get("test_r2",   np.nan),   # ✅ era "r2_test"
            "mae_test"   : metrics.get("test_mae",  np.nan),   # ✅ era "mae_test"
            "rmse_test"  : metrics.get("test_rmse", np.nan),   # ✅ era "rmse_test"
        })

        # ── Cargar predicciones nuevas ────────────────────────────────────────
        # ✅ REEMPLAZAR POR
        # ── Buscar archivo de predicciones (fase 4) ───────────────────────────
        pred_path_new  = mp.parent / f"predictions_{combo_name}.npz"
        pred_path_test = mp.parent / "predictions.npz"   # generado en fase 3

        if pred_path_new.exists():
            pred_data  = np.load(str(pred_path_new), allow_pickle=True)
            xmax_pred  = pred_data["xmax_pred"]
            logE_pred  = pred_data.get("logE_new", None)
            log.info(f" {combo_name}: usando predicciones nuevas (fase 4)")
        elif pred_path_test.exists():
            pred_data  = np.load(str(pred_path_test), allow_pickle=True)
            xmax_pred  = pred_data["y_pred_test"]
            logE_pred  = pred_data.get("logE_test", None)
            log.info(f" {combo_name}: usando predicciones test (fase 3)")
        else:
            log.warning(
                f" {combo_name}: no hay archivo de predicciones → "
                f"saltando gráficas Xmax.\n"
                f"  Buscados: {pred_path_new.name} | {pred_path_test.name}"
            )
            continue

        if logE_pred is None:
            log.warning(f" {combo_name}: no hay logE → saltando gráficas Xmax.")
            continue

        log.info(f"\n  Generando gráficas para: {combo_name}")

        # ── Figura 1: <Xmax> ──────────────────────────────────────────────────
        plot_xmax_mean(
            logE_pred  = logE_pred,
            xmax_pred  = xmax_pred,
            logE_real  = logE_real,
            xmax_real  = xmax_real,
            combo_name = combo_name,
            out_path   = figures_dir / f"xmax_mean_{combo_name}.png",
        )

        # ── Figura 2: σ(Xmax) ─────────────────────────────────────────────────
        plot_xmax_sigma(
            logE_pred  = logE_pred,
            xmax_pred  = xmax_pred,
            logE_real  = logE_real,
            xmax_real  = xmax_real,
            combo_name = combo_name,
            out_path   = figures_dir / f"xmax_sigma_{combo_name}.png",
        )

        # ── AÑADIR dentro del bucle for mp in metric_paths: ──────────────────────
        # (después de generar las gráficas xmax_mean y xmax_sigma)

        # ── Pendiente 5: Métricas por bin de energía ─────────────────────────────
        if logE_pred is not None and pred_path_test.exists():
            # Cargar y_true del conjunto test
            pred_test_data = np.load(str(pred_path_test), allow_pickle=True)
            y_true_test    = pred_test_data.get("y_true_test", None)
            y_pred_test    = pred_test_data.get("y_pred_test", None)
            logE_test      = pred_test_data.get("logE_test",   None)

        if y_true_test is not None and logE_test is not None:
            # Convertir logE_test a log10 si está en eV
            if logE_test.max() > 100:          # está en eV, no en log10
                logE_test_log10 = np.where(
                    logE_test > 0,
                    np.log10(logE_test.astype(np.float64)).astype(np.float32),
                    np.nan
                )
            else:
                logE_test_log10 = logE_test    # ya está en log10

            bin_metrics = compute_metrics_by_bin(
                logE    = logE_test_log10,
                y_true  = y_true_test,
                y_pred  = y_pred_test,
                bins    = ENERGY_BINS,
                min_n   = int(p5.get("min_events_per_bin", 5)),
            )

            # Gráfica por combinación
            plot_metrics_by_bin(
                bin_metrics = bin_metrics,
                combo_name  = combo_name,
                out_dir     = figures_dir,
            )

            # Acumular para CSV global
            all_bin_metrics[combo_name] = bin_metrics

            # Log rápido de resumen
            if bin_metrics:
                r2_vals = [b["r2"] for b in bin_metrics if not np.isnan(b["r2"])]
                log.info(
                    f"  Métricas por bin: {len(bin_metrics)} bins | "
                    f"R²_min={min(r2_vals):.3f} | R²_max={max(r2_vals):.3f}"
                )


    # ── AÑADIR antes de "Figura 3: Heatmaps" ─────────────────────────────────
    # (fuera del bucle for)
    save_metrics_by_bin_csv(all_bin_metrics, results_dir)

    # ── Figura 3: Heatmaps por encoding ──────────────────────────────────────
    log.info("\n  Generando heatmaps R²...")
    encodings = sorted({r["enc"] for r in records})
    for enc in encodings:
        plot_heatmap_r2(
            records  = records,
            enc      = enc,
            out_path = figures_dir / f"heatmap_r2_{enc}.png",
        )

    # ── Figura 4: Comparación global ──────────────────────────────────────────
    log.info("\n  Generando comparación clásico vs híbrido...")
    plot_comparison(
        records     = records,
        baseline_r2 = baseline_r2,
        out_path    = figures_dir / "comparison_r2.png",
    )

    # ── Resumen CSV ───────────────────────────────────────────────────────────
    summary_path = results_dir / "summary.csv"
    df_summary   = pd.DataFrame(records).sort_values("r2_test", ascending=False)
    df_summary.to_csv(summary_path, index=False)
    log.info(f"\n  Resumen guardado: {summary_path}")

    # ── Log del top-3 ─────────────────────────────────────────────────────────
    log.info("\n  Top-3 modelos por R² (test):")
    for _, row in df_summary.head(3).iterrows():
        log.info(
            f"    {row['combo_name']:20s} | "
            f"R²_test={row['r2_test']:.4f} | "
            f"MAE={row['mae_test']:.2f} | "
            f"RMSE={row['rmse_test']:.2f}"
        )

    log.info(f"\nFASE 5 completada ✓")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)