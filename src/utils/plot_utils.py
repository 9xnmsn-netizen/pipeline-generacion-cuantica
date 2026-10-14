"""
plot_utils.py
-------------
Funciones comunes de graficación y cálculo de momentos para el pipeline.

Extrae las funciones repetidas en los scripts originales:
  - calcular_puntos_sem      (de metricas_por_bines_*.py)
  - bin_statistics           (de phase5_analysis.py)
  - RailsFunc                (de graficar_comparacion_Xmax.py)
  - load_references          (carga datos Auger desde config.yaml)
  - load_hadronic_models     (carga modelos hadrónicos desde config.yaml)
  - setup_matplotlib         (estilo global consistente)
  - graficar_con_referencias (gráfica unificada estilo Auger)

Uso:
    from src.utils.plot_utils import (
        calcular_puntos_sem,
        bin_statistics,
        RailsFunc,
        load_references,
        load_hadronic_models,
        setup_matplotlib,
        graficar_con_referencias,
    )
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Any

from src.utils.log_utils import get_logger

log = get_logger("plot_utils")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL DE MATPLOTLIB
# ══════════════════════════════════════════════════════════════════════════════

def setup_matplotlib(cfg: dict) -> None:
    """
    Aplica estilo global de matplotlib desde config.yaml → plot_styles.matplotlib.

    Parameters
    ----------
    cfg : dict
        Configuración global cargada con load_config().
    """
    style = cfg.get("plot_styles", {}).get("matplotlib", {})

    plt.rcParams.update({
        "font.family"      : style.get("font_family", "serif"),
        "font.size"        : style.get("font_size", 12),
        "axes.titlesize"   : style.get("font_size", 12) + 1,
        "axes.labelsize"   : style.get("font_size", 12),
        "legend.fontsize"  : style.get("font_size", 12) - 2,
        "figure.dpi"       : style.get("figure_dpi", 150),
        "axes.grid"        : style.get("axes_grid", True),
        "grid.alpha"       : style.get("grid_alpha", 0.3),
        "axes.linewidth"   : style.get("axes_linewidth", 1.2),
        "xtick.direction"  : style.get("tick_direction", "in"),
        "ytick.direction"  : style.get("tick_direction", "in"),
    })

    log.debug("Estilo matplotlib aplicado")


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE MOMENTOS POR BINS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_puntos_sem(
    y_pred:        np.ndarray,
    logE:          np.ndarray,
    energy_points: np.ndarray,
    ancho_bin:     float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binea predicciones y calcula media ± SEM por punto de energía.

    Originalmente en: metricas_por_bines_v2/v3/v4.py

    Parameters
    ----------
    y_pred        : predicciones (N,)
    logE          : log10(E/eV) por evento (N,)
    energy_points : centros de bins donde evaluar (M,)
    ancho_bin     : semi-ancho del bin (±ancho_bin)

    Returns
    -------
    xmax_bin : media de y_pred en cada bin (M,)
    sem_bin  : error estándar de la media en cada bin (M,)
    """
    xmax_list, sem_list = [], []

    for ep in energy_points:
        mask = (logE >= ep - ancho_bin) & (logE < ep + ancho_bin)
        if mask.sum() > 0:
            xmax_list.append(np.mean(y_pred[mask]))
            sem_list.append(np.std(y_pred[mask]) / np.sqrt(mask.sum()))
        else:
            xmax_list.append(np.nan)
            sem_list.append(np.nan)

    return np.array(xmax_list), np.array(sem_list)


def bin_statistics(
    logE:  np.ndarray,
    xmax:  np.ndarray,
    bins:  np.ndarray,
    min_n: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula ⟨Xmax⟩, σ(Xmax), N por bins de log10(E).

    Originalmente en: phase5_analysis.py

    Parameters
    ----------
    logE  : log10(E/eV) por evento (N,)
    xmax  : Xmax por evento (N,)
    bins  : bordes de los bins (M+1,)
    min_n : mínimo de eventos por bin

    Returns
    -------
    centers : centros de bins con suficientes eventos
    means   : ⟨Xmax⟩ por bin
    sigmas  : σ(Xmax) por bin (ddof=1)
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
# MODELOS HADRÓNICOS
# ══════════════════════════════════════════════════════════════════════════════

def RailsFunc(lgE: np.ndarray, p: list | np.ndarray) -> np.ndarray:
    """
    Función de Rails para modelos hadrónicos.

    Originalmente en: graficar_comparacion_Xmax.py

    Xmax(lgE) = a + b·(lgE − 18) + c·(lgE − 18)²

    Parameters
    ----------
    lgE : log10(E/eV) — escalar o array
    p   : [a, b, c] — parámetros del modelo

    Returns
    -------
    Xmax : array con misma forma que lgE
    """
    p = np.asarray(p)
    return p[0] + p[1] * (lgE - 18.0) + p[2] * (lgE - 18.0) ** 2


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DESDE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_references(
    cfg:     dict,
    names:   list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Carga datos de referencia Auger desde cfg['references'].

    Parameters
    ----------
    cfg   : dict — configuración global
    names : lista de nombres a cargar (None → todos)
            Opciones: 'PRD_111', 'PRD_114', 'PRD_90'

    Returns
    -------
    dict con estructura:
        {
            'PRD_111': {'logE': ..., 'xmax': ..., 'xmax_stat': ..., ...},
            'PRD_114': {...},
        }
    """
    refs_all = cfg.get("references", {})
    if not refs_all:
        log.warning("cfg['references'] está vacío o no existe")
        return {}

    if names is None:
        names = [k for k in refs_all.keys() if k.startswith("PRD")]

    refs = {}
    for name in names:
        if name not in refs_all:
            log.warning(f"Referencia '{name}' no encontrada en config")
            continue

        raw = refs_all[name]
        refs[name] = {
            k: np.asarray(v, dtype=float) if isinstance(v, list) else v
            for k, v in raw.items()
        }

    # ── Aplicar filtro a PRD_114 (solo logE >= 18.0) ───────────────────────
    if "PRD_114" in refs and "logE_full" in refs["PRD_114"]:
        ref = refs["PRD_114"]
        mask = ref["logE_full"] >= 18.0
        refs["PRD_114"] = {
            "logE":      ref["logE_full"][mask],
            "xmax":      ref["xmax_full"][mask],
            "xmax_stat": ref["err_full"][mask],
        }
        log.debug(
            f"PRD_114 filtrado: {mask.sum()}/{len(mask)} bins con logE ≥ 18.0"
        )

    log.debug(f"Referencias cargadas: {list(refs.keys())}")
    return refs


def load_hadronic_models(cfg: dict) -> dict[str, dict]:
    """
    Carga parámetros de modelos hadrónicos desde cfg['hadronic_models'].

    Parameters
    ----------
    cfg : dict — configuración global

    Returns
    -------
    dict con estructura:
        {
            'EPOS-LHC': {
                'H':  {'a': 748.78, 'b': 57.73, 'c': -0.85},
                'Fe': {'a': 648.60, 'b': 63.12, 'c': -1.97},
                'color': 'red', 'linestyle': '-'
            },
            ...
        }
    """
    models_raw = cfg.get("hadronic_models", {})
    if not models_raw:
        log.warning("cfg['hadronic_models'] está vacío o no existe")
        return {}

    # Excluir claves auxiliares que no son modelos
    aux_keys = {"Fe_color", "Fe_linestyle"}
    models = {
        k: v for k, v in models_raw.items()
        if k not in aux_keys and isinstance(v, dict)
    }

    log.debug(f"Modelos hadrónicos cargados: {list(models.keys())}")
    return models


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA UNIFICADA CON REFERENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def graficar_con_referencias(
    logE:            np.ndarray,
    xmax:            np.ndarray,
    refs:            dict[str, dict[str, np.ndarray]],
    models:          dict[str, dict],
    title:           str  = "Xmax vs log10(E)",
    out_path:        Path | str | None = None,
    label_model:     str  = "Modelo híbrido",
    show_sigma:      bool = False,
    offset:          float = 0.0,
    fine_bins:       tuple[float, float, float] | None = None,
) -> plt.Figure:
    """
    Gráfica unificada de Xmax (y opcionalmente σ(Xmax)) vs log10(E)
    con referencias Auger y modelos hadrónicos.

    Reemplaza a:
      - graficar_comparacion_Xmax.py
      - graficar_configuraciones_separadas.py
      - predecir_datos_reales_baseO.py (sección de gráficas)
      - metricas_por_bines_v4.py (parte de gráficas)

    Parameters
    ----------
    logE         : log10(E/eV) de las predicciones
    xmax         : predicciones del modelo
    refs         : salida de load_references()
    models       : salida de load_hadronic_models()
    title        : título de la figura
    out_path     : si se indica, guarda la figura
    label_model  : etiqueta para el modelo
    show_sigma   : si True, panel doble (⟨Xmax⟩, σ)
    offset       : corrección empírica a las predicciones (g/cm²)
    fine_bins    : (start, stop, step) — bins para promediar predicciones

    Returns
    -------
    matplotlib.Figure
    """
    # ── Bins por defecto ──────────────────────────────────────────────────
    if fine_bins is None:
        fine_bins = (18.5, 20.05, 0.1)

    bins = np.arange(fine_bins[0], fine_bins[1] + 1e-9, fine_bins[2])

    # ── Aplicar offset ────────────────────────────────────────────────────
    xmax_adj = xmax + offset

    # ── Estadísticas del modelo ───────────────────────────────────────────
    c_p, m_p, s_p, n_p = bin_statistics(logE, xmax_adj, bins)

    # ── Crear figura ──────────────────────────────────────────────────────
    if show_sigma:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax2 = None

    # ════════════════════════════════════════════════════════════════════════
    # Panel 1: ⟨Xmax⟩
    # ════════════════════════════════════════════════════════════════════════
    logE_grid = np.linspace(fine_bins[0], fine_bins[1], 200)

    # Modelos hadrónicos (H y Fe)
    for name, m in models.items():
        p_H = m.get("H", {})
        p_Fe = m.get("Fe", {})
        color = m.get("color", "red")
        ls = m.get("linestyle", "-")

        if p_H:
            ax1.plot(
                logE_grid,
                RailsFunc(logE_grid, [p_H["a"], p_H["b"], p_H["c"]]),
                color=color, ls=ls, lw=1.5,
                label=f"{name} (p)",
            )
        if p_Fe:
            ax1.plot(
                logE_grid,
                RailsFunc(logE_grid, [p_Fe["a"], p_Fe["b"], p_Fe["c"]]),
                color="blue", ls=ls, lw=1.5,
                label="_nolegend_",
            )

    # Referencias Auger
    for name, ref in refs.items():
        if "logE" not in ref:
            continue
        style = cfg_style_for_ref(name)
        yerr = ref.get("xmax_stat", None)
        ax1.errorbar(
            ref["logE"], ref["xmax"],
            yerr=yerr,
            fmt=style["marker"], color=style["color"],
            ecolor=style["color"], elinewidth=1.5, capsize=3,
            markersize=8, alpha=style.get("alpha", 1.0),
            label=name, zorder=style.get("zorder", 5),
        )

    # Predicciones del modelo
    if len(c_p) > 0:
        err_p = s_p / np.sqrt(np.maximum(1, n_p))
        ax1.errorbar(
            c_p, m_p,
            yerr=err_p,
            fmt='^', color='black', ecolor='black',
            elinewidth=2, capsize=4, capthick=1.5,
            markersize=9, zorder=10,
            label=label_model,
        )

    # Formato
    ax1.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$", fontsize=13)
    ax1.set_ylabel(r"$\langle X_{\max} \rangle$ [g/cm²]", fontsize=13)
    ax1.set_xlim(fine_bins[0] - 0.5, fine_bins[1] + 0.15)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Anotaciones p / Fe
    ax1.text(19.5, 810, "p", fontsize=14, color="red", style="italic")
    ax1.text(19.5, 680, "Fe", fontsize=14, color="blue", style="italic")

    # ════════════════════════════════════════════════════════════════════════
    # Panel 2: σ(Xmax)  (opcional)
    # ════════════════════════════════════════════════════════════════════════
    if show_sigma and ax2 is not None:
        # Referencias Auger
        for name, ref in refs.items():
            if "sigma" not in ref:
                continue
            style = cfg_style_for_ref(name)
            yerr = ref.get("sigma_stat", None)
            ax2.errorbar(
                ref["logE"], ref["sigma"],
                yerr=yerr,
                fmt=style["marker"], color=style["color"],
                ecolor=style["color"], elinewidth=1.5, capsize=3,
                markersize=8, alpha=style.get("alpha", 1.0),
                label=name, zorder=style.get("zorder", 5),
            )

        # Predicciones del modelo
        if len(c_p) > 0:
            err_s = s_p / np.sqrt(2 * np.maximum(1, n_p - 1))
            ax2.errorbar(
                c_p, s_p,
                yerr=err_s,
                fmt='^', color='black', ecolor='black',
                elinewidth=2, capsize=4, capthick=1.5,
                markersize=9, zorder=10,
                label=label_model,
            )

        ax2.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$", fontsize=13)
        ax2.set_ylabel(r"$\sigma(X_{\max})$ [g/cm²]", fontsize=13)
        ax2.set_xlim(fine_bins[0] - 0.5, fine_bins[1] + 0.15)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ════════════════════════════════════════════════════════════════════════
    # Guardar
    # ════════════════════════════════════════════════════════════════════════
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info(f"Figura guardada: {out_path}")

    return fig


def cfg_style_for_ref(name: str) -> dict:
    """
    Estilos por defecto para referencias Auger.
    (Puede sobreescribirse desde config si se desea.)
    """
    defaults = {
        "PRD_111": {"marker": 'D', "color": 'cyan',    "alpha": 0.7, "zorder": 5},
        "PRD_114": {"marker": 'd', "color": 'purple',  "alpha": 0.7, "zorder": 4},
        "PRD_90":  {"marker": 's', "color": 'steelblue', "alpha": 0.7, "zorder": 3},
    }
    return defaults.get(name, {"marker": 'o', "color": 'gray', "alpha": 0.7, "zorder": 3})


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE BINS
# ══════════════════════════════════════════════════════════════════════════════

def make_fine_bins(cfg: dict) -> np.ndarray:
    """
    Construye bins finos desde cfg['fine_bins'].

    Returns
    -------
    bins : np.ndarray con bordes de bins
    """
    fb = cfg.get("fine_bins", {})
    start = float(fb.get("start", 18.5))
    stop  = float(fb.get("stop",  20.05))
    step  = float(fb.get("step",  0.1))
    return np.arange(start, stop + 1e-9, step)


def get_energy_bins_cfg(cfg: dict) -> dict:
    """
    Devuelve la sección energy_bins del config.

    Returns
    -------
    dict con claves 'bajo', 'medio', 'alto', 'todo'
    """
    return cfg.get("energy_bins", {})


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "setup_matplotlib",
    "calcular_puntos_sem",
    "bin_statistics",
    "RailsFunc",
    "load_references",
    "load_hadronic_models",
    "graficar_con_referencias",
    "make_fine_bins",
    "get_energy_bins_cfg",
]