"""
compare_papers.py
-----------------
Comparación de predicciones del modelo con PRD 111 y PRD 90.

Migrado de: comparar_con_papers.py

Uso:
    from src.analysis import compare_papers
    compare_papers.run(cfg)
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.io_utils     import load_npz, save_json, ensure_dirs
from src.utils.log_utils    import get_logger
from src.utils.plot_utils   import bin_statistics, load_references

log = get_logger("compare_papers")


def run(cfg: dict, config_name: str = "4q_3l_angle") -> None:
    """
    Compara predicciones del modelo con PRD 111 y PRD 90.

    Parameters
    ----------
    cfg         : configuración global
    config_name : nombre de la combinación
    """
    log.info("=" * 60)
    log.info("ANÁLISIS — Comparación con papers")
    log.info("=" * 60)

    paths    = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    res_dir  = paths["results_dir"] / config_name
    fig_dir  = Path("data/results/figures")
    ensure_dirs(fig_dir)

    # ── Cargar predicciones ───────────────────────────────────────────────
    pred_file = res_dir / "predictions.npz"
    if not pred_file.exists():
        log.error(f"No encontrado: {pred_file}")
        return

    data = load_npz(pred_file)
    logE   = data["logE_test"]
    y_pred = data["y_pred_test"]
    y_true = data["y_true_test"]

    # ── Referencias ───────────────────────────────────────────────────────
    refs = load_references(cfg, names=["PRD_111", "PRD_90"])

    # ── Bins ──────────────────────────────────────────────────────────────
    bins = np.arange(17.8, 20.2, 0.1)
    logE_c, xp, sp, n_p = bin_statistics(logE, y_pred, bins, min_n=10)
    _,      xt, st, _   = bin_statistics(logE, y_true, bins, min_n=10)

    # ── Figuras ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Comparación con Auger — {config_name}", fontsize=13)

    # Panel 1: ⟨Xmax⟩
    ax = axes[0]

    for name, ref in refs.items():
        marker = 'o' if name == "PRD_111" else 's'
        color  = 'navy' if name == "PRD_111" else 'steelblue'
        ax.errorbar(ref["logE"], ref["xmax"], ref["xmax_stat"],
                    fmt=marker, color=color, ms=5, capsize=3,
                    label=f"Auger {name}", alpha=0.7)

    ax.errorbar(logE_c, xp, sp / np.sqrt(np.maximum(1, n_p)),
                fmt='D', color='crimson', ms=5, capsize=3,
                label=f"Modelo ({config_name})")
    ax.errorbar(logE_c, xt, st / np.sqrt(np.maximum(1, n_p)),
                fmt='^', color='darkorange', ms=5, capsize=3,
                label="Real (test set)", alpha=0.7)

    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel(r"$\langle X_{\max} \rangle$ [g/cm²]")
    ax.set_xlim(17.7, 20.2)
    ax.set_ylim(680, 820)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: σ(Xmax)
    ax = axes[1]

    for name, ref in refs.items():
        if "sigma" not in ref:
            continue
        marker = 'o' if name == "PRD_111" else 's'
        color  = 'navy' if name == "PRD_111" else 'steelblue'
        ax.errorbar(ref["logE"], ref["sigma"], ref["sigma_stat"],
                    fmt=marker, color=color, ms=5, capsize=3,
                    label=f"Auger {name}", alpha=0.7)

    ax.errorbar(logE_c, sp, sp * 0.1,
                fmt='D', color='crimson', ms=5, capsize=3,
                label=f"Modelo ({config_name})")
    ax.errorbar(logE_c, st, st * 0.1,
                fmt='^', color='darkorange', ms=5, capsize=3,
                label="Real (test set)", alpha=0.7)

    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel(r"$\sigma(X_{\max})$ [g/cm²]")
    ax.set_xlim(17.7, 20.2)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig = fig_dir / f"comparacion_papers_{config_name}.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura: {out_fig}")

    # ── Métricas de acuerdo ───────────────────────────────────────────────
    if len(logE_c) >= 3 and "PRD_111" in refs:
        ref = refs["PRD_111"]
        xp_interp = np.interp(ref["logE"], logE_c, xp)
        sp_interp = np.interp(ref["logE"], logE_c, sp)

        desv_xmax  = np.abs(xp_interp - ref["xmax"])
        desv_sigma = np.abs(sp_interp - ref["sigma"])

        log.info(f"Desviación media ⟨Xmax⟩ vs PRD111: {desv_xmax.mean():.1f} g/cm²")
        log.info(f"Desviación media σ(Xmax) vs PRD111: {desv_sigma.mean():.1f} g/cm²")

        save_json(fig_dir / f"desviaciones_{config_name}.json", {
            "desv_xmax_mean":  float(desv_xmax.mean()),
            "desv_sigma_mean": float(desv_sigma.mean()),
        })

    log.info("\nANÁLISIS completado ✓")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)