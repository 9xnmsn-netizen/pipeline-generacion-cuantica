"""
diagnostics.py
--------------
Diagnóstico rápido del estado de predicciones.

Migrado de: diagnostico_predicciones.py

Responde: ¿qué rango tiene xmax_pred? ¿hay correlación con logE?

Uso:
    from src.analysis import diagnostics
    diagnostics.run(cfg)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.io_utils     import load_npz, ensure_dirs
from src.utils.log_utils    import get_logger

log = get_logger("diagnostics")


def run(cfg: dict, config_name: str = "4q_3l_angle") -> None:
    """
    Diagnóstico de predicciones.

    Parameters
    ----------
    cfg         : configuración global
    config_name : nombre de la combinación
    """
    log.info("=" * 60)
    log.info("ANÁLISIS — Diagnóstico de predicciones")
    log.info("=" * 60)

    paths   = get_paths(cfg)
    res_dir = paths["results_dir"] / config_name
    fig_dir = Path("data/results/figures")
    ensure_dirs(fig_dir)

    pred_file = res_dir / "predictions.npz"
    if not pred_file.exists():
        log.error(f"No encontrado: {pred_file}")
        return

    data = load_npz(pred_file)
    y_pred = data["y_pred_test"]
    y_true = data["y_true_test"]
    logE   = data["logE_test"]

    # ── Estadísticas ──────────────────────────────────────────────────────
    log.info(f"y_pred rango:  [{y_pred.min():.1f}, {y_pred.max():.1f}] g/cm²")
    log.info(f"y_true rango:  [{y_true.min():.1f}, {y_true.max():.1f}] g/cm²")
    log.info(f"y_pred media:  {y_pred.mean():.1f} ± {y_pred.std():.1f} g/cm²")
    log.info(f"y_true media:  {y_true.mean():.1f} ± {y_true.std():.1f} g/cm²")
    log.info(f"logE rango:    [{logE.min():.2f}, {logE.max():.2f}]")
    log.info(f"Corr(logE, y_pred): {np.corrcoef(logE, y_pred)[0,1]:.4f}")
    log.info(f"Corr(logE, y_true): {np.corrcoef(logE, y_true)[0,1]:.4f}")

    # ── Figura ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: pred vs true
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=5)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--')
    axes[0].set_xlabel("Xmax real")
    axes[0].set_ylabel("Xmax predicho")
    axes[0].set_title("Predicción vs Real")

    # Panel 2: Xmax vs logE
    axes[1].scatter(logE, y_pred, alpha=0.3, s=5, label="pred")
    axes[1].scatter(logE, y_true, alpha=0.1, s=5, label="real", color='orange')
    axes[1].set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
    axes[1].set_ylabel("Xmax [g/cm²]")
    axes[1].set_title("Xmax vs logE")
    axes[1].legend()

    # Panel 3: residuos
    residuos = y_pred - y_true
    axes[2].hist(residuos, bins=50, edgecolor='black')
    axes[2].axvline(0, color='r', linestyle='--')
    axes[2].set_xlabel("Residuo [g/cm²]")
    axes[2].set_title(f"Residuos: μ={residuos.mean():.1f}, σ={residuos.std():.1f}")

    fig.tight_layout()
    out_fig = fig_dir / f"diagnostico_{config_name}.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    log.info(f"Figura: {out_fig}")

    log.info("\nANÁLISIS completado ✓")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)