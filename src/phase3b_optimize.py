"""
phase3b_optimize.py
-------------------
Fase 3b: Optimización profunda de hiperparámetros XGBoost.

Migrado de: optimizar_xgboost_pca_sin_repetidos_datos_unicos.py

Estado: ESQUELETO — pendiente de migración completa.
Por ahora delega a phase3_train si no está implementado.

Uso:
    from src import phase3b_optimize
    phase3b_optimize.run(cfg)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_utils import load_config, get_paths
from src.utils.log_utils    import get_logger

log = get_logger("phase3b_optimize")


def run(cfg: dict) -> None:
    """
    Fase 3b: búsqueda profunda de hiperparámetros.

    TODO: migrar lógica de optimizar_xgboost_pca_sin_repetidos_datos_unicos.py
    """
    log.warning("=" * 60)
    log.warning("FASE 3b: AÚN NO IMPLEMENTADA")
    log.warning("=" * 60)
    log.warning("Pendiente de migración.")
    log.warning("Mientras tanto, usa --phase 3 (barrido base).")
    # No lanzar excepción — permite que el pipeline continúe


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)