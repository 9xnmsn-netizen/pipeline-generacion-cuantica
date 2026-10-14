"""
src/analysis/__init__.py

Módulos de análisis post-entrenamiento:
  - learning_curve   : curva de aprendizaje vs N muestras
  - lr_search        : búsqueda de learning rate óptimo
  - compare_papers   : comparación con PRD 111 y PRD 90
  - diagnostics      : diagnóstico rápido de predicciones
"""

from . import learning_curve
from . import lr_search
from . import compare_papers
from . import diagnostics

__all__ = [
    "learning_curve",
    "lr_search",
    "compare_papers",
    "diagnostics",
]