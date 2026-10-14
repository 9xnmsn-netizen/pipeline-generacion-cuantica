#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Script maestro del pipeline QML de rayos cósmicos.
Coordina la ejecución de las fases 0 a 7 + análisis en orden o aislado.

Uso
---
  python run_pipeline.py                        # todas las fases
  python run_pipeline.py --phase 2              # solo fase 2
  python run_pipeline.py --from-phase 3         # fase 3 en adelante
  python run_pipeline.py --phase A1             # solo learning curve
  python run_pipeline.py --list-combos          # ver combinaciones
  python run_pipeline.py --list-phases          # ver fases disponibles
  python run_pipeline.py --dry-run              # simular sin ejecutar
  python run_pipeline.py --config otro.yaml     # config alternativo
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# ── Asegurar que la raíz del proyecto está en el path ────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_utils import (
    load_config,
    get_paths,
    get_combos,
    combo_name,
    ensure_dirs,
)
from src.utils.log_utils import get_logger


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE FASES
# ══════════════════════════════════════════════════════════════════════════════

PHASE_REGISTRY: dict[str, dict] = {
    # ── Fases principales ─────────────────────────────────────────────────
    "0": {
        "name":        "Preparación y validación del dataset",
        "module":      "src.phase0_prepare",
        "fn":          "run",
        "needs_combo": False,
    },
    "1": {
        "name":        "Reducción de dimensionalidad (PCA/UMAP)",
        "module":      "src.phase1_reduce",
        "fn":          "run",
        "needs_combo": False,
    },
    "2": {
        "name":        "Generación de features cuánticas",
        "module":      "src.phase2_quantum_features",
        "fn":          "run",
        "needs_combo": False,
    },
    "2b": {
        "name":        "Optimización variacional (COBYLA+MSE)",
        "module":      "src.phase2b_optimize",
        "fn":          "run",
        "needs_combo": False,
    },
    "3": {
        "name":        "Entrenamiento XGBoost (barrido)",
        "module":      "src.phase3_train",
        "fn":          "run",
        "needs_combo": False,
    },
    "3b": {
        "name":        "Optimización profunda XGBoost (mejor modelo)",
        "module":      "src.phase3b_optimize",
        "fn":          "run",
        "needs_combo": False,
    },
    "4": {
        "name":        "Predicción sobre datos nuevos",
        "module":      "src.phase4_predict",
        "fn":          "run",
        "needs_combo": False,
    },
    "5": {
        "name":        "Análisis y visualizaciones (Xmax + σ)",
        "module":      "src.phase5_analysis",
        "fn":          "run",
        "needs_combo": False,
    },
    "6": {
        "name":        "P3 — 4 features + ⟨Z⟩ cuánticas",
        "module":      "src.phase6_p3",
        "fn":          "run",
        "needs_combo": False,
    },
    "7": {
        "name":        "P4 — Optimización con N filas (COBYLA)",
        "module":      "src.phase7_p4",
        "fn":          "run",
        "needs_combo": False,
    },

    # ── Análisis post-entrenamiento (A1–A4) ───────────────────────────────
    "A1": {
        "name":        "Curva de aprendizaje vs N muestras",
        "module":      "src.analysis.learning_curve",
        "fn":          "run",
        "needs_combo": False,
    },
    "A2": {
        "name":        "Búsqueda de learning rate óptimo",
        "module":      "src.analysis.lr_search",
        "fn":          "run",
        "needs_combo": False,
    },
    "A3": {
        "name":        "Comparación con papers (PRD 111 / PRD 90)",
        "module":      "src.analysis.compare_papers",
        "fn":          "run",
        "needs_combo": False,
    },
    "A4": {
        "name":        "Diagnóstico rápido de predicciones",
        "module":      "src.analysis.diagnostics",
        "fn":          "run",
        "needs_combo": False,
    },
}


def _phase_sort_key(k: str) -> tuple:
    """
    Ordena claves de fase correctamente:
    "0" < "1" < "2" < "2b" < "3" < "3b" < "4" < "5" < "6" < "7" < "A1" < "A2"
    """
    k = str(k)
    # Separar prefijo alfabético (A) del numérico
    if k.startswith("A"):
        return (100, int(k[1:]))          # A1 → (100, 1)
    num_part = ""
    str_part = ""
    for ch in k:
        if ch.isdigit():
            num_part += ch
        else:
            str_part += ch
    return (int(num_part) if num_part else 0, str_part)


ALL_PHASES = sorted(PHASE_REGISTRY.keys(), key=_phase_sort_key)


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTOS CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = "Pipeline QML — Rayos Cósmicos",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = __doc__,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Ruta al archivo de configuración (default: config.yaml)",
    )
    parser.add_argument(
        "--phase", type=str, default=None,
        choices=ALL_PHASES, metavar="N",
        help=f"Fase a ejecutar: {', '.join(ALL_PHASES)}",
    )
    parser.add_argument(
        "--from-phase", type=str, default=None,
        dest="from_phase", metavar="N",
        help="Ejecutar desde la fase N hasta el final",
    )
    parser.add_argument(
        "--list-combos", action="store_true", dest="list_combos",
        help="Mostrar combinaciones disponibles y salir",
    )
    parser.add_argument(
        "--list-phases", action="store_true", dest="list_phases",
        help="Mostrar fases disponibles y salir",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Mostrar qué se ejecutaría sin ejecutar nada",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Activar logging DEBUG",
    )
    parser.add_argument("--nq",       type=int, default=None,
                        help="Número de qubits (sobrescribe config)")
    parser.add_argument("--layers",   type=int, default=None,
                        help="Número de capas L (sobrescribe config)")
    parser.add_argument("--encoding", type=str, default=None,
                        help="Tipo de encoding: angle | zz")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DINÁMICA DE FASES
# ══════════════════════════════════════════════════════════════════════════════

def load_phase_fn(phase_id: str):
    """Importa dinámicamente el módulo de la fase y devuelve su run()."""
    import importlib
    info   = PHASE_REGISTRY[phase_id]
    module = importlib.import_module(info["module"])
    return getattr(module, info["fn"])


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE UNA FASE
# ══════════════════════════════════════════════════════════════════════════════

def run_phase(
    phase_id: str,
    cfg:      dict,
    dry_run:  bool,
    log,
) -> bool:
    """
    Ejecuta una fase completa.
    Todas las fases reciben solo (cfg,) — cada una itera internamente.

    Returns True si OK, False si hubo error.
    """
    info = PHASE_REGISTRY[phase_id]
    log.info("=" * 70)
    log.info(f"  FASE {phase_id}: {info['name']}")
    log.info("=" * 70)

    if dry_run:
        log.info(f"  [DRY-RUN] Ejecutaría: {info['module']}.run(cfg)")
        return True

    t0 = time.perf_counter()
    try:
        fn = load_phase_fn(phase_id)
        fn(cfg)
        elapsed = time.perf_counter() - t0
        log.info(f"  ✅ Fase {phase_id} completada en {elapsed:.1f}s")
        return True
    except Exception as e:
        log.error(f"  ❌ Fase {phase_id} falló: {e}")
        log.debug(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _determine_phases(args: argparse.Namespace) -> list[str]:
    """Determina qué fases ejecutar según los argumentos."""
    if args.phase is not None:
        return [args.phase]
    if args.from_phase is not None:
        idx = ALL_PHASES.index(args.from_phase)
        return ALL_PHASES[idx:]
    return ALL_PHASES


def _print_combos(combos: list, log) -> None:
    log.info(f"Combinaciones disponibles ({len(combos)}):")
    for i, c in enumerate(combos, 1):
        nq, L, enc = c
        log.info(
            f" [{i:>2d}] {combo_name(nq, L, enc):<20s} "
            f"nq={nq}, L={L}, enc={enc}"
        )


def _print_phases(log) -> None:
    """Muestra el registro de fases agrupado por categoría."""
    log.info("─" * 70)
    log.info("FASES DISPONIBLES")
    log.info("─" * 70)

    # Fases principales
    log.info("\n  Pipeline principal:")
    for p in ALL_PHASES:
        if not p.startswith("A"):
            info = PHASE_REGISTRY[p]
            log.info(f"    {p:>3} | {info['name']}")

    # Análisis
    log.info("\n  Análisis post-entrenamiento:")
    for p in ALL_PHASES:
        if p.startswith("A"):
            info = PHASE_REGISTRY[p]
            log.info(f"    {p:>3} | {info['name']}")

    log.info("─" * 70)


def _print_plan(phases: list[str], log) -> None:
    log.info("─" * 70)
    log.info("PLAN DE EJECUCIÓN")
    log.info("─" * 70)
    for p in phases:
        info = PHASE_REGISTRY[p]
        log.info(f"  Fase {p:>3}: {info['name']}")
    log.info("─" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()

    # ── Cargar configuración ──────────────────────────────────────────────
    cfg   = load_config(args.config)
    paths = get_paths(cfg)

    # ── Configurar logging ────────────────────────────────────────────────
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file  = None
    if cfg.get("logging", {}).get("save_to_file", True):
        log_file = paths["logs_dir"] / "pipeline.log"

    log = get_logger("pipeline", log_file=log_file, level=log_level)

    # ── Crear directorios ─────────────────────────────────────────────────
    ensure_dirs(paths)

    # ── Modo: listar combos ───────────────────────────────────────────────
    if args.list_combos:
        combos = get_combos(cfg)
        _print_combos(combos, log)
        return 0

    # ── Modo: listar fases ────────────────────────────────────────────────
    if args.list_phases:
        _print_phases(log)
        return 0

    # ── Determinar y mostrar plan ─────────────────────────────────────────
    phases = _determine_phases(args)
    _print_plan(phases, log)

    if args.dry_run:
        log.info("  [DRY-RUN] No se ejecutará nada.")

    # ── Ejecutar fases ────────────────────────────────────────────────────
    t_start = time.perf_counter()
    results = {}
    all_ok  = True

    for phase_id in phases:
        ok = run_phase(
            phase_id = phase_id,
            cfg      = cfg,
            dry_run  = args.dry_run,
            log      = log,
        )
        results[phase_id] = ok
        if not ok:
            all_ok = False

    # ── Resumen final ─────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("RESUMEN FINAL")
    log.info("=" * 70)
    for p, ok in results.items():
        status = "✅ OK" if ok else "❌ ERROR"
        name   = PHASE_REGISTRY[p]["name"]
        log.info(f"  Fase {p:>3} | {name:<45s} | {status}")
    log.info(f"  Tiempo total: {t_total:.1f}s")
    log.info("=" * 70)

    return 0 if all_ok else 1


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(main())