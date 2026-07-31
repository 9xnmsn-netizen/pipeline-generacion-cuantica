#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Script maestro del pipeline QML de rayos cósmicos.
Coordina la ejecución de las fases 0 a 5 en orden o de forma aislada.

Uso
---
  python run_pipeline.py                        # todas las fases
  python run_pipeline.py --phase 2              # solo fase 2
  python run_pipeline.py --from-phase 3         # fase 3 en adelante
  python run_pipeline.py --list-combos          # ver combinaciones
  python run_pipeline.py --dry-run              # simular sin ejecutar
  python run_pipeline.py --config otro.yaml     # config alternativo
"""

import argparse
from html import parser
from html import parser
import sys
import time
import traceback
from pathlib import Path

# ── Asegurar que la raíz del proyecto está en el path ────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# run_pipeline.py  (línea ~29)
from src.utils.config_utils import (
    load_config,
    get_paths,
    get_combos,
    combo_name,
    ensure_dirs,
)
from src.utils.log_utils    import get_logger

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE FASES
# ══════════════════════════════════════════════════════════════════════════════
PHASE_REGISTRY: dict[int, dict] = {
    "0": {
        "name"       : "Preparación y validación",
        "module"     : "src.phase0_prepare",           # ✅ corregido
        "fn"         : "run",
        "needs_combo": False,
    },
    "1": {
        "name"       : "Reducción de dimensionalidad",
        "module"     : "src.phase1_reduce",            # ✅ corregido
        "fn"         : "run",
        "needs_combo": False,
    },
    "2": {
        "name"       : "Generación de features cuánticas",
        "module"     : "src.phase2_quantum_features",  # ✅ corregido (nombre real)
        "fn"         : "run",
        "needs_combo": False,   # ✅ phase2 itera internamente sobre combos
    },
    "2b": {
    "name"       : "Optimización variacional (COBYLA+MSE)",
    "module"     : "src.phase2b_optimize",
    "fn"         : "run",
    "needs_combo": False,
    },
    "3": {
        "name"       : "Entrenamiento XGBoost",
        "module"     : "src.phase3_train",             # ✅ corregido
        "fn"         : "run",
        "needs_combo": False,   # ✅ phase3 itera internamente sobre archivos
    },
    "4": {
        "name"       : "Predicción sobre datos nuevos",
        "module"     : "src.phase4_predict",           # ✅ corregido
        "fn"         : "run",
        "needs_combo": False,   # ✅ phase4 itera internamente sobre modelos
    },
    "5": {
        "name"       : "Análisis y visualizaciones",
        "module"     : "src.phase5_analysis",          # ✅ corregido
        "fn"         : "run",
        "needs_combo": False,   # ✅ phase5 itera internamente
    },
}

def _phase_sort_key(k: str):
    """
    Ordena claves de fase correctamente:
    "0" < "1" < "2" < "2b" < "3" < "4" < "5"
    """
    k = str(k)
    # Separar parte numérica de parte alfabética
    # "2b" → (2, "b") | "3" → (3, "") | "0" → (0, "")
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
        description      = "Pipeline QML — Rayos Cósmicos",
        formatter_class  = argparse.RawDescriptionHelpFormatter,
        epilog           = __doc__,
    )
    parser.add_argument(
        "--config",
        type    = str,
        default = "config.yaml",
        help    = "Ruta al archivo de configuración (default: config.yaml)",
    )
    parser.add_argument(
        "--phase",
        type    = str,
        default = None,
        choices = ALL_PHASES,
        metavar = "N",
        help    = "Fase a ejecutar: 0,1,2,2b,3,4,5",
    )
    parser.add_argument(
        "--from-phase",
        type    = int,
        default = None,
        dest    = "from_phase",
        choices = ALL_PHASES,
        metavar = "N",
        help    = "Ejecutar desde la fase N hasta el final",
    )
    parser.add_argument(
        "--list-combos",
        action = "store_true",
        dest   = "list_combos",
        help   = "Mostrar combinaciones disponibles y salir",
    )
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        dest   = "dry_run",
        help   = "Mostrar qué se ejecutaría sin ejecutar nada",
    )
    parser.add_argument(
        "--verbose",
        action = "store_true",
        help   = "Activar logging DEBUG",
    )
    parser.add_argument("--nq",       type=int,  default=None,
                    help="Número de qubits (sobreescribe config.yaml)")
    parser.add_argument("--layers",   type=int,  default=None,
                    help="Número de capas L (sobreescribe config.yaml)")
    parser.add_argument("--encoding", type=str,  default=None,
                    help="Tipo de encoding: angle | zz")
    
    return parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE FASE (lazy import)
# ══════════════════════════════════════════════════════════════════════════════
def load_phase_fn(phase_id: int):
    """Importa dinámicamente el módulo de la fase y devuelve su run()."""
    import importlib
    info   = PHASE_REGISTRY[phase_id]
    module = importlib.import_module(info["module"])
    return getattr(module, info["fn"])

# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE UNA FASE
# ══════════════════════════════════════════════════════════════════════════════
def run_phase(
    phase_id : int,
    cfg      : dict,
    dry_run  : bool,
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
def _determine_phases(args: argparse.Namespace) -> list[int]:
    if args.phase is not None:
        return [args.phase]
    if args.from_phase is not None:
        return [p for p in ALL_PHASES if p >= args.from_phase]
    return ALL_PHASES

def _print_combos(combos: list[dict], log) -> None:
    log.info(f"Combinaciones disponibles ({len(combos)}):")
    for i, c in enumerate(combos, 1):
        nq, L, enc = c
        log.info(
            f" [{i:>2d}] {combo_name(nq, L, enc):<20s} "
            f"nq={nq}, L={L}, enc={enc}"
        )

def _print_plan(phases: list[int], log) -> None:
    log.info("─" * 70)
    log.info("PLAN DE EJECUCIÓN")
    log.info("─" * 70)
    for p in phases:
        info = PHASE_REGISTRY[p]
        log.info(f"  Fase {p}: {info['name']}")
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
        log.info(f"  Fase {p}: {status}")
    log.info(f"  Tiempo total: {t_total:.1f}s")
    log.info("=" * 70)

    return 0 if all_ok else 1

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(main())