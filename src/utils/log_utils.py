"""
log_utils.py
------------
Logger centralizado con formato consistente.

Uso:
    from src.utils.log_utils import get_logger
    log = get_logger("phase1")
    log.info("Mensaje")
"""

import logging
import sys
from pathlib import Path


def get_logger(
    name:      str,
    log_file:  str | Path | None = None,
    level:     int               = logging.INFO,
) -> logging.Logger:
    """
    Crea o recupera un logger con formato estándar.

    Parameters
    ----------
    name     : nombre del logger (usualmente el nombre de la fase)
    log_file : si se indica, también escribe en ese archivo
    level    : nivel de logging (default: INFO)

    Returns
    -------
    logging.Logger configurado
    """
    logger = logging.getLogger(name)

    # Evitar handlers duplicados si se llama varias veces
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt      = "[%(asctime)s] [%(name)-12s] %(levelname)-8s %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
    )

    # ── Handler consola ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Handler archivo (opcional) ────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger