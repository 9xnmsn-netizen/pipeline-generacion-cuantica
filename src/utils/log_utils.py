"""
log_utils.py
------------
Logger centralizado con formato consistente.

Uso:
    from src.utils.log_utils import get_logger
    log = get_logger("phase1")
    log.info("Mensaje")
"""

import logging # ¿Qué es logging en Python? Hacer un ejemplo de logging con debug, info, warning, error y critical.
import sys # ¿Qué es sys en Python? Hacer un ejemplo de sys.argv y sys.exit().
from pathlib import Path # ¿Para qué se usa pathlib en Python? Hacer un ejemplo de pathlib.Path y pathlib.Path.mkdir().

# Una función que crea o recupera un logger con formato estándar, con opción de escribir en un archivo. 
# ¿Qué es un logger en Python? Hacer un ejemplo de logging con diferentes niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL.
def get_logger(
    name:      str,
    log_file:  str | Path | None = None,
    level:     int               = logging.INFO, # ¿Qué es logging.INFO en Python? Hacer un ejemplo de logging con diferentes niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL.
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
    logger = logging.getLogger(name) # ¿Qué hace logging.getLogger() en Python? Hacer un ejemplo de logging.getLogger() y logging.basicConfig().

    # Evitar handlers duplicados si se llama varias veces
    if logger.handlers: # ¿Para qué sirve logger.handlers en Python? Hacer un ejemplo de logger.handlers y logger.addHandler().
        return logger

    logger.setLevel(level) # .setLevel() establece el nivel de severidad del logger. Los niveles son: DEBUG, INFO, WARNING, ERROR, CRITICAL.

    fmt = logging.Formatter( # ¿Por qué se usa logging.Formatter() en Python? Hacer un ejemplo de logging.Formatter() y logging.StreamHandler().
        fmt      = "[%(asctime)s] [%(name)-12s] %(levelname)-8s %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
    )

    # ── Handler consola ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout) # ¿Qué hace logging.StreamHandler() en Python? Hacer un ejemplo de logging.StreamHandler() y logging.FileHandler().
    ch.setLevel(level) 
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ¿Qué es Handler en logging de Python? Hacer un ejemplo de logging con StreamHandler y FileHandler.
    # ── Handler archivo (opcional) ────────────────────────────────────────────
    if log_file is not None: 
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger