"""
quantum_utils.py
----------------
Construcción del circuito cuántico y generación de features.
Esta es la implementación ÚNICA del circuito — phase2 y phase4 la usan.

Arquitectura del circuito:
  [Encoding] → [Capa Variacional × L] → [⟨Z_i⟩ por qubit]

Contrato de entrada:
  X     : np.ndarray shape (n_samples, n_features)
  nq    : int  — número de qubits
  L     : int  — número de capas variacionales
  theta : np.ndarray shape (L, nq, 2)
           dim 2: [ángulo RZ, ángulo RY] por qubit por capa

Contrato de salida:
  features : np.ndarray shape (n_samples, nq)
             valores de expectación ⟨Z_i⟩ ∈ [-1, 1]
"""

import numpy as np
import pennylane as qml
from typing import Callable
from src.utils.log_utils import get_logger

log = get_logger("quantum_utils") # Ver de nuevo get_logger() en log_utils.py para entender qué hace y cómo se usa. Hacer un ejemplo de logging con diferentes niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL.

# ══════════════════════════════════════════════════════════════════════════════
# ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def angle_encoding(x: np.ndarray, nq: int) -> None:
    """
    Codificación por ángulo: RY(x_i mod n_features) en cada qubit i.

    Parameters
    ----------
    x  : array 1D de longitud n_features (ya normalizado en [-π, π]) # ¿Por qué usar 1D array? Hacer un ejemplo de np.ndarray 1D y 2D.
    nq : número de qubits
    """
    n = len(x)
    for i in range(nq):
        qml.RY(float(x[i % n]), wires=i) # ¿Qué hace % en Python? Hacer un ejemplo de operador módulo % con enteros y flotantes.


def zz_encoding(x: np.ndarray, nq: int) -> None:
    """
    Codificación ZZ Feature Map (Havlíček et al. 2019).

    Capa 1 : H en todos los qubits + RZ(2·x_i)
    Capa 2 : CNOT(i, i+1) + RZ(2·(π-x_i)·(π-x_{i+1})) + CNOT(i, i+1) # ¿De dónde vienen las fórmulas de codificación ZZ? Investigar el paper de Havlíček et al. 2019 y hacer un resumen.

    Parameters
    ----------
    x  : array 1D de longitud n_features
    nq : número de qubits
    """
    n = len(x)
    # Capa 1
    for i in range(nq):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * float(x[i % n]), wires=i)
    # Capa 2 — interacciones ZZ entre pares adyacentes
    for i in range(nq - 1):
        xi = float(x[i % n]) # ¿Qué son xi y xj? Hacer un ejemplo de cómo acceder a elementos de un array 1D en Python usando índices y el operador módulo %.
        xj = float(x[(i + 1) % n])
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2.0 * (np.pi - xi) * (np.pi - xj), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])


# ══════════════════════════════════════════════════════════════════════════════
# CAPA VARIACIONAL
# ══════════════════════════════════════════════════════════════════════════════

def variational_layer(theta: np.ndarray, nq: int, layer: int) -> None:
    """
    Capa variacional: RZ(θ₀) + RY(θ₁) en cada qubit + CNOT en anillo.

    Parameters
    ----------
    theta : array shape (L, nq, 2) # ¿Por qué tiene forma theta (L, nq, 2)? Hacer un ejemplo de np.ndarray con forma (L, nq, 2) y cómo acceder a sus elementos.
            theta[layer, i, 0] → ángulo RZ del qubit i
            theta[layer, i, 1] → ángulo RY del qubit i
    nq    : número de qubits
    layer : índice de capa (0-indexed)
    """
    for i in range(nq):
        qml.RZ(float(theta[layer, i, 0]), wires=i)
        qml.RY(float(theta[layer, i, 1]), wires=i)
    # Entrelazamiento en anillo
    for i in range(nq):
        qml.CNOT(wires=[i, (i + 1) % nq])


# ══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DE PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

def init_theta(L: int, nq: int, seed: int = 42) -> np.ndarray:
    """
    Inicializa parámetros variacionales θ ~ U(0, 2π). $ ¿Qué es U(0, 2π)? Hacer un ejemplo de distribución uniforme U(a, b) en Python con numpy.random.uniform().

    Returns
    -------
    theta : np.ndarray shape (L, nq, 2) float32
            dim 2: [RZ_angle, RY_angle]
    """
    rng   = np.random.default_rng(seed) # ¿Qué hace np.random.default_rng()? Hacer un ejemplo de cómo usar np.random.default_rng() para generar números aleatorios.
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(L, nq, 2)) # ¿Qué es .uniform() en numpy? Hacer un ejemplo de numpy.random.default_rng().uniform() para generar números aleatorios uniformes.
    return theta.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL CIRCUITO
# ══════════════════════════════════════════════════════════════════════════════

def build_circuit(nq: int, L: int, enc: str) -> Callable:
    """
    Construye y devuelve un QNode de PennyLane.

    Arquitectura:
      [Encoding] → [Variational Layer 1] → ... → [Variational Layer L]
      → Medición: ⟨Z_i⟩ en cada qubit

    Parameters
    ----------
    nq  : número de qubits
    L   : número de capas variacionales
    enc : 'angle' | 'zz'

    Returns
    -------
    circuit : Callable(x, theta) → list[float] de longitud nq
              x     : array 1D shape (n_features,)
              theta : array shape (L, nq, 2)

    Raises
    ------
    ValueError si enc no es 'angle' ni 'zz'
    """
    if enc not in ("angle", "zz"):
        raise ValueError(
            f"encoding='{enc}' no válido. Opciones: 'angle', 'zz'."
        )

    _enc_fn = angle_encoding if enc == "angle" else zz_encoding # ¿Qué forma de programación es esta? Hacer un ejemplo de asignación condicional en Python con if-else.
    # ¿Qué forma tiene _enc_fn? Hacer un ejemplo de función como objeto en Python y cómo llamarla.
    dev     = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, interface="autograd")
    def circuit(x: np.ndarray, theta: np.ndarray) -> list:
        _enc_fn(x, nq)
        for l in range(L):
            variational_layer(theta, nq, l)
        return [qml.expval(qml.PauliZ(i)) for i in range(nq)] # ¿Qué hace qml.expval() en PennyLane? Hacer un ejemplo de qml.expval() y qml.PauliZ() para medir la expectativa de Z en un qubit.

    log.info(
        f"Circuito construido: nq={nq}, L={L}, enc={enc} | "
        f"parámetros θ: {L * nq * 2}"
    )
    return circuit


# ══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def get_qfeatures(
    X         : np.ndarray,
    circuit   : Callable,
    theta     : np.ndarray,
    batch_size: int  = 64,
    verbose   : bool = True, # ¿Qué hace verbose en Python? Hacer un ejemplo de función con argumento verbose y cómo usarlo para mostrar información opcionalmente.
) -> np.ndarray:
    """
    Aplica el circuito cuántico a todas las muestras de X.

    Parameters
    ----------
    X          : array shape (n_samples, n_features)
    circuit    : QNode devuelto por build_circuit()
    theta      : parámetros variacionales shape (L, nq, 2)
    batch_size : cada cuántas muestras se imprime progreso
    verbose    : si True, muestra progreso

    Returns
    -------
    features : np.ndarray shape (n_samples, nq) float32
               valores ⟨Z_i⟩ ∈ [-1, 1] por qubit y muestra

    Notas
    -----
    - X debe estar normalizado en [-π, π] antes de llamar esta función
    - theta es FIJO (no se optimiza aquí)
    - nq se infiere de theta.shape[1]
    """
    n_samples = X.shape[0]
    nq        = theta.shape[1]
    features  = np.zeros((n_samples, nq), dtype=np.float32) # No estoy seguro si necesito np.zeros ¿Qué hace np.zeros() en numpy? Hacer un ejemplo de np.zeros() para crear un array de ceros con forma específica.
    # ¿Qué hace np.zeros() en numpy? Hacer un ejemplo de np.zeros() para crear un array de ceros con forma específica.

    for i in range(n_samples):
        result      = circuit(X[i], theta)
        features[i] = np.array(result, dtype=np.float32)

        if verbose and (i + 1) % batch_size == 0: # ¿Qué significa la condición (i + 1) % batch_size == 0? Hacer un ejemplo de operador módulo % en Python para determinar si un número es múltiplo de otro.
            pct = 100.0 * (i + 1) / n_samples
            log.info(
                f"  Progreso: {i+1:>6d}/{n_samples} "
                f"({pct:.1f}%)"
            )

    if verbose:
        log.info(
            f"  Features cuánticas: shape={features.shape} | "
            f"rango=[{features.min():.3f}, {features.max():.3f}]"
        )
    return features


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def circuit_summary(nq: int, L: int, enc: str) -> dict:
    """
    Resumen del circuito sin ejecutarlo.

    Returns
    -------
    dict con n_qubits, n_layers, encoding, n_params,
             n_cnots_per_layer, output_dim
    """
    # CNOT en anillo: nq gates por capa variacional
    # ZZ encoding: nq-1 pares adicionales
    n_cnots_var = nq                    # anillo variacional ¿Por qué anillo variacional? Hacer un ejemplo de circuito cuántico con CNOT en anillo y explicar su propósito.
    n_cnots_enc = (nq - 1) if enc == "zz" else 0
    return {
        "n_qubits"         : nq,
        "n_layers"         : L,
        "encoding"         : enc,
        "n_params"         : L * nq * 2,   # RZ + RY por qubit por capa
        "n_cnots_per_layer": n_cnots_var,
        "n_cnots_encoding" : n_cnots_enc,
        "output_dim"       : nq,
    }