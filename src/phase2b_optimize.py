"""
phase2b_optimize.py
-------------------
Fase 2b: Optimización variacional del circuito cuántico.
Basado en el artículo s10052-025-15189-4 (adaptado para regresión).

Diferencia con phase2:
- phase2: θ aleatorio fijo (sin optimizar)
- phase2b: θ optimizado minimizando MSE con COBYLA
"""
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config_utils import load_config, get_paths, get_combos
from src.utils.io_utils import load_npz, save_npz, ensure_dirs
from src.utils.log_utils import get_logger
from src.utils.quantum_utils import build_circuit, get_qfeatures, init_theta

log = get_logger("phase2b")

def mse_cost(theta_flat, X_train_n, y_train, circuit, nq, L):
    """
    Función de costo MSE para el optimizador COBYLA.
    
    Qué hace:
    1. Reconstruir θ desde vector plano (COBYLA trabaja con vectores 1D)
    2. Evaluar circuito con θ actual
    3. Hacer predicción lineal simple
    4. Calcular MSE
    """
    # Reconstruir θ con la forma correcta (L, nq, 2)
    theta = theta_flat.reshape(L, nq, 2).astype(np.float32)
    
    # Evaluar circuito (sin verbose para no llenar el log)
    X_q = get_qfeatures(X_train_n, circuit, theta, 
                        batch_size=256, verbose=False)
    
    # Predicción lineal: promedio ponderado de features cuánticas
    # (capa de salida simple)
    y_pred = X_q.mean(axis=1) * (y_train.max() - y_train.min()) + y_train.mean()
    
    # MSE
    mse = float(np.mean((y_train - y_pred) ** 2))
    return mse

def optimize_theta(nq, L, enc, X_train_n, y_train, seed=42, max_iter=100):
    """
    Optimiza θ usando COBYLA para minimizar MSE.
    
    Parámetros
    ----------
    max_iter : máximo de iteraciones (100-200 es razonable)
    
    Devuelve
    --------
    theta_opt : array (L, nq, 2) — parámetros optimizados
    history   : lista de valores MSE durante la optimización
    """
    circuit = build_circuit(nq, L, enc)
    theta_0 = init_theta(L, nq, seed=seed).flatten()  # vector 1D para COBYLA
    
    history = []
    iteration = [0]
    
    def callback_mse(theta_flat):
        """Registra el progreso cada 10 iteraciones."""
        iteration[0] += 1
        if iteration[0] % 10 == 0:
            mse = mse_cost(theta_flat, X_train_n, y_train, circuit, nq, L)
            history.append(mse)
            log.info(f"   Iteración {iteration[0]:3d} | MSE = {mse:.4f}")
    
    log.info(f"Optimizando θ con COBYLA: nq={nq}, L={L}, enc={enc}")
    log.info(f"Parámetros a optimizar: {len(theta_0)}")
    
    result = minimize(
        fun     = mse_cost,
        x0      = theta_0,
        args    = (X_train_n, y_train, circuit, nq, L),
        method  = "COBYLA",         # ← optimizador del artículo
        options = {
            "maxiter": max_iter,
            "rhobeg" : 0.5,         # radio inicial de búsqueda
        },
        callback = callback_mse,
    )
    
    theta_opt = result.x.reshape(L, nq, 2).astype(np.float32)
    log.info(f"Optimización completada: MSE_final = {result.fun:.4f}")
    log.info(f"Convergió: {result.success} | Mensaje: {result.message}")
    
    return theta_opt, history

def run(cfg: dict) -> None:
    """
    Fase 2b: Igual que fase 2 pero con θ optimizado por COBYLA+MSE.
    Solo corre para una combinación a la vez (es costoso).
    """
    log.info("=" * 60)
    log.info("FASE 2b — Optimización variacional (COBYLA + MSE)")
    log.info("=" * 60)
    
    paths = get_paths(cfg)
    proc_dir = paths["processed_dir"]
    
    # Cargar datos reducidos
    reduced = load_npz(proc_dir / "reduced.npz")
    splits  = load_npz(proc_dir / "splits.npz")
    
    X_train = reduced["X_train_r"]
    y_train = splits["y_train"]
    
    # Normalizar a [-π, π] (igual que en phase2)
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    rng   = X_max - X_min
    rng[rng == 0] = 1.0
    X_train_n = (2.0 * np.pi * (X_train - X_min) / rng - np.pi).astype(np.float32)
    
    # Configuración del barrido
    p2b = cfg.get("phase2b", {})
    max_iter = int(p2b.get("max_iter", 100))
    combos   = get_combos(cfg)
    
    for nq, L, enc in combos:
        log.info(f"\nCombinación: nq={nq}, L={L}, enc={enc}")
        
        # Optimizar θ
        theta_opt, history = optimize_theta(
            nq, L, enc, X_train_n, y_train,
            seed=42, max_iter=max_iter
        )
        
        # Guardar θ optimizado (sobreescribe el aleatorio de phase2)
        fname = proc_dir / f"qfeatures_{nq}q_{L}l_{enc}.npz"
        if fname.exists():
            data = dict(np.load(str(fname), allow_pickle=True))
            data["theta"]         = theta_opt
            data["theta_history"] = np.array(history)
            np.savez_compressed(str(fname), **data)
            log.info(f"θ optimizado guardado en: {fname.name}")
    
    log.info("\nFASE 2b completada ✓")

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run(cfg)