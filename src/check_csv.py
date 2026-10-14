"""
check_csv.py — Diagnóstico rápido de archivos CSV del proyecto qml_xmax
Ejecutar con: python src/check_csv.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas rutas si es necesario
# ─────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

ARCHIVOS_RAW = [
    "datosFD_num.csv",
    "datosSD_num.csv",
    "SDnewversion.csv",
]

TARGET_COL = "FD_RS_fXmax"

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────
def separador(titulo):
    print("\n" + "="*60)
    print(f"  {titulo}")
    print("="*60)

def check_archivo_raw(nombre):
    ruta = os.path.join(RAW_DIR, nombre)
    print(f"\n📄 Archivo: {nombre}")

    # ¿Existe?
    if not os.path.exists(ruta):
        print(f"   ❌ NO ENCONTRADO en {ruta}")
        return None

    # Tamaño en MB
    size_mb = os.path.getsize(ruta) / (1024 * 1024)
    print(f"   ✅ Existe  — Tamaño: {size_mb:.2f} MB")

    # Cargar
    try:
        df = pd.read_csv(ruta, sep='\t')
        if df.shape[1] == 1:
            # Si sigue siendo c1 colunmna, intentar con coma
            df = pd.read_csv(ruta, sep = ',')
    except Exception as e:
        print(f"   ❌ Error al leer: {e}")
        return None

    # Forma
    print(f"   📐 Shape  : {df.shape[0]} filas × {df.shape[1]} columnas")

    # NaN totales
    nan_total = df.isna().sum().sum()
    print(f"   🔢 NaN totales: {nan_total}")

    # Columnas con NaN (solo las que tienen al menos 1)
    cols_nan = df.isna().sum()
    cols_nan = cols_nan[cols_nan > 0]
    if len(cols_nan) > 0:
        print(f"   ⚠️  Columnas con NaN ({len(cols_nan)} columnas):")
        for col, n in cols_nan.items():
            pct = 100 * n / len(df)
            print(f"       {col:<40} → {n:>5} NaN  ({pct:.1f}%)")
    else:
        print(f"   ✅ Sin NaN en ninguna columna")

    # ¿Existe la columna target?
    if TARGET_COL in df.columns:
        n_target_nan = df[TARGET_COL].isna().sum()
        print(f"   🎯 Target '{TARGET_COL}': ✅ existe  — {n_target_nan} NaN")
    else:
        print(f"   🎯 Target '{TARGET_COL}': ❌ NO existe en este archivo")

    # Primeras 3 columnas
    print(f"   📋 Primeras 5 columnas: {list(df.columns[:5])}")

    # Tipos de datos
    n_num = df.select_dtypes(include=[np.number]).shape[1]
    n_obj = df.select_dtypes(include=['object']).shape[1]
    print(f"   🔣 Columnas numéricas: {n_num}  |  Columnas texto: {n_obj}")

    return df

def check_processed():
    separador("BLOQUE 3 — Archivos Procesados (data/processed/)")

    if not os.path.exists(PROCESSED_DIR):
        print(f"\n   ❌ Directorio '{PROCESSED_DIR}' no existe")
        return

    archivos = os.listdir(PROCESSED_DIR)

    if len(archivos) == 0:
        print(f"\n   ⚠️  El directorio existe pero está VACÍO")
        print(f"   → Phase0 no ha corrido todavía (o falló)")
        return

    print(f"\n   📁 Archivos encontrados en processed/ ({len(archivos)} total):")
    for f in sorted(archivos):
        ruta = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(ruta) / 1024
        print(f"   ✅  {f:<45} — {size_kb:.1f} KB")

def check_results():
    separador("BLOQUE 4 — Archivos de Resultados (data/results/)")

    results_dir = "data/results"
    if not os.path.exists(results_dir):
        print(f"\n   ❌ Directorio '{results_dir}' no existe")
        return

    # Recorrer subdirectorios
    for root, dirs, files in os.walk(results_dir):
        archivos_json = [f for f in files if f.endswith('.json')]
        archivos_csv  = [f for f in files if f.endswith('.csv')]
        archivos_png  = [f for f in files if f.endswith('.png')]

        if archivos_json or archivos_csv or archivos_png:
            rel = os.path.relpath(root, results_dir)
            print(f"\n   📁 {rel}/")
            if archivos_json:
                print(f"      JSON : {len(archivos_json)} archivos")
            if archivos_csv:
                print(f"      CSV  : {len(archivos_csv)} archivos")
            if archivos_png:
                print(f"      PNG  : {len(archivos_png)} archivos")

def check_config():
    separador("BLOQUE 5 — Configuración (config.yaml)")

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"\n   ❌ config.yaml NO encontrado")
        return

    print(f"\n   ✅ config.yaml existe")
    print(f"\n   📄 Contenido completo:")
    print("   " + "-"*50)
    with open(config_path, "r") as f:
        for line in f:
            print("   " + line, end="")
    print("\n   " + "-"*50)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "🔍 "*20)
    print("  CHECK_CSV.PY — Diagnóstico del proyecto qml_xmax")
    print("🔍 "*20)

    # Directorio de trabajo
    print(f"\n📂 Directorio de trabajo actual: {os.getcwd()}")

    # ── Bloque 1: Archivos raw ──
    separador("BLOQUE 1 — Archivos Raw (data/raw/)")
    dfs = {}
    for nombre in ARCHIVOS_RAW:
        df = check_archivo_raw(nombre)
        if df is not None:
            dfs[nombre] = df

    # ── Bloque 2: Consistencia entre FD y SD ──
    separador("BLOQUE 2 — Consistencia FD vs SD")
    fd_name = "datosFD_num.csv"
    sd_name = "datosSD_num.csv"

    if fd_name in dfs and sd_name in dfs:
        n_fd = len(dfs[fd_name])
        n_sd = len(dfs[sd_name])
        print(f"\n   datosFD_num.csv : {n_fd} filas")
        print(f"   datosSD_num.csv : {n_sd} filas")
        if n_fd == n_sd:
            print(f"   ✅ Mismo número de filas — están alineados")
        else:
            print(f"   ❌ DIFERENTE número de filas — revisar merge")

        # Columnas en común
        cols_fd = set(dfs[fd_name].columns)
        cols_sd = set(dfs[sd_name].columns)
        comunes = cols_fd & cols_sd
        print(f"\n   Columnas en común: {len(comunes)}")
        if comunes:
            print(f"   → {sorted(comunes)}")
    else:
        print("\n   ⚠️  No se pudieron cargar ambos archivos para comparar")

    # ── Bloque 3: Archivos procesados ──
    check_processed()

    # ── Bloque 4: Resultados ──
    check_results()

    # ── Bloque 5: Config ──
    check_config()

    # ── Resumen final ──
    separador("RESUMEN FINAL")
    print(f"\n   Archivos raw encontrados : {len(dfs)} / {len(ARCHIVOS_RAW)}")

    problemas = []
    for nombre, df in dfs.items():
        if TARGET_COL not in df.columns and nombre != "SDnewversion.csv":
            problemas.append(f"❌ {nombre} no tiene columna target '{TARGET_COL}'")
        nan_total = df.isna().sum().sum()
        if nan_total > 0:
            problemas.append(f"⚠️  {nombre} tiene {nan_total} NaN en total")

    if problemas:
        print(f"\n   🚨 Problemas detectados:")
        for p in problemas:
            print(f"      {p}")
    else:
        print(f"\n   ✅ Sin problemas críticos detectados")

    print(f"\n{'='*60}")
    print(f"  FIN DEL DIAGNÓSTICO")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()