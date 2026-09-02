"""
AUDITORIA FORENSE READ-ONLY -- equivalencia de datos donor_train.
NO modifica nada. NO reentrena. NO regenera output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_SHA = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"
CANONICAL_MEAN = np.array([0.0008114289710088066, 0.02602580514891484, 16.06027218135258], dtype=np.float64)
CANONICAL_STD = np.array([0.023515504591060377, 0.01672428879172832, 1.0933253360280637], dtype=np.float64)
CHANNEL_ORDER = ["log_return", "log_high_low_range", "log1p_volume"]
EXPECTED_SHAPE = (4910, 65, 3)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def reconstruct_tensor(row) -> np.ndarray:
    return np.asarray(row, dtype=np.float64).reshape(65, 3)


print("=" * 70)
print("1) IDENTIFICAR LOS DOS DONORS")
print("=" * 70)

marco_path = Path("data/features/windows/donor_train.parquet")
marco_sha = sha256_file(marco_path)
marco_df = pd.read_parquet(marco_path)

print(f"\nDONOR_MARCO_LOCAL")
print(f"  path:    {marco_path}")
print(f"  SHA256:  {marco_sha}")
print(f"  rows:    {len(marco_df)}")
print(f"  columns: {marco_df.columns.tolist()}")
print(f"  dtypes:\n{marco_df.dtypes}")

print(f"\nDONOR_CANONICAL")
print(f"  SHA256 esperado: {CANONICAL_SHA}")
print(f"  shape esperada:  {EXPECTED_SHAPE}")

print("\n--- Buscando fisicamente cualquier archivo con el SHA canonico en el repo ---")
search_roots = [Path("data"), Path("common_pipeline"), Path("generadores")]
canonical_path_found = None
for root in search_roots:
    if not root.exists():
        continue
    for p in root.rglob("*.parquet"):
        try:
            if sha256_file(p) == CANONICAL_SHA:
                canonical_path_found = p
                break
        except Exception:
            continue
    if canonical_path_found:
        break

if canonical_path_found:
    print(f"  ENCONTRADO: {canonical_path_found}")
else:
    print("  NO ENCONTRADO fisicamente en data/, common_pipeline/, generadores/.")
    print("  Segun protocolo: no se descarga copia nueva. Se continua con lo que")
    print("  SI es posible sin el archivo fisico (secciones 5 y 7), y se marca")
    print("  la comparacion elemento-a-elemento (secciones 2-4) como NO REALIZABLE.")

print("\n" + "=" * 70)
print("2-4) COMPARACION ESTRUCTURAL Y RAW -- solo si el canonico esta disponible")
print("=" * 70)

if canonical_path_found:
    canon_df = pd.read_parquet(canonical_path_found)
    print(f"\nshape A (marco):    {len(marco_df)} filas, columnas {marco_df.columns.tolist()}")
    print(f"shape B (canonico): {len(canon_df)} filas, columnas {canon_df.columns.tolist()}")

    sort_keys = [c for c in ["ticker", "window_start_date"] if c in marco_df.columns]
    marco_sorted = marco_df.sort_values(sort_keys).reset_index(drop=True)
    canon_sorted = canon_df.sort_values(sort_keys).reset_index(drop=True)

    tensor_a = np.stack([reconstruct_tensor(r) for r in marco_sorted["features_flat"]])
    tensor_b = np.stack([reconstruct_tensor(r) for r in canon_sorted["features_flat"]])

    print(f"\ntensor A: {tensor_a.shape}   tensor B: {tensor_b.shape}")
    if tensor_a.shape == tensor_b.shape:
        diff = np.abs(tensor_a - tensor_b)
        n_diff_elements = int((diff > 0).sum())
        pct_diff = 100 * n_diff_elements / diff.size
        print(f"\nallclose exact (atol=0): {np.array_equal(tensor_a, tensor_b)}")
        print(f"max_abs_diff global:  {diff.max():.10e}")
        print(f"mean_abs_diff global: {diff.mean():.10e}")
        print(f"elementos distintos:  {n_diff_elements} / {diff.size} ({pct_diff:.6f}%)")
        for i, ch in enumerate(CHANNEL_ORDER):
            print(f"  canal {ch:20s}: max_abs_diff={diff[:,:,i].max():.6e}  mean_abs_diff={diff[:,:,i].mean():.6e}")

        if "ticker" in marco_sorted.columns:
            same_tickers = marco_sorted["ticker"].equals(canon_sorted["ticker"])
            print(f"\nmismos tickers (tras ordenar): {same_tickers}")
        if "window_start_date" in marco_sorted.columns:
            same_dates = marco_sorted["window_start_date"].equals(canon_sorted["window_start_date"])
            print(f"mismas fechas (tras ordenar):  {same_dates}")
    else:
        print("SHAPES DISTINTAS -- no se puede hacer diff elemento a elemento directamente.")
else:
    print("\nOMITIDO -- archivo canonico no disponible fisicamente para comparacion directa.")

print("\n" + "=" * 70)
print("5) COMPARAR NORMALIZACION (float64 estricto, sin necesitar el archivo canonico)")
print("=" * 70)

tensor_marco = np.stack([reconstruct_tensor(r) for r in marco_df["features_flat"]])
print(f"\ntensor_marco shape: {tensor_marco.shape}  (esperado: {EXPECTED_SHAPE})")
print(f"shape coincide con esperada: {tensor_marco.shape == EXPECTED_SHAPE}")

mean_marco = tensor_marco.mean(axis=(0, 1), dtype=np.float64)
std_marco = tensor_marco.std(axis=(0, 1), ddof=0, dtype=np.float64)

print(f"\n{'canal':22s} {'mean_marco':>16s} {'mean_canonico':>16s} {'diff_abs':>12s} {'diff_rel':>12s}")
for i, ch in enumerate(CHANNEL_ORDER):
    d_abs = mean_marco[i] - CANONICAL_MEAN[i]
    d_rel = d_abs / CANONICAL_MEAN[i]
    print(f"{ch:22s} {mean_marco[i]:16.10f} {CANONICAL_MEAN[i]:16.10f} {d_abs:12.3e} {d_rel:12.3e}")

print(f"\n{'canal':22s} {'std_marco':>16s} {'std_canonico':>16s} {'diff_abs':>12s} {'diff_rel':>12s}")
for i, ch in enumerate(CHANNEL_ORDER):
    d_abs = std_marco[i] - CANONICAL_STD[i]
    d_rel = d_abs / CANONICAL_STD[i]
    print(f"{ch:22s} {std_marco[i]:16.10f} {CANONICAL_STD[i]:16.10f} {d_abs:12.3e} {d_rel:12.3e}")

mean_match = np.allclose(mean_marco, CANONICAL_MEAN, rtol=1e-6)
std_match = np.allclose(std_marco, CANONICAL_STD, rtol=1e-6)
print(f"\nmean coincide (rtol=1e-6): {mean_match}")
print(f"std coincide (rtol=1e-6):  {std_match}")

std_guarded = np.where(std_marco < 1e-8, 1.0, std_marco)
print(f"\nguard canonico aplicado (sigma<1e-8 -> 1.0): activado en algun canal = {(std_marco < 1e-8).any()}")

print("\n" + "=" * 70)
print("6) CAUSA DEL SHA DISTINTO -- metadata parquet propia (no comparativa, canonico no disponible)")
print("=" * 70)

try:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(marco_path)
    meta = pf.metadata
    print(f"\npyarrow version usada para leer: {pq.__import__('pyarrow').__version__ if hasattr(pq, '__import__') else 'ver abajo'}")
    import pyarrow
    print(f"pyarrow version: {pyarrow.__version__}")
    print(f"num_row_groups: {meta.num_row_groups}")
    print(f"created_by: {meta.created_by}")
    print(f"format_version: {meta.format_version}")
    for i in range(min(meta.num_row_groups, 3)):
        rg = meta.row_group(i)
        print(f"  row_group {i}: num_rows={rg.num_rows}  compression={rg.column(0).compression}")
except Exception as e:
    print(f"No se pudo leer metadata pyarrow: {e}")

print("\nVEREDICTO de esta seccion: POSSIBLE (no VERIFIED) -- sin el archivo canonico")
print("fisico no se puede comparar metadata lado a lado. Solo se reporta la propia.")

print("\n" + "=" * 70)
print("7) VINCULO CON EL CHECKPOINT -- cadena temporal de artefactos locales")
print("=" * 70)

artifacts = {
    "donor_train.parquet": marco_path,
    "cache_donor_train_shared.npz": Path("cache_donor_train_shared.npz"),
    "scaler_donors.npz": Path("scaler_donors.npz"),
    "vae_donors_weights.weights.h5": Path("vae_donors_weights.weights.h5"),
    "loss_history.npz": Path("loss_history.npz"),
}
import datetime
for name, p in artifacts.items():
    if p.exists():
        mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        print(f"  {name:38s} mtime={mtime}")
    else:
        print(f"  {name:38s} NO ENCONTRADO")

if Path("scaler_donors.npz").exists():
    scaler = np.load("scaler_donors.npz")
    scaler_matches_marco_recalc = np.allclose(scaler["mean"], mean_marco, rtol=1e-6) and np.allclose(scaler["std"], std_marco, rtol=1e-6)
    print(f"\nscaler_donors.npz (usado por el checkpoint actual) coincide con")
    print(f"recalculo float64 del donor_marco_local actual (rtol=1e-6): {scaler_matches_marco_recalc}")
    print("(Si True: el checkpoint conservado SI corresponde al donor_train.parquet")
    print(" que existe ahora mismo en disco -- no hay evidencia de que se usara otro.)")
