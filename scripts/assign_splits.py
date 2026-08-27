#!/usr/bin/env python3
"""Asigna splits temporales a ventanas features-0.2.0 (solo lectura de data/features/).

Uso:
  .venv/bin/python scripts/assign_splits.py

Entrada (no se modifica):
  data/features/features_manifest.json  (data_version == features-0.2.0)
  data/features/windows_65_stride{1,10,30,65}.parquet  (SHA verificados)

Salidas (splits-0.1.0):
  data/splits/window_splits_stride{1,10,30,65}.parquet
  data/splits/split_manifest.json
  data/splits/checksums.sha256
  data/splits/README.md

No entrena, no genera sintéticos, no calibra, no mezcla strides, no Ridge.
No recalcula features. No escribe en data/clean/ ni data/features/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_VERSION = "splits-0.1.0"
FEATURES_DATA_VERSION = "features-0.2.0"
PRIMARY_STRIDE = 1
STRIDES = [1, 10, 30, 65]

DONORS = frozenset(
    ["AMD", "INTC", "QCOM", "AVGO", "MU", "TXN", "ADI", "MCHP", "MRVL", "NXPI"]
)
TARGET = "NVDA"

SPLIT_LABELS = (
    "donor_train",
    "donor_val",
    "nvda_visible",
    "nvda_test",
    "unused",
)

# Inclusive date ranges by window_end_date (contrato congelado)
DONOR_TRAIN_START = pd.Timestamp("2012-01-01")
DONOR_TRAIN_END = pd.Timestamp("2021-12-31")
DONOR_VAL_START = pd.Timestamp("2022-01-01")
DONOR_VAL_END = pd.Timestamp("2022-12-31")
NVDA_VISIBLE_START = pd.Timestamp("2022-07-01")
NVDA_VISIBLE_END = pd.Timestamp("2022-12-31")
NVDA_TEST_START = pd.Timestamp("2023-01-01")
NVDA_TEST_END = pd.Timestamp("2025-12-31")

EXPECTED_WINDOWS_SHA256 = {
    1: "58bf4c4788cc4ae4feed14c2173419dea876a322c9e6f66d510c9dfb6c00bccf",
    10: "5bb0cd6ce8dd0fd1ff09fcafd56f0d124f6c8b8ab4a9e0a1ae21c006da754e9d",
    30: "6caf6893c6ab526d5ed95caa8d51f95cdd896211e070adc2f88bdcf34bf5225c",
    65: "1ce567ee7074050840e032a42c93d6e17d1b3523afb02b3ae098296db3556827",
}

FEATURES_DIR = ROOT / "data" / "features"
FEATURES_MANIFEST = FEATURES_DIR / "features_manifest.json"
SPLITS_DIR = ROOT / "data" / "splits"
MANIFEST_PATH = SPLITS_DIR / "split_manifest.json"
CHECKSUMS_PATH = SPLITS_DIR / "checksums.sha256"
README_PATH = SPLITS_DIR / "README.md"

META_COLS = ["ticker", "window_start_date", "window_end_date"]
SPLIT_COLS = ["ticker", "window_start_date", "window_end_date", "window_row", "split"]


def windows_path(stride: int) -> Path:
    return FEATURES_DIR / f"windows_65_stride{stride}.parquet"


def splits_path(stride: int) -> Path:
    return SPLITS_DIR / f"window_splits_stride{stride}.parquet"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_date(series: pd.Series) -> pd.Series:
    """Fecha calendario naive (sin timezone), normalizada a medianoche."""
    dt = pd.to_datetime(series)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt.dt.normalize()


def assign_split_labels(ticker: pd.Series, window_end: pd.Series) -> pd.Series:
    """Una ventana → una etiqueta (mutuamente excluyentes). NVDA nunca en donor_*."""
    is_donor = ticker.isin(DONORS)
    is_nvda = ticker.eq(TARGET)

    donor_train = is_donor & window_end.between(DONOR_TRAIN_START, DONOR_TRAIN_END)
    donor_val = is_donor & window_end.between(DONOR_VAL_START, DONOR_VAL_END)
    nvda_visible = is_nvda & window_end.between(NVDA_VISIBLE_START, NVDA_VISIBLE_END)
    nvda_test = is_nvda & window_end.between(NVDA_TEST_START, NVDA_TEST_END)

    labels = pd.Series("unused", index=ticker.index, dtype="object")
    labels = labels.mask(donor_train, "donor_train")
    labels = labels.mask(donor_val, "donor_val")
    labels = labels.mask(nvda_visible, "nvda_visible")
    labels = labels.mask(nvda_test, "nvda_test")
    return labels


def build_split_frame(windows: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in META_COLS if c not in windows.columns]
    if missing:
        raise RuntimeError(f"ventanas sin columnas {missing}; got={list(windows.columns)}")

    out = pd.DataFrame(
        {
            "ticker": windows["ticker"].astype(str),
            "window_start_date": normalize_date(windows["window_start_date"]),
            "window_end_date": normalize_date(windows["window_end_date"]),
            "window_row": range(len(windows)),
        }
    )
    out["split"] = assign_split_labels(out["ticker"], out["window_end_date"])
    return out[SPLIT_COLS]


def assert_split_integrity(splits: pd.DataFrame, n_windows: int, stride: int) -> dict:
    """Join 1:1, partición exhaustiva, NVDA ∉ donor_train ∪ donor_val."""
    if len(splits) != n_windows:
        raise RuntimeError(
            f"stride={stride}: len(splits)={len(splits)} != n_windows={n_windows}"
        )
    if splits["window_row"].nunique() != n_windows:
        raise RuntimeError(f"stride={stride}: window_row no es 1:1 con ventanas")
    if not splits["window_row"].is_monotonic_increasing:
        raise RuntimeError(f"stride={stride}: window_row no está en orden 0..n-1")
    if int(splits["window_row"].iloc[0]) != 0 or int(splits["window_row"].iloc[-1]) != n_windows - 1:
        raise RuntimeError(f"stride={stride}: window_row no cubre 0..{n_windows - 1}")

    unknown = sorted(set(splits["split"]) - set(SPLIT_LABELS))
    if unknown:
        raise RuntimeError(f"stride={stride}: etiquetas desconocidas {unknown}")

    counts = {label: int((splits["split"] == label).sum()) for label in SPLIT_LABELS}
    if sum(counts.values()) != n_windows:
        raise RuntimeError(
            f"stride={stride}: partición no exhaustiva "
            f"sum={sum(counts.values())} != {n_windows}"
        )

    nvda_in_donor = splits.loc[
        splits["ticker"].eq(TARGET) & splits["split"].isin(["donor_train", "donor_val"])
    ]
    n_nvda_in_donor = int(len(nvda_in_donor))
    if n_nvda_in_donor != 0:
        raise RuntimeError(
            f"stride={stride}: NVDA en donor_train/val (n={n_nvda_in_donor})"
        )

    # Clave natural única por stride (join alternativo a window_row)
    key = splits[["ticker", "window_start_date", "window_end_date"]]
    if key.duplicated().any():
        raise RuntimeError(
            f"stride={stride}: clave (ticker, start, end) no es única"
        )

    return {
        "n_windows": n_windows,
        "counts": counts,
        "n_unused": counts["unused"],
        "n_nvda_in_donor_train_or_val": n_nvda_in_donor,
        "join_1to1_ok": True,
        "partition_exhaustive_ok": True,
        "nvda_excluded_from_donor_ok": True,
        "natural_key_unique_ok": True,
    }


def write_readme(
    *,
    windows_sha_verified: dict[int, str],
    counts_by_stride: dict[str, dict[str, int]],
    n_unused_by_stride: dict[str, int],
    asserts_ok: dict[str, bool],
) -> None:
    lines_counts = []
    for s in STRIDES:
        key = f"stride{s}"
        c = counts_by_stride[key]
        lines_counts.append(
            f"| {s} | {c['donor_train']} | {c['donor_val']} | "
            f"{c['nvda_visible']} | {c['nvda_test']} | {c['unused']} |"
        )

    unused_lines = "\n".join(
        f"| {s} | {n_unused_by_stride[f'stride{s}']} |" for s in STRIDES
    )
    sha_lines = "\n".join(
        f"| `windows_65_stride{s}.parquet` | `{windows_sha_verified[s]}` |"
        for s in STRIDES
    )

    text = f"""# Splits temporales de ventanas (`{DATA_VERSION}`)

Etiquetas de split por `window_end_date` sobre ventanas `{FEATURES_DATA_VERSION}`.

**Solo asignación de splits.** Sin entrenamiento, sin sintéticos, sin calibración
NVDA, sin mixes, sin Ridge, sin recalcular features. No se modifica
`data/clean/` ni `data/features/`.

> **Mantenimiento:** cambios de protocolo deben actualizar **este README** y
> `scripts/assign_splits.py` (y regenerar artefactos + `split_manifest.json`).

## Contrato (congelado)

| Rol | Tickers |
|---|---|
| Donors | AMD, INTC, QCOM, AVGO, MU, TXN, ADI, MCHP, MRVL, NXPI |
| Target | NVDA |

Asignación por `window_end_date` (inclusivo), **idéntica en cada stride**:

1. `donor_train`: ticker ∈ donors AND `window_end_date` ∈ [2012-01-01, 2021-12-31]
2. `donor_val`: ticker ∈ donors AND `window_end_date` ∈ [2022-01-01, 2022-12-31]
3. `nvda_visible`: ticker == NVDA AND `window_end_date` ∈ [2022-07-01, 2022-12-31]
4. `nvda_test`: ticker == NVDA AND `window_end_date` ∈ [2023-01-01, 2025-12-31]
5. `unused`: todo lo demás

Reglas:

- Una ventana → una sola etiqueta (mutuamente excluyentes).
- NVDA **nunca** en `donor_train` ni `donor_val`.
- No se borran ventanas de features; el split es artefacto aparte.
- Hay un parquet de splits **por stride** (menú 1/10/30/65).

## Entrada (solo lectura; SHA verificados)

| Fichero | SHA256 |
|---|---|
{sha_lines}

`features_manifest.json` debe reportar `data_version == {FEATURES_DATA_VERSION}`.

## Salidas

| Artefacto | Descripción |
|---|---|
| `window_splits_stride{{1,10,30,65}}.parquet` | etiquetas por ventana |
| `split_manifest.json` | manifest `{DATA_VERSION}` |
| `checksums.sha256` | digests de salidas |
| `README.md` | este documento |

### Columnas de cada `window_splits_stride*.parquet`

| Columna | Notas |
|---|---|
| `ticker` | |
| `window_start_date` | |
| `window_end_date` | criterio de split |
| `window_row` | índice 0..n-1 alineado 1:1 con el parquet `windows_65_stride*` |
| `split` | una de: {", ".join(SPLIT_LABELS)} |

Join canónico: `window_row` (mismo orden de filas que el windows del stride).
Join alternativo: `(ticker, window_start_date, window_end_date)` (única por stride).

## Conteos (n_windows por split × stride)

| stride | donor_train | donor_val | nvda_visible | nvda_test | unused |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines_counts)}

### n_unused por stride

| stride | n_unused |
|---|---:|
{unused_lines}

## Protocolo de uso (para generadores)

- Entrenar generadores **solo** con `donor_train` (donors).
- Validar hiperparámetros / early-stop con `donor_val` (donors).
- `nvda_visible`: NVDA H2-2022 reservado (visible para inspección / calibración
  **fuera de este chat**).
- `nvda_test`: hold-out NVDA 2023–2025.
- `unused`: no usar en el protocolo oficial (p. ej. NVDA pre-2022-07, donors post-2022).

`primary_stride = {PRIMARY_STRIDE}` es solo la nota de dataset comparable del menú
de features; **existen splits para todos los strides**. Un compañero elige stride
y consume el `window_splits_stride{{N}}` correspondiente.

## Asserts

| Check | Resultado |
|---|---|
| Join/cardinalidad 1:1 con cada `windows_65_stride*` | {"OK" if asserts_ok["join_1to1"] else "FAIL"} |
| Partición exhaustiva por stride | {"OK" if asserts_ok["partition"] else "FAIL"} |
| 0 NVDA en donor_train ∪ donor_val (cada stride) | {"OK" if asserts_ok["nvda_excluded"] else "FAIL"} |

## Fuera de alcance

Generación de sintéticos, calibración NVDA, mixes, Ridge y entrenamiento de
VAE/GAN/Diffusion **no** se hacen aquí (Chat de generadores / notebooks
posteriores).
"""
    README_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    if not FEATURES_MANIFEST.is_file():
        print(f"FALLO: falta {FEATURES_MANIFEST}", file=sys.stderr)
        return 1

    features_manifest = json.loads(FEATURES_MANIFEST.read_text(encoding="utf-8"))
    fv = features_manifest.get("data_version")
    if fv != FEATURES_DATA_VERSION:
        print(
            f"FALLO: features data_version={fv!r}; esperado {FEATURES_DATA_VERSION!r}",
            file=sys.stderr,
        )
        return 1

    windows_sha_verified: dict[int, str] = {}
    for stride in STRIDES:
        path = windows_path(stride)
        if not path.is_file():
            print(f"FALLO: falta {path}", file=sys.stderr)
            return 1
        got = sha256_file(path)
        expected = EXPECTED_WINDOWS_SHA256[stride]
        if got != expected:
            print(
                f"FALLO SHA windows stride={stride}:\n"
                f"  esperado: {expected}\n"
                f"  obtenido: {got}\n"
                "No se modifica data/features/. Abortando.",
                file=sys.stderr,
            )
            return 1
        windows_sha_verified[stride] = got

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    counts_by_stride: dict[str, dict[str, int]] = {}
    n_unused_by_stride: dict[str, int] = {}
    integrity_by_stride: dict[str, dict] = {}
    splits_shas: dict[int, str] = {}

    for stride in STRIDES:
        windows = pd.read_parquet(windows_path(stride))
        n_windows = int(len(windows))
        splits = build_split_frame(windows)
        integrity = assert_split_integrity(splits, n_windows=n_windows, stride=stride)

        out = splits_path(stride)
        splits.to_parquet(out, index=False)
        splits_shas[stride] = sha256_file(out)

        key = f"stride{stride}"
        counts_by_stride[key] = integrity["counts"]
        n_unused_by_stride[key] = integrity["n_unused"]
        integrity_by_stride[key] = integrity

    asserts_ok = {
        "join_1to1": all(v["join_1to1_ok"] for v in integrity_by_stride.values()),
        "partition": all(v["partition_exhaustive_ok"] for v in integrity_by_stride.values()),
        "nvda_excluded": all(
            v["nvda_excluded_from_donor_ok"] for v in integrity_by_stride.values()
        ),
    }
    if not all(asserts_ok.values()):
        print(f"FALLO asserts: {asserts_ok}", file=sys.stderr)
        return 1

    write_readme(
        windows_sha_verified=windows_sha_verified,
        counts_by_stride=counts_by_stride,
        n_unused_by_stride=n_unused_by_stride,
        asserts_ok=asserts_ok,
    )

    checksums: dict[str, str] = {}
    for s in STRIDES:
        checksums[str(splits_path(s).relative_to(ROOT))] = splits_shas[s]
    checksums[str(README_PATH.relative_to(ROOT))] = sha256_file(README_PATH)

    CHECKSUMS_PATH.write_text(
        "\n".join(f"{digest}  {rel}" for rel, digest in checksums.items()) + "\n",
        encoding="utf-8",
    )

    date_rules = {
        "donor_train": {
            "tickers": "donors",
            "window_end_date_inclusive": ["2012-01-01", "2021-12-31"],
        },
        "donor_val": {
            "tickers": "donors",
            "window_end_date_inclusive": ["2022-01-01", "2022-12-31"],
        },
        "nvda_visible": {
            "tickers": "NVDA",
            "window_end_date_inclusive": ["2022-07-01", "2022-12-31"],
        },
        "nvda_test": {
            "tickers": "NVDA",
            "window_end_date_inclusive": ["2023-01-01", "2025-12-31"],
        },
        "unused": {"tickers": "any", "note": "everything else"},
    }

    manifest = {
        "data_version": DATA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_data_version": FEATURES_DATA_VERSION,
        "features_manifest_path": str(FEATURES_MANIFEST.relative_to(ROOT)),
        "primary_stride": PRIMARY_STRIDE,
        "primary_stride_note": (
            "DEFAULT RECOMENDADO comparable from features menu; "
            "splits exist for all strides 1/10/30/65"
        ),
        "donors": sorted(DONORS),
        "target": TARGET,
        "date_rules": date_rules,
        "windows_sha256_expected": {
            f"stride{s}": EXPECTED_WINDOWS_SHA256[s] for s in STRIDES
        },
        "windows_sha256_verified": {
            f"stride{s}": windows_sha_verified[s] for s in STRIDES
        },
        "windows_paths": {
            f"stride{s}": str(windows_path(s).relative_to(ROOT)) for s in STRIDES
        },
        "n_windows_by_split_by_stride": counts_by_stride,
        "n_unused_by_stride": n_unused_by_stride,
        "asserts": {
            "join_1to1_with_windows": asserts_ok["join_1to1"],
            "partition_exhaustive_per_stride": asserts_ok["partition"],
            "zero_nvda_in_donor_train_or_val_per_stride": asserts_ok["nvda_excluded"],
            "detail_by_stride": {
                k: {
                    "n_nvda_in_donor_train_or_val": v["n_nvda_in_donor_train_or_val"],
                    "join_1to1_ok": v["join_1to1_ok"],
                    "partition_exhaustive_ok": v["partition_exhaustive_ok"],
                    "natural_key_unique_ok": v["natural_key_unique_ok"],
                }
                for k, v in integrity_by_stride.items()
            },
        },
        "schema": {
            "split_columns": SPLIT_COLS,
            "join_key": "window_row (row-aligned 1:1 with windows_65_stride*)",
            "alternate_join_key": "(ticker, window_start_date, window_end_date)",
            "split_labels": list(SPLIT_LABELS),
        },
        "output_paths": {
            "splits_by_stride": {
                f"stride{s}": str(splits_path(s).relative_to(ROOT)) for s in STRIDES
            },
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "checksums": str(CHECKSUMS_PATH.relative_to(ROOT)),
            "readme": str(README_PATH.relative_to(ROOT)),
        },
        "checksums_sha256": checksums,
        "notes": [
            "Same date rules for every stride; pick a stride and use that split file.",
            "NVDA never in donor_train or donor_val.",
            "Does not delete or rewrite windows under data/features/.",
            "Generation / calibration / mixes / Ridge / training are out of this chat.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Re-hash README already in checksums; append manifest digest after write
    # (manifest intentionally listed separately in print; checksums cover splits+readme)
    print("OK — splits asignados")
    print(f"  data_version:       {DATA_VERSION}")
    print(f"  features_version:   {FEATURES_DATA_VERSION}")
    print(f"  primary_stride:     {PRIMARY_STRIDE} (nota; splits para todos)")
    for s in STRIDES:
        c = counts_by_stride[f"stride{s}"]
        print(
            f"  stride={s:<2}  train={c['donor_train']:<6} "
            f"val={c['donor_val']:<5} vis={c['nvda_visible']:<4} "
            f"test={c['nvda_test']:<4} unused={c['unused']}"
        )
    print(f"  asserts NVDA∉donor: {asserts_ok['nvda_excluded']}")
    for s in STRIDES:
        print(f"  splits s={s} sha:   {splits_shas[s]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
