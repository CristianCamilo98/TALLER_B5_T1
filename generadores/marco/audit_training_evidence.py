"""
Auditoria de evidencia de entrenamiento -- SOLO LECTURA.
No modifica nada. Solo reporta que existe realmente en disco/git.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

MARCO_DIR = Path("generadores/marco")
ROOT = Path(".")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check(label: str, path: Path) -> bool:
    exists = path.exists()
    size = f"{path.stat().st_size / 1024:.1f} KB" if exists else "-"
    print(f"  [{'OK' if exists else '--'}] {label:35s} {str(path):55s} {size}")
    return exists


print("=== 1) EVIDENCIA EN DISCO (rutas conocidas del pipeline de Marco) ===\n")

checkpoint_path = ROOT / "vae_donors_weights.weights.h5"
loss_history_path = ROOT / "loss_history.npz"
scaler_path = ROOT / "scaler_donors.npz"
figure_path = MARCO_DIR / "figures" / "loss_vae_donors.png"
requirements_path = MARCO_DIR / "requirements.txt"

has_checkpoint = check("checkpoint (.weights.h5)", checkpoint_path)
has_loss_history = check("loss_history.npz", loss_history_path)
has_scaler = check("scaler_donors.npz", scaler_path)
has_figure = check("figura loss_vae_donors.png", figure_path)
has_requirements = check("requirements.txt (VAE)", requirements_path)

print("\n=== 2) ESTADO EN GIT -- estos archivos, estan trackeados o son locales/gitignored? ===\n")
result = subprocess.run(["git", "check-ignore", "-v", str(checkpoint_path), str(loss_history_path), str(scaler_path)],
                         capture_output=True, text=True)
print("git check-ignore (checkpoint/loss_history/scaler):")
print(result.stdout if result.stdout else "  (ninguno esta gitignored -- podrian estar trackeados o simplemente ausentes de git)")

print("\n=== 3) CONTENIDO DE loss_history.npz (si existe) ===\n")
if has_loss_history:
    import numpy as np
    data = np.load(loss_history_path)
    print(f"  keys: {list(data.keys())}")
    if "train" in data and "val" in data:
        print(f"  epochs registradas: {len(data['train'])}")
        print(f"  best_epoch guardado: {data['best_epoch'] if 'best_epoch' in data else 'NO GUARDADO'}")
        print(f"  train[-1]={data['train'][-1]:.4f}  val[-1]={data['val'][-1]:.4f}")
else:
    print("  NO DISPONIBLE -- no se puede publicar curva de convergencia reproducible desde raw history.")

print("\n=== 4) SCALER -- estadisticas realmente usadas ===\n")
if has_scaler:
    import numpy as np
    scaler = np.load(scaler_path)
    mean, std = scaler["mean"], scaler["std"]
    print(f"  mean: {mean.tolist()}")
    print(f"  std:  {std.tolist()}")
    print(f"  algun std < 1e-8 (zero-variance guard activo)?: {(std < 1e-8).any()}")
else:
    print("  NO DISPONIBLE")

print("\n=== 5) CHECKPOINT -- hash SHA256 (si existe) ===\n")
if has_checkpoint:
    print(f"  SHA256: {sha256_file(checkpoint_path)}")
else:
    print("  CHECKPOINT_NOT_AVAILABLE")

print("\n=== 6) DONOR -- verificar SHA del origen de datos ===\n")
donor_path = Path("data/features/windows/donor_train.parquet")
EXPECTED_DONOR_SHA = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"
if donor_path.exists():
    actual_sha = sha256_file(donor_path)
    print(f"  donor_train.parquet SHA256 real: {actual_sha}")
    print(f"  esperado (canonico):             {EXPECTED_DONOR_SHA}")
    print(f"  coincide: {actual_sha == EXPECTED_DONOR_SHA}")
else:
    print(f"  {donor_path} NO ENCONTRADO")

print("\n=== 7) IMPORTS REALES usados por el pipeline de Marco (para requirements minimos) ===\n")
py_files = sorted(MARCO_DIR.glob("*.py"))
all_imports = set()
for f in py_files:
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            all_imports.add(line)
for imp in sorted(all_imports):
    print(f"  {imp}")
