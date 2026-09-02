"""
Investigacion adicional del donor SHA -- SOLO LECTURA.
"""
import hashlib
import json
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

donor_path = Path("data/features/windows/donor_train.parquet")
EXPECTED_CANONICAL = "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f"

actual_sha = sha256_file(donor_path)
print(f"SHA real del archivo en disco ahora mismo: {actual_sha}")
print(f"SHA esperado (segun companeros):            {EXPECTED_CANONICAL}")

manifest_path = Path("data/features/features_manifest.json")
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    recorded = manifest.get("checksums_sha256", {})
    donor_key = next((k for k in recorded if "donor_train" in k), None)
    if donor_key:
        recorded_sha = recorded[donor_key]
        print(f"\nSHA registrado en features_manifest.json ({donor_key}): {recorded_sha}")
        print(f"  coincide con archivo en disco ahora mismo: {recorded_sha == actual_sha}")
        print(f"  coincide con el 'canonico' esperado:       {recorded_sha == EXPECTED_CANONICAL}")
    else:
        print("\nNo se encontro entrada de donor_train en features_manifest.json checksums")
else:
    print("\nfeatures_manifest.json no encontrado")

print("\n=== git log del archivo (ultimos cambios, solo lectura) ===")
import subprocess
result = subprocess.run(["git", "log", "--oneline", "-5", "--", str(donor_path)], capture_output=True, text=True)
print(result.stdout)
