import hashlib
import json
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

EVIDENCE_DIR = Path("generadores/marco/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_path = Path("vae_donors_weights.weights.h5")
donor_path = Path("data/features/windows/donor_train.parquet")

manifest = {
    "generator": "marco_vae",
    "architecture": "TimeVAE (Desai et al., 2022), sin condicionante",
    "latent_dim": 8,
    "framework": "tensorflow/keras",
    "training_seed": 42,
    "checkpoint": {
        "path": "vae_donors_weights.weights.h5 (local, gitignored)",
        "sha256": sha256_file(checkpoint_path),
        "status": "AVAILABLE",
    },
    "donor_provenance": {
        "canonical_sha256_per_features_manifest": "5f1e33f69b02bad86d89dcc2f67a1018cef68aaeacfbf72c310a1b7902fc268f",
        "sha256_of_local_file_used_for_this_audit": sha256_file(donor_path),
        "match": False,
        "status": "NOT_VERIFIABLE",
        "note": (
            "data/features/windows/donor_train.parquet no esta versionado en git "
            "(excluido explicitamente en .gitignore de la base comun). El SHA del "
            "archivo local en el momento de esta auditoria NO coincide con el SHA "
            "canonico registrado en features_manifest.json. No se puede confirmar "
            "con la evidencia disponible que el checkpoint actual se entreno con "
            "exactamente el donor_train certificado como oficial por el equipo. "
            "No se ha reentrenado para resolver esta discrepancia (fuera de scope "
            "de esta auditoria, ver instrucciones: NO reentrenar automaticamente)."
        ),
    },
    "hyperparameters": {
        "max_epochs": 60,
        "kl_warmup_epochs": 15,
        "kl_weight_max": 0.01,
        "patience": 10,
        "free_bits": 0.25,
        "batch_size": 64,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
    },
    "training_result": {
        "epochs_completed": 21,
        "best_epoch": 10,
        "train_recon_final": 0.2218,
        "val_recon_final": 0.3211,
        "val_recon_best": "ver loss_history.npz, epoca 10",
        "raw_history_status": "AVAILABLE (loss_history.npz, no versionado en git, gitignored)",
    },
    "scaler": {
        "status": "AVAILABLE",
        "path": "generadores/marco/evidence/scaler_stats.json",
        "fit_dtype": "float64 (corregido -- ver README, seccion Limitaciones)",
        "zero_variance_guard": "std == 0 -> 1e-8 (difiere del guard comun: sigma < 1e-8 -> 1.0)",
        "zero_variance_guard_triggered": False,
        "numerical_impact_of_discrepancy": "NINGUNO -- ningun canal tiene std < 1e-8, ambas implementaciones producen el mismo scaler en la practica",
    },
}

with open(EVIDENCE_DIR / "training_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
