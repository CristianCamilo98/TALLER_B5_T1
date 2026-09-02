import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

EVIDENCE_DIR = Path("generadores/marco/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

history = np.load("loss_history.npz")
train_loss, val_loss, best_epoch = history["train"], history["val"], int(history["best_epoch"])

epochs = range(len(train_loss))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_loss, label="train (reconstruccion)", marker="o", markersize=3)
ax.plot(epochs, val_loss, label="validation (reconstruccion)", marker="o", markersize=3)
ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"mejor epoca ({best_epoch})")
ax.set_xlabel("epoca")
ax.set_ylabel("Huber loss (datos escalados)")
ax.set_title("Convergencia del TimeVAE -- reproducida desde loss_history.npz")
ax.legend()
plt.tight_layout()
plt.savefig(EVIDENCE_DIR / "convergence_curve.png", dpi=120)
print(f"Guardado: {EVIDENCE_DIR / 'convergence_curve.png'}")
print(f"epochs={len(train_loss)}  best_epoch={best_epoch}  train[-1]={train_loss[-1]:.4f}  val[-1]={val_loss[-1]:.4f}")
