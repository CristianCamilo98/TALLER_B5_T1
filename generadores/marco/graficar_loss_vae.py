import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

history = np.load("loss_history.npz")
train_loss = history["train"]
val_loss = history["val"]
best_epoch = int(history["best_epoch"])

epochs = range(len(train_loss))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_loss, label="train (reconstrucción)", marker="o", markersize=3)
ax.plot(epochs, val_loss, label="validation (reconstrucción)", marker="o", markersize=3)
ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"mejor época ({best_epoch})")
ax.set_xlabel("época")
ax.set_ylabel("Huber loss (datos escalados)")
ax.set_title("Convergencia del TimeVAE -- donors semiconductores")
ax.legend()
plt.tight_layout()
plt.savefig("loss_vae_donors.png", dpi=120)
print("Guardado: loss_vae_donors.png")