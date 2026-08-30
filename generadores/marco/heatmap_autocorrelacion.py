import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_LAG = 15
CANALES = ["log_return", "log_high_low_range", "log1p_volume"]

def autocorr_matrix(data, max_lag):
    n_channels = data.shape[-1]
    result = np.zeros((n_channels, max_lag))
    for c in range(n_channels):
        serie = data[:, :, c]
        for lag in range(1, max_lag + 1):
            x_t, x_tl = serie[:, :-lag], serie[:, lag:]
            corrs = [np.corrcoef(x_t[w], x_tl[w])[0, 1] for w in range(serie.shape[0])
                     if x_t[w].std() > 1e-8 and x_tl[w].std() > 1e-8]
            result[c, lag - 1] = np.mean(corrs) if corrs else np.nan
    return result

donor_train = np.load("cache_donor_train_shared.npz")["values"]
synthetic = np.load("synthetic_scaled_raw.npz")["values"]

ac_real = autocorr_matrix(donor_train, MAX_LAG)
ac_synth = autocorr_matrix(synthetic, MAX_LAG)

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
vmax = max(np.nanmax(np.abs(ac_real)), np.nanmax(np.abs(ac_synth)))
for ax, mat, title in zip(axes, [ac_real, ac_synth], ["donor_train (real)", "sintetico (sin calibrar)"]):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(MAX_LAG)); ax.set_xticklabels(range(1, MAX_LAG + 1))
    ax.set_yticks(range(3)); ax.set_yticklabels(CANALES)
    ax.set_xlabel("lag (dias)"); ax.set_title(title)
fig.colorbar(im, ax=axes, label="autocorrelacion media")
plt.savefig("heatmap_autocorrelacion.png", dpi=120, bbox_inches="tight")
print("Guardado: heatmap_autocorrelacion.png")
