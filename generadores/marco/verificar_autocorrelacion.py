import numpy as np

donor_train = np.load("cache_donor_train_shared.npz")["values"]
canales = ["log_return", "log_high_low_range", "log1p_volume"]

print("=== Autocorrelacion lag-1 media, por canal, dentro de cada ventana ===")
for i, nombre in enumerate(canales):
    serie = donor_train[:, :, i]
    x_t, x_t1 = serie[:, :-1], serie[:, 1:]
    autocorrs = []
    for w in range(serie.shape[0]):
        if x_t[w].std() > 1e-8 and x_t1[w].std() > 1e-8:
            autocorrs.append(np.corrcoef(x_t[w], x_t1[w])[0, 1])
    print(f"{nombre:20s}: autocorrelacion lag-1 media = {np.mean(autocorrs):.4f}")
