# GAN — Cristian (WGAN-GP)

Generador adversarial de ventanas donor `[65, 3]` para el experimento NVDA sintético.

## Contrato


| Elemento      | Valor                                                                  |
| ------------- | ---------------------------------------------------------------------- |
| Entrada train | `data/features/windows/donor_train.parquet` (4.910 ventanas, stride 5) |
| Validación    | `data/features/windows/donor_validation.parquet` (380 ventanas)        |
| Canales       | `log_return`, `log_high_low_range`, `log1p_volume`                     |
| Shape         | 65 × 3 (195 floats)                                                    |
| Modelo        | WGAN-GP (1D Conv critic + MLP generator)                               |
| Prohibido     | Tunear con `nvda_visible` o `test_index`                               |


## Estructura

```text
generadores/cristian/
├── configs/wgan_gp.yaml
├── scripts/
│   ├── train_wgan_gp.py
│   └── generate_synthetic.py
├── src/
│   ├── data.py          # carga parquets, normalización por canal
│   ├── models.py        # generator + critic
│   ├── wgan_gp.py       # bucle de entrenamiento
│   ├── metrics.py       # MMD, stats, autocorrelación
│   └── io.py            # checkpoints y parquets sintéticos
├── notebooks/
│   ├── 01_eda_donor_windows.ipynb
│   ├── 02_train_wgan_gp.ipynb
│   ├── 03_validate_synthetic.ipynb
│   ├── 04_eval_generative.ipynb
│   └── 05_generate_synthetic.ipynb
├── artifacts/           # checkpoints (gitignored)
└── outputs/             # parquets sintéticos (gitignored)
```



## Instalación

Desde la raíz del repo (con el `.venv` del common core):

```bash
uv pip install --python .venv/bin/python -r generadores/cristian/requirements.txt
```



## Entrenamiento

```bash
cd generadores/cristian
../../.venv/bin/python scripts/train_wgan_gp.py --seed 42 --epochs 5000
```

Artefactos en `artifacts/seed_42/`:

- `checkpoints/generator_epoch_*.keras`
- `normalizer.json` (media/std por canal, solo donor_train)
- `loss_history.csv`
- `run_metadata.json`



## Generación

```bash
../../.venv/bin/python scripts/generate_synthetic.py \
  --run-dir seed_42 \
  --validate
```

Por defecto genera **5000** ventanas (`n_synthetic_windows` en `configs/wgan_gp.yaml`).
Override: `--n-samples 10000`.

Salida: `outputs/synthetic_seed42_n5000.parquet` + informe JSON de validación (con `--validate`).

## Notebooks

1. **01_eda_donor_windows** — distribución de canales, autocorrelación, PCA.
2. **02_train_wgan_gp** — entrenamiento interactivo y curvas de loss.
3. **03_validate_synthetic** — real vs sintético sobre `donor_validation`.
4. **04_eval_generative** — t-SNE, marginales (val/train vs synth), clasificador logístico (C2ST).
5. **05_generate_synthetic** — generar 5000 ventanas desde checkpoint (standalone).



## Referencia de clase

Patrones tomados de:

- `gan_examples/scripts_GANs/GAN_2_Simple_GAN_CIFAR10.ipynb` (clase `GAN`, Keras)
- `gan_examples/scripts_GANs_extra/scripts_GANs_part_II/dualgan_val.py` (Wasserstein loss)

Adaptado a series temporales multicanal en lugar de imágenes.

## Seeds oficiales

Según `configs/experiment.yaml`:

- Seeds: 42, 123, 2026

Entrenar un checkpoint por seed; generar **5000** ventanas sintéticas por seed (`n_synthetic_windows` en yaml). Mezcla con reales (ratios) = fase posterior.