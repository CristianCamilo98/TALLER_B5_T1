# Runtime results (not tracked)

This folder holds the local output of the most recent run of
`evaluate_fidelity.py` / `plot_fidelity.py`: `evaluation_manifest.json`,
`tables/*.csv`, and `figures/*.png`. All three are gitignored, so after a
fresh `git clone` this folder is effectively empty except for this file and
`.gitignore`.

Running the pipeline overwrites this folder's contents in place:

```bash
python common_pipeline/02_fidelity/evaluate_fidelity.py
python common_pipeline/02_fidelity/plot_fidelity.py
```

## Where to find the final numbers without running anything

The authoritative, tracked, byte-exact copy of the STRICT_FINAL fidelity
results is at
[`artifacts/final/strict_final_20260902/fidelity/`](../../../artifacts/final/strict_final_20260902/fidelity/),
frozen at git tag `strict-final-20260902`.
