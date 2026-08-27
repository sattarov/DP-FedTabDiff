# Paper-to-code guide

The implementation follows the notation in arXiv:2412.16083v2.

- **Equations 1–3:** `GaussianDiffusion.q_sample` and
  `GaussianDiffusion.predict_mean` in `src/dp_fedtabdiff/diffusion.py`.
- **FinDiff representation:** `TabularMLP` accepts the embedded tabular
  vector, sinusoidal timestep embedding, and optional label embedding.
- **Equation 4:** `fedavg` in `src/dp_fedtabdiff/federated.py`; client sizes
  are the aggregation weights.
- **Equation 6:** `clip_per_sample_gradients` followed by
  `add_gaussian_noise` in `src/dp_fedtabdiff/privacy.py`.
- **Flower workflow:** `DPFedTabDiffClient` and `start_simulation` in
  `src/dp_fedtabdiff/federated_flower.py`.
- **Persistent accounting:** `PersistentPrivacyEngine` in
  `src/dp_fedtabdiff/privacy.py`; attach it once per client and retain the
  returned private optimizer across all local steps and communication rounds.
- **FinDiff-style preprocessing:** `DataTransformer` in
  `src/dp_fedtabdiff/data.py`, adapted from the `DataTransformer` pattern in
  the companion DP-FinDiff repository.

The original research code used Flower and Opacus. Those integrations are
optional dependencies so that the mathematical components remain runnable
without a federated runtime or GPU. The next integration layer should keep a
single privacy accountant alive across all local steps and communication
rounds and report achieved `(epsilon, delta)` for every experiment.
