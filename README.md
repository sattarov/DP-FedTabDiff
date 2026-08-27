# DP-FedTabDiff

Educational reference implementation of **DP-FedTabDiff**, a federated
diffusion model for differentially private synthesis of mixed-type tabular
data.

This repository is being rebuilt from the original research implementation
used for [the paper](https://arxiv.org/abs/2412.16083). The code is organized
around the three ideas in the paper:

1. a FinDiff-style diffusion model for numerical and categorical columns;
2. federated local training and weighted aggregation; and
3. per-example clipping and Gaussian noise for differential privacy.

Flower adapters are provided for simulated client/server experiments, and the
same client can keep one Opacus accountant alive across all local steps and
communication rounds.

The implementation is intentionally small and readable. It is suitable for
studying the method and for CPU smoke tests; the paper-scale experiments
require a suitable GPU and the original datasets.

## Quick start

```bash
uv venv .venv --python 3.11
uv sync --all-extras
uv run pytest
```

`uv.lock` pins the resolved environment. Use `uv run ...` for project
commands so the project-local `.venv` is selected automatically. To activate
it for interactive work, run `source .venv/bin/activate`.

The core building blocks can be used independently:

```python
import torch
from dp_fedtabdiff.diffusion import GaussianDiffusion, TabularMLP
from dp_fedtabdiff.federated import fedavg

diffusion = GaussianDiffusion(steps=500)
model = TabularMLP(data_dim=16, hidden_dims=(128, 128), condition_dim=64)
x_t, noise = diffusion.q_sample(torch.randn(8, 16), torch.randint(0, 500, (8,)))
```

## Repository map

| Paper concept | Code |
| --- | --- |
| Forward/reverse diffusion, Equations 1–3 | `src/dp_fedtabdiff/diffusion.py` |
| FedAvg and local-update protocol, Equation 4 | `src/dp_fedtabdiff/federated.py` |
| DP clipping/noise, Equation 6 | `src/dp_fedtabdiff/privacy.py` |
| Mixed-type preprocessing | `src/dp_fedtabdiff/data.py` |
| Evaluation helpers | `src/dp_fedtabdiff/evaluation.py` |
| FinDiff categorical embeddings | `src/dp_fedtabdiff/findiff.py` |

The original scripts and YAML experiments are preserved outside this package
as migration references while the public API is stabilized.

The main walkthrough is
[`examples/uci_credit_federated_findiff.ipynb`](examples/uci_credit_federated_findiff.ipynb).
It uses UCI dataset ID 350, simulates multiple clients, aggregates FinDiff
weights, evaluates on a held-out test set, and reports SDV quality metrics.

## Privacy note

Differential privacy is not implied merely by using federated learning.
Production use must configure a privacy accountant, sampling rate, clipping
norm, noise multiplier, number of steps, and delta. The utilities in this
repository expose these quantities explicitly, but they are not a substitute
for a threat-model review or legal/privacy advice.

## Citation

```bibtex
@article{sattarov2024dpfedtabdiff,
  title={Federated Diffusion Modeling with Differential Privacy for Tabular Data Synthesis},
  author={Sattarov, Timur and Schreyer, Marco and Borth, Damian},
  journal={arXiv preprint arXiv:2412.16083},
  year={2024}
}
```

## License

Code is released under the MIT license. Dataset licenses remain with their
respective providers; datasets must not be committed to this repository.
