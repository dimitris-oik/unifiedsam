# Unified SAM

**One update rule that unifies SAM and USAM — and the first general-purpose convergence theory for both.**

[![Paper: ICLR 2025](https://img.shields.io/badge/Paper-ICLR%202025-1f6feb)](https://openreview.net/forum?id=8rvqpiTTFv)
[![arXiv](https://img.shields.io/badge/arXiv-2503.02225-b31b1b)](https://arxiv.org/abs/2503.02225)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`unifiedSAM` is the official PyTorch implementation of the optimizer proposed in:

> **Sharpness-Aware Minimization: General Analysis and Improved Rates**
> Dimitris Oikonomou, Nicolas Loizou. *ICLR 2025.*

The package provides a single `torch.optim.Optimizer` subclass — `unifiedSAM` — that subsumes both Sharpness-Aware Minimization (SAM) and its unnormalized variant (USAM) under one parametric update rule controlled by a single coefficient $\lambda \in [0, 1]$. Setting $\lambda = 0$ recovers USAM, $\lambda = 1$ recovers SAM, and intermediate or **time-varying** schedules ($\lambda_t = 1/t$, $\lambda_t = 1-1/t$) open up a continuum of SAM-style methods that have never been explicitly studied before. Our analysis provides the **first** convergence guarantees for SAM-type methods under the *Expected Residual* condition — replacing the much stronger bounded-variance / bounded-gradient assumptions of prior work — and supports arbitrary sampling (uniform, importance, mini-batch).

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The algorithm](#the-algorithm)
- [API reference](#api-reference)
- [Experiments](#experiments)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Installation

From source:

```bash
git clone https://github.com/dimitris-oik/unifiedsam.git
cd unifiedsam
pip install -r requirements.txt
```

Then `unifiedsam.py` can be imported directly from the repo root, or copied next to your training script.

Requirements: `torch`, `numpy`, `scipy` (only for the numpy experiments), Python 3.8+.

---

## Quick start

Like all SAM-style optimizers, `unifiedSAM` performs **two forward/backward passes per step** and therefore requires a `closure` that re-evaluates the loss:

```python
import torch
from unifiedsam import unifiedSAM

model     = MyModel()
criterion = torch.nn.CrossEntropyLoss()

optimizer = unifiedSAM(
    model.parameters(),
    base_optimizer=torch.optim.SGD,   # inner optimizer used after the ascent step
    rho=0.1,                          # sharpness radius
    lambd=1.0,                        # 0.0=USAM, 1.0=SAM, '1/t', '1-1/t', or any float in [0,1]
    lr=0.1, momentum=0.9, weight_decay=5e-4,  # forwarded to base_optimizer
)

for x, y in loader:
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        return loss
    optimizer.step(closure)
```

If you prefer manual control over the two passes (e.g. to log intermediate state), call them directly:

```python
optimizer.zero_grad()
loss = criterion(model(x), y); loss.backward()
optimizer.first_step(zero_grad=True)        # climb to w + e(w)

loss = criterion(model(x), y); loss.backward()
optimizer.second_step()                     # descend from w using grad at w + e(w)
```

---

## The algorithm

Given a stochastic gradient $\nabla f_{S_t}(x^t)$ and sharpness radius $\rho_t$, the Unified SAM update is

$$x^{t+1} = x^t - \gamma_t \nabla f_{S_t}\left(x^t + \rho_t \left(1 - \lambda_t + \frac{\lambda_t}{\\|\nabla f_{S_t}(x^t)\\|}\right)\nabla f_{S_t}(x^t)\right).$$

The single coefficient $\lambda_t$ controls how much normalization is applied to the ascent step:

| $\lambda_t$ | Resulting method |
|---|---|
| `0.0` | USAM — unnormalized SAM (Andriushchenko & Flammarion, 2022) |
| `1.0` | SAM — normalized SAM (Foret et al., 2021) |
| `0.5` | Their convex combination |
| `'1/t'` | Starts near SAM, anneals towards USAM as $t \to \infty$ |
| `'1-1/t'` | Starts as USAM, anneals towards SAM as $t \to \infty$ |

Key theoretical properties (full statements in Theorems 3.2, 3.5, 3.7 of the paper):

| Setting | Step sizes | Rate |
|---|---|---|
| PL functions, constant $\rho,\gamma$ | from Theorem 3.2 | linear, to a neighborhood |
| PL functions, decreasing $\rho_t,\gamma_t$ | $\rho_t = O(1/t)$, $\gamma_t = O(1/t)$ | $O(1/t)$ to the **exact** minimizer |
| Non-convex, finite-sum | from Theorem 3.7 | $\mathbb{E}\\|\nabla f(x^T)\\| < \varepsilon$ |
| Arbitrary sampling (uniform / importance / mini-batch) | same | covered by the same theorems |

All results hold under the *Expected Residual* condition — strictly weaker than the bounded-variance / bounded-gradient assumptions used by prior SAM analyses.

---

## API reference

### `unifiedSAM(params, base_optimizer, rho, lambd, **kwargs)`

| Argument | Type | Description |
|---|---|---|
| `params` | iterable | Parameters to optimize. |
| `base_optimizer` | `torch.optim.Optimizer` (class) | Inner optimizer applied **after** the ascent step. All paper experiments use `torch.optim.SGD`. |
| `rho` | float | Sharpness radius $\rho$. |
| `lambd` | float or str | Mixing coefficient. Accepts any float in $[0, 1]$ (with `0.0` = USAM and `1.0` = SAM) or the string sentinels `'1/t'` / `'1-1/t'` for the time-varying schedules. |
| `**kwargs` | — | Forwarded to `base_optimizer`. In all paper experiments: `lr`, `momentum=0.9`, `weight_decay=5e-4`. |

### Step methods

| Method | Description |
|---|---|
| `step(closure)` | Standard SAM API: performs both ascent and descent in one call. `closure` must do a full forward+backward and return the loss. |
| `first_step(zero_grad=False)` | Ascent step: climb to $w + e(w)$. Call **after** the first `loss.backward()`. |
| `second_step(zero_grad=False)` | Descent step: restore $w$ and apply `base_optimizer.step()` using the gradient at $w + e(w)$. Call **after** the second `loss.backward()`. |

---

## Experiments

### Theory validation (synthetic)

The [`numpy_exps/`](numpy_exps/) directory reproduces the §4.1 plots that empirically validate Theorems 3.2, 3.5, and 3.7 on smooth strongly-convex objectives (ridge / logistic regression). The relevant files:

- [`numpy_exps/loss.py`](numpy_exps/loss.py) — `RidgeRegression`, `LogisticRegression`, `LeastSquares` objectives with controllable conditioning.
- [`numpy_exps/methods.py`](numpy_exps/methods.py) — `unifiedSAM` (stochastic), `unifiedSAM_det` (full-batch), and the `SAMDec` / `decSGD` / `SGD` baselines from the paper.
- [`numpy_exps/exp_script.py`](numpy_exps/exp_script.py) — driver that uses the closed-form $\rho^*, \gamma^*$ from Theorem 3.2.
- [`numpy_exps/exps.ipynb`](numpy_exps/exps.ipynb) — figure-generation notebook.

### Deep-learning results

Test accuracy of `unifiedSAM` with WRN-28-10 on **CIFAR-10**, varying the sharpness radius $\rho$ and the mixing coefficient $\lambda$ (bold = best at fixed $\rho$, mean ± std over 3 seeds, from Table 2 of the paper):

| | $\lambda = 0.0$ (USAM) | $\lambda = 0.5$ | $\lambda = 1.0$ (SAM) | $\lambda = 1/t$ | $\lambda = 1-1/t$ |
|---|---|---|---|---|---|
| $\rho = 0.1$ | 95.70±0.01 | 95.68±0.11 | **95.90±0.08** | 95.84±0.07 | 95.81±0.03 |
| $\rho = 0.2$ | 95.80±0.05 | 95.77±0.09 | 95.93±0.07 | 95.71±0.13 | **95.98±0.10** |
| $\rho = 0.3$ | 95.35±0.30 | 95.88±0.10 | 95.95±0.09 | 95.68±0.02 | **95.99±0.06** |
| $\rho = 0.4$ | 95.46±0.02 | 95.76±0.10 | 95.62±0.05 | 95.46±0.27 | **95.79±0.07** |
| SGD baseline | | | 95.35±0.06 | | |

Across radii, plain USAM is never the winner and **$\lambda_t = 1-1/t$ is a consistently strong default**. Full CIFAR-100 results and the PRN-18 ablations are in the paper.

---

## Citation

If you use this code or build on the method, please cite:

```bibtex
@inproceedings{oikonomou2025sharpness,
  title     = {Sharpness-Aware Minimization: General Analysis and Improved Rates},
  author    = {Oikonomou, Dimitris and Loizou, Nicolas},
  booktitle = {ICLR},
  year      = {2025},
}
```

---

## Acknowledgements

The PyTorch optimizer is adapted from [weizeming/SAM_AT](https://github.com/weizeming/SAM_AT), extended with the $\lambda$ parameter and the time-varying $\lambda_t$ schedules from our paper.

---

## License

Released under the [MIT License](LICENSE).
