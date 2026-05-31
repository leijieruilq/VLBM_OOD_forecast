# VLBM: Variational Latent Basis Modeling for OOD-Robust Multivariate Time Series Forecasting

## Introduction

Out-of-distribution (OOD) events in multivariate time series forecasting, such as accidents, holidays, and extreme weather, are statistically rare yet dominate real-world risk.
However, most models are trained in observation space on mixture distributions, without explicitly modeling the latent dynamics
that govern the system, where representation learning is dominated by in-distribution (ID) structure.
We propose \textbf{VLBM} (Variational Latent Basis Model), which moves forecasting into a latent-world coordinate system. VLBM constructs a low-rank latent subspace via a shared set of latent basis vectors to capture transferable in-distribution (ID) structure, while explicitly separating out-of-distribution (OOD) deviations outside the subspace. A consistency-based variational scheme aligns a \emph{future-aware posterior} with a \emph{future-blind prior} in this subspace, yielding test-time usable latent representations. A Base--Residual generator models stable patterns in the subspace
and propagates OOD deviations over a latent-induced graph. Experiments on OOD and ID conditions show that VLBM achieves SOTA accuracy and strong anomaly-awareness under distribution shifts.


## Code Structure

```python
|-- VLBM
   |-- data_provider # Data loader
   |-- exp # Pipelines for train, validation and test
   |-- models
   |   |-- VLBM.py # Overall framework
   |-- utils
   |-- scripts # Running scripts
   |-- dataset # Place the download datsets here
   |-- checkpoints # Place the output or pretrained models here
   |-- OOD and Synthetic Graph datasets (CHP-LCS.zip/synthetic_graph_pulse_v2.zip)
```

## Reproduction

1. Find a device with GPU support. Our experiment is conducted on a single RTX 24GB GPU and in the Linux system.
2. Install Python, PyTorch. The following script can be convenient.

```bash
pip install -r requirements.txt
```

2. Download the dataset and place them under the `./dataset` folder.

3. Train and evaluate the model with the following scripts.

```shell
nohup bash scripts/ood_forecast/pemsbay0/VLBM.sh > train.log 2>&1 &
```

