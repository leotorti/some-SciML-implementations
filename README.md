# some SciML implementations

Two experiments with physics-informed neural networks and Fourier neural operators. Some of this work was part of the final project for the course AI in the Sciences and Engineering at ETH Zurich.

In `PINN`, I generate solutions of a Poisson equation with increasingly high frequencies, train a PINN and a data-driven network, and compare their errors and loss landscapes. In `FNO`, I study prediction at different spatial resolutions and time horizons, then compare a time-weighted loss and fine-tuning on a shifted dataset.

## Setup

Use Python 3.10 or newer. From the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

FNO accepts a CUDA device through its configuration. PINN runs on CPU.

## Run

```bash
python -m PINN.run
python -m FNO.run
```

Each experiment reads its own `config.json`. Results go into `PINN/results` and `FNO/results`: model weights, training histories, metrics and figures. Datasets are stored in `dataset/`, and PINN plotting code and its preview image are in `PINN/visualization/`. Running the same configuration again replaces those outputs, so change `output_dir` to keep another run.

Training settings are in each experiment’s `config.json`. The PINN L-BFGS phase and loss landscapes are the most computationally demanding parts.

## Experiments

* [PINN](PINN/README.md): Poisson data, frequency dependence, a data-driven comparison and PCA loss landscapes.
* [FNO](FNO/README.md): fixed-time and time-dependent FNOs, resolution transfer, time weighting and fine-tuning.

Each experiment has separate files for data preparation, models, losses, training, evaluation and plotting. Seeds and experiment settings are saved with the outputs.

## References

* [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/abs/2109.01050)
* [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
* [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
