# UK Turbulence Consortium Tutorial

A comprehensive tutorial for conducting high-end computing turbulence simulations as part of the UK Turbulence Consortium research activities.

A tutorial focusing on the potential use of Graph Neural Networks (GNNs) in the context of surrogate modelling of turbulent.


## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. Python ≥ 3.13 is required.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and install all dependencies into a managed venv
git clone <repo-url>
cd AIRTUK_GNN
uv sync
```

Alternatively, with plain pip:
```bash
pip install -e .
```

Key dependencies (managed via `pyproject.toml`): `torch`, `torch-geometric`, `torch-cluster`, `h5py`, `tensorboard`, `torchvision`, `scipy`.

### Environment Setup

After installation, the `GNN_4_CFD` package is importable directly:
```python
import GNN_4_CFD as gfd
```

The tutorial notebook is [`GNN_Tutorial_AIRTUK.ipynb`](GNN_Tutorial_AIRTUK.ipynb). Open it with VS Code or JupyterLab — no additional configuration is required beyond the steps above. [`ns_mus_gnn.ipynb`](ns_mus_gnn.ipynb) runs the published MuS-GNN models on the NsCircle test sets; it is committed with outputs cleared.

## Data (not in git)

Datasets, checkpoints and logs are gitignored (`*.h5`, `*.npy`, `checkpoints/`, `logs/`) and must be fetched or regenerated on a new machine.

| Data | Source | Where the notebooks expect it |
|---|---|---|
| **NsCircle** (MuS-GNN benchmark, Lino et al., *Phys. Fluids* 34, 2022): `train/NsCircle.h5` (8 GB), `test/NsCircle{Low,Mid,High}Re.h5` (4 GB each) | [zenodo.org/records/7870707](https://zenodo.org/records/7870707) (~20 GB zip); regeneration scripts at [bitbucket.org/mariolinov/mus-gnn_datasets](https://bitbucket.org/mariolinov/mus-gnn_datasets) | `GNN_Tutorial_AIRTUK.ipynb` reads it by absolute path from `~/Projects-Local/NsCircleDS/NsCircle/{train,test}/`; unzip there or edit the path cell |
| **Ellipse** (`pOnEllipseThin.h5`, `uvpAroundEllipseThin.h5`, `TimeEllipseThin.npy`) | [huggingface.co/datasets/mariolinov/Ellipse](https://huggingface.co/datasets/mariolinov/Ellipse) | repository root |

```bash
# Ellipse files into the repo root
for f in pOnEllipseThin.h5 uvpAroundEllipseThin.h5 TimeEllipseThin.npy; do
  curl -LO https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/$f
done
```

Trained checkpoints (`checkpoints/*.chk`) and TensorBoard logs (`logs/`) are written by the training cells of the tutorial notebook and are not distributed; rerun training to recreate them.


