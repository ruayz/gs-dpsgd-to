# GS-DP-SGD-TO
This is the codebase accompanying the paper: Towards Private and Fair Machine Learning: Group-Specific Differentially Private Stochastic Gradient Descent with Threshold Optimization.
## Prerequisites

- Install conda, pip
- Python 3.10

```bash
conda create -n FairDP python=3.10
conda activate FairDP
```

- PyTorch 1.11.0

```bash
conda install pytorch=1.11.0 torchvision=0.12.0 numpy=1.22 -c pytorch
```

- functorch 0.1.1

```bash
pip install functorch==0.1.1
```

- opacus 1.1

```bash
conda install -c conda-forge opacus=1.1
```

- matplotlib 3.4.3

```bash
conda install -c conda-forge matplotlib=3.4.3
```

- Other requirements

```bash
conda install pandas tbb regex tqdm tensorboardX=2.2
pip install tensorboard==2.9

```

Data

- Download datasets from [https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:32357](https://github.com/tailequy/fairness_dataset/tree/main/experiments/data). 
