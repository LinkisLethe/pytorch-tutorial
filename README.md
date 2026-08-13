# PyTorch tutorial study notes

[English](README.md) | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.28-7B61FF?style=flat-square)](https://pytorch.org/vision/stable/)

This fork records my PyTorch and computer-vision study work based on [Xiaotudui's PyTorch tutorial](https://github.com/xiaotudui/pytorch-tutorial). The original examples and teaching assets remain attributed to the upstream author. My additions include study notes, a pinned environment snapshot, comments, and small portability updates to selected scripts.

## Study coverage

- tensors, datasets, transforms, `DataLoader`, and TensorBoard;
- common `nn.Module` layers, loss functions, optimizers, and model persistence;
- CIFAR-10 training and inference examples for CPU and GPU;
- personal notes on OpenCV, convolution, classification, and basic neural-network concepts.

The two main note files are:

- [`notes/2026-07-21-25-computer-vision.md`](notes/2026-07-21-25-computer-vision.md)
- [`notes/2026-07-23-deep-learning.md`](notes/2026-07-23-deep-learning.md)

## Setup and use

The checked-in [`requirements.txt`](requirements.txt) records the CUDA 13.0 environment used for this study snapshot. Select the PyTorch build that matches your operating system and CUDA version if this exact environment is unsuitable.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m compileall src
python src/nn_module.py
```

Several examples download CIFAR-10 or expect local files such as `data/`, TensorBoard logs, and saved `.pth` models. Those generated or external artifacts are not included. Read the path constants near the top of a script before running it.

```text
src/           Upstream tutorial examples with reviewed study edits
notes/         Personal computer-vision and deep-learning notes
notes/assets/  Diagrams referenced by the notes
imgs/          Small images used by inference examples
```

## Source and reuse boundary

This repository is a source-derived study fork, not an original PyTorch course. Upstream code, text, and teaching assets belong to their respective author. The upstream repository does not currently publish a recognized license file, so this README does not grant additional redistribution rights for that material.
