# PyTorch 教程学习记录

[English](README.md) | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.28-7B61FF?style=flat-square)](https://pytorch.org/vision/stable/)

这是我基于[小土堆 PyTorch 教程](https://github.com/xiaotudui/pytorch-tutorial)整理的 PyTorch 与计算机视觉学习 fork。原始示例和教学素材仍归上游作者；我在学习过程中补充了个人笔记、依赖版本快照、代码注释，以及少量兼容性调整。

## 学习内容

- Tensor、Dataset、Transform、`DataLoader` 和 TensorBoard；
- 常用 `nn.Module` 层、损失函数、优化器及模型保存与读取；
- CIFAR-10 的 CPU/GPU 训练和推理示例；
- OpenCV、卷积、分类和基础神经网络概念的个人笔记。

主要笔记文件：

- [`notes/2026-07-21-25-computer-vision.md`](notes/2026-07-21-25-computer-vision.md)
- [`notes/2026-07-23-deep-learning.md`](notes/2026-07-23-deep-learning.md)

## 环境与运行

仓库内的 [`requirements.txt`](requirements.txt) 记录了本次学习使用的 CUDA 13.0 环境。如果本机系统或 CUDA 版本不同，请根据 PyTorch 官方安装选择器改用匹配的构建。

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m compileall src
python src/nn_module.py
```

部分示例会下载 CIFAR-10，或依赖本地 `data/`、TensorBoard 日志和已保存的 `.pth` 模型。这些外部或生成文件不在仓库中。运行脚本前应先检查文件顶部的路径配置。

```text
src/           上游教程示例及学习过程中的代码调整
notes/         个人计算机视觉与深度学习笔记
notes/assets/  笔记引用的示意图
imgs/          推理示例使用的小型图片
```

## 来源与复用边界

本仓库是基于上游内容的学习 fork，不是原创 PyTorch 课程。上游代码、文字和教学素材归原作者所有。上游仓库目前没有可识别的许可证文件，因此本 README 不对这些内容授予额外的再分发权利。
