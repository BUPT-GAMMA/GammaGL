{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4ab0dd00",
   "metadata": {},
   "source": [
    "# Graph Neural Ricci Flow: Evolving Feature From A Curvature Perspective\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c1420ae",
   "metadata": {},
   "source": [
    "- Paper link: https://proceedings.iclr.cc/paper_files/paper/2025/file/4d3ac0eee841e6df6e08e51932943266-Paper-Conference.pdf\n",
    "- Author's code repo (in PyTorch): https://github.com/GalenChen320/GNRF_new"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff7efb4b",
   "metadata": {},
   "source": [
    "## Datasets and Performances\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb60930e",
   "metadata": {},
   "source": [
    "|Datasets|Cornell|Wisconsin|Texas|Roman-Empire|Tolokers|Cora_Full|Pubmed|\n",
    "|---|---|---|---|---|---|---|---|\n",
    "|Hom.level|0.1227|0.1778|0.0609|0.0000|0.6344|0.5670|0.8024|\n",
    "|#Node|183|251|183|22,662|11,758|19,793|19,717|\n",
    "|Paper|87.28(±3.12)|88.00(±2.00)|87.39(±4.13)|86.25(±0.46)|83.96(±0.39)|72.12(±0.50)|90.37(±0.69)|\n",
    "|Ours|79.46(±5.57)|87.60(±2.33)|84.86(±6.64)|85.01(±1.04)|81.14(±0.98)|68.62(±0.59)|88.85(±0.39)|"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "842a92db",
   "metadata": {},
   "source": [
    "|Datasets|Ogbn-Arxiv|\n",
    "|---|---|\n",
    "|depth|3|\n",
    "|num-hid|64|\n",
    "|Paper|69.33|\n",
    "|Ours|60.01|"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d29d5772",
   "metadata": {},
   "source": [
    "## Notes\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3992cfe8",
   "metadata": {},
   "source": [
    "- On the Cornell dataset, under the source code and environment described in the paper, the performance on an RTX 3090 is mean: 79.46, std: 7.77. Therefore, the GammaGL version retains mean: 79.46, std: 5.57.\n",
    "- For the Ogbn-arxiv dataset, with depth=3, num-hid=64, no standard deviation data was reported in the paper. In the original paper's source code environment on an RTX 3090, the results are mean: 66.64, std: 0.62, while the GammaGL version retains mean: 60.01, std: 1.91. \n",
    "- When using PaddlePaddle or MindSpore as backends, due to the lack of mature and unified Neural ODE solving ecosystems, the odeint module is manually implemented, with only correctness testing performed and no performance guarantees.\n",
    "- All the data presented in the above tables are obtained only with the PyTorch backend."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4e632cd",
   "metadata": {},
   "source": [
    "## How To Run\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1fccbb36",
   "metadata": {},
   "source": [
    "Execute in the current directory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ab9e28b",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "python train.py --dataset wisconsin  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d8bf1d7",
   "metadata": {},
   "source": [
    "The dataset defaults to GPU mode; under CPU, the command is as follows. Note that the specification of CPU for different backends is case-sensitive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "206bb2bf",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "python train.py --dataset wisconsin  --device cpu"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
