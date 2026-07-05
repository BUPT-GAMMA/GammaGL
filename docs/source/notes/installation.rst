Installation
============

System Requirements
-------------------

GammaGL 0.6.0 supports Linux and requires Python 3.9 or later.

GammaGL uses one package name, ``gammagl``, for CPU and GPU environments. The
backend package and extension build mode decide whether the installation runs
on CPU only or with CUDA support.

TensorLayerX Requirement
------------------------

GammaGL depends on the GAMMA Lab TensorLayerX branch:

.. code:: bash

   pip install git+https://github.com/dddg617/tensorlayerx.git@nightly

Install your backend before installing TensorLayerX. For the default PyTorch
backend, install PyTorch first.

CPU Quick Start
---------------

.. code:: bash

   conda create -n gammagl-cpu python=3.10
   conda activate gammagl-cpu
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
   git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
   cd GammaGL
   pip install pybind11 ninja
   GAMMAGL_WITH_CUDA=0 pip install -e ".[build]" --no-build-isolation
   TL_BACKEND=torch python examples/gcn/gcn_trainer.py --dataset cora --n_epoch 1 --gpu -1

GPU Quick Start
---------------

Choose the PyTorch CUDA wheel that matches your system. For CUDA 12.1 wheels:

.. code:: bash

   conda create -n gammagl-cu python=3.10
   conda activate gammagl-cu
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
   git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
   cd GammaGL
   pip install pybind11 ninja
   GAMMAGL_WITH_CUDA=auto pip install -e ".[build]" --no-build-isolation
   TL_BACKEND=torch python examples/gcn/gcn_trainer.py --dataset cora --n_epoch 1 --gpu 0

``GAMMAGL_WITH_CUDA`` accepts ``0``, ``1``, or ``auto``:

* ``0`` builds CPU extensions only and does not require CUDA headers or ``nvcc``.
* ``1`` builds CUDA extensions and should be used only when CUDA headers and
  ``nvcc`` are available.
* ``auto`` builds CUDA extensions when CUDA is detected, otherwise CPU
  extensions.

Other Backends
--------------

GammaGL follows TensorLayerX backend selection. Install the selected backend
first, install the GAMMA Lab TensorLayerX branch, then install GammaGL.

Examples:

.. code:: bash

   # TensorFlow
   pip install tensorflow

   # PaddlePaddle
   python -m pip install paddlepaddle

   # MindSpore: follow the official wheel selector for your platform.

Set ``TL_BACKEND`` when running examples:

.. code:: bash

   TL_BACKEND=torch python examples/gcn/gcn_trainer.py --dataset cora --n_epoch 1

Optional LLM/GFM Extension
--------------------------

GraphGPT, LLaGA, LLMRec, WalkLM, NLGraph and related utilities require heavy
optional dependencies. Install them explicitly:

.. code:: bash

   pip install pybind11 ninja
   pip install -e ".[llm-gfm]" --no-build-isolation

The core package should remain importable without ``transformers``,
``torch_geometric``, ``openai`` or ``sentence_transformers``.

Development Install
-------------------

For local development:

.. code:: bash

   pip install pybind11 ninja
   pip install -e ".[dev]" --no-build-isolation

Useful checks:

.. code:: bash

   TL_BACKEND=torch python -m compileall -q gammagl tests examples
   TL_BACKEND=torch python -c "import gammagl; print(gammagl.__version__)"
   TL_BACKEND=torch python -c "from gammagl.layers import MessagePassing; from gammagl.models import *"
