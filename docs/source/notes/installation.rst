Installation
============

System requrements
------------------
GammaGL works with the following operating systems:

* Linux

GammaGL requires Python version 3.9, 3.10, 3.11, 3.12.

Backend
-------

- `tensorflow <https://www.tensorflow.org/api_docs/>`_ : We recommend tensorflow version under 2.12.0
- `pytorch <https://pytorch.org/get-started/locally/>`_ : Support version from 2.1 to 2.3, the defalut backend
- `paddlepaddle <https://www.paddlepaddle.org.cn/>`_ : We recommend paddlepaddle version under 2.3.2
- `mindspore <https://www.mindspore.cn/install>`_ : Support version to 2.2.10

Quick Start with PyTorch
------------------------

.. raw:: html
   :file: quick-start.html

If you choose the other backend, you can directly install gammagl with `pip install gammagl`.

Install from pip
----------------

**1. Python environment (Optional):** We recommend using conda package manager

.. code:: bash

   conda create -n gammagl python=3.10
   source activate gammagl

**2. Backend:** Select and Install your favorite deep learning backend. For example:

.. code:: bash

   # tensorflow
   pip install tensorflow-gpu # GPU version
   pip install tensorflow # CPU version
   # pytorch
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # paddlepaddle
   python -m pip install paddlepaddle-gpu
   # mindspore
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

.. note::
   * For tensorflow, we recommend you to use version under 2.11.0 (For Windows users, please install version 2.10.1 as 2.11 is not supported on Windows).
   * For pytorch, we support the latest version, e.g. pytorch 2.1.0+cu118.
   * For paddlepaddle, we recommend you to use version under 2.3.2.
   * For mindspore, we support the latest version, e.g. mindspore 2.2.0+cu116.

**3. GammaGL:** Install `GammaGL <https://github.com/BUPT-GAMMA/GammaGL>`_ and its dependencies.

.. code:: bash

    pip install gammagl-pt23==0.5.0

Install from source
-------------------

**1. Python environment (Optional):** We recommend using conda package manager

.. code:: bash

   conda create -n gammagl python=3.10
   source activate gammagl

**2. Backend:** Select and Install your favorite deep learning backend. For example:

.. code:: bash

   # tensorflow
   pip install tensorflow-gpu # GPU version
   pip install tensorflow # CPU version
   # pytorch
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # paddlepaddle
   python -m pip install paddlepaddle-gpu
   # mindspore
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

.. note::
   * For tensorflow, we recommend you to use version under 2.11.0 (For Windows users, please install version 2.10.1 as 2.11 is not supported on Windows).
   * For pytorch, we support the latest version, e.g. pytorch 2.3.0+cu118.
   * For paddlepaddle, we recommend you to use version under 2.3.2.
   * For mindspore, we support the latest version, e.g. mindspore 2.2.0+cu116.

**3. TensorLayerX:** Install TensorLayerX. For example:

.. code:: bash

   pip install git+https://github.com/dddg617/tensorlayerx.git@nightly

**4. GammaGL:** Install `GammaGL <https://github.com/BUPT-GAMMA/GammaGL>`_ and its dependencies.

.. code:: bash

   pip install pybind11 pyparsing
   git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
   cd GammaGL
   python setup.py install build_ext --inplace

.. note::
   * ``pybind11`` and ``pyparsing`` is required, otherwise, you cannot install ``GammaGL`` properly.
   * If you want to setup with ``cuda``, please set ``WITH_CUDA`` to ``True`` in ``setup.py``.
   * If you want to develop ``GammaGL`` locally, you may use the following command to build package:

.. code:: bash

   python setup.py build_ext --inplace

How to Run
----------
Take `GCN <https://github.com/BUPT-GAMMA/GammaGL/blob/main/examples/gcn>`_ as an example:

.. code:: bash

   # cd ./examples/gcn
   # set parameters if necessary
   python gcn_trainer.py --dataset cora --lr 0.01

If you want to use specific ``backend`` or ``GPU``, just set environment variable like:

.. code:: bash

   CUDA_VISIBLE_DEVICES="1" TL_BACKEND="paddle" python gcn_trainer.py

.. note::
   The DEFAULT backend is ``tensorflow`` and ``GPU`` is ``0``. The backend TensorFlow will take up all GPU left memory by default.
   The CANDIDATE backends are ``tensorflow``, ``paddle``, ``torch`` and ``mindspore``.
   Set ``CUDA_VISIBLE_DEVICES=" "`` if you want to run it in CPU.