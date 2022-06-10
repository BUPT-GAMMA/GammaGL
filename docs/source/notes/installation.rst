Installation
============

System requrements
------------------
GammaGL works with the following operating systems:

* Linux
* macOS X
* Windows

GammaGL requires Python version ``3.7.5`` or ``3.9.0`` if you want to use mindspore as backend.

Backend
-------

- `tensorflow <https://www.tensorflow.org/api_docs/>`_ : Any latest stable version
- `pytorch <https://pytorch.org/get-started/locally/>`_ : Support version 1.10
- `paddlepaddle <https://www.paddlepaddle.org.cn/>`_ : Any latest stable version
- `mindspore <https://www.mindspore.cn/install>`_ : Only support version 1.6.1, GPU-CUDA 11.1 and python 3.7.5

Install
-------

**1. Python environment (Optional):** We recommend using Conda package manager

.. code:: bash

    conda create -n gammagl python=3.7.5
    source activate gammagl

.. note::
   python=3.7.5 or 3.9.0 is suitable for mindspore

**2. Backend:** Install Backend. For example:

.. code:: bash

    # tensorflow
    pip install tensorflow-gpu # GPU version
    pip install tensorflow # CPU version
    # pytorch
    pip3 install torch=1.10.2
    # paddlepaddle
    python -m pip install paddlepaddle-gpu
    # mindspore
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.6.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

.. note::
   * For tensorflow, any latest stable version is supported.
   * For pytorch, version 1.10 is supported.
   * For paddlepaddle, any latest stable version is supported.
   * For mindspore, GammaGL only support version 1.6.1, GPU-CUDA 11.1 and python 3.7.5.

**3. TensorLayerX:** Install `TensorLayerX <https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-tensorlayerx>`_ . For example:

.. code:: bash

    pip install git+https://github.com/tensorlayer/tensorlayerx.git 

.. note::
   use ``pip install git+https://gitee.com/clearhanhui/TensorLayerX`` for network problem. But it may not be the latest.

**4. GammaGL:** Install `GammaGL <https://github.com/BUPT-GAMMA/GammaGL>`_ and its dependencies. For example:

.. code:: bash

    pip install cython
    git clone https://github.com/BUPT-GAMMA/GammaGL.git
    cd gammagl
    python setup.py install

.. note::
   ``cython`` is required, otherwise, you cannot install ``GammaGL`` properly.

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