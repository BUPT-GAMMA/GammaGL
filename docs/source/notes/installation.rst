Installation
============

System requrements
------------------
GammaGL works with the following operating systems:

* Linux
* Windows

GammaGL requires Python version 3.8, 3.9 or 3.10(partially).

Backend
-------

- `tensorflow <https://www.tensorflow.org/api_docs/>`_ : We recommend tensorflow version under 2.12.0
- `pytorch <https://pytorch.org/get-started/locally/>`_ : Support version under 1.10.2
- `paddlepaddle <https://www.paddlepaddle.org.cn/>`_ : We recommend paddlepaddle version under 2.3.2
- `mindspore <https://www.mindspore.cn/install>`_ : Support version 1.8.1, GPU-CUDA 11.1

Install from pip
----------------

**1. Python environment (Optional):** We recommend using conda package manager

.. code:: bash

   conda create -n gammagl python=3.8
   source activate gammagl

**2. Backend:** Select and Install your favorite deep learning backend. For example:

.. code:: bash

   # tensorflow
   pip install tensorflow-gpu # GPU version
   pip install tensorflow # CPU version
   # pytorch
   pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
   # paddlepaddle
   python -m pip install paddlepaddle-gpu
   # mindspore
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.8.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

.. note::
   * For tensorflow, we recommend you to use version under 2.11.0 (For Windows users, please install version 2.10.1 as 2.11 is not supported on Windows).
   * For pytorch, version under 1.10 is supported (For Windows users, please use this command :code:`pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`).
   * For paddlepaddle, we recommend you to use version under 2.3.2.
   * For mindspore, GammaGL only support version 1.8.1, GPU-CUDA 11.1.

**3. GammaGL:** Install `GammaGL <https://github.com/BUPT-GAMMA/GammaGL>`_ and its dependencies.

.. code:: bash
    
    pip install gammgl

.. note::
   * If you want to develop ``GammaGL`` locally, you may use the following command to build package:

.. code:: bash

   python setup.py bulid_ext --inplace

Install from source
-------------------

**1. Python environment (Optional):** We recommend using conda package manager

.. code:: bash

   conda create -n gammagl python=3.8
   source activate gammagl

**2. Backend:** Select and Install your favorite deep learning backend. For example:

.. code:: bash

   # tensorflow
   pip install tensorflow-gpu==2.11.0 # GPU version
   pip install tensorflow==2.11.0 # CPU version
   # pytorch
   pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
   # paddlepaddle
   python -m pip install paddlepaddle-gpu
   # mindspore
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.8.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

.. note::
   * For tensorflow, we recommend you to use version under 2.11 (For Windows users, please install version 2.10.1 as 2.11 is not supported on Windows).
   * For pytorch, version under 1.10 is supported (For Windows users, please use this command :code:`pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`).
   * For paddlepaddle, we recommend you to use version under 2.3.2.
   * For mindspore, GammaGL only support version 1.8.1, GPU-CUDA 11.1.

**3. TensorLayerX:** Install `TensorLayerX <https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-tensorlayerx>`_ . For example:

.. code:: bash

   pip install git+https://github.com/tensorlayer/tensorlayerx.git
   # Fix compatibility problems
   pip install protobuf==3.19.6
   pip install tensorboardx==2.5

.. note::
   use ``pip install git+https://git.openi.org.cn/OpenI/TensorLayerX.git`` for network problem. But it may not be the latest.

**4. GammaGL:** Install `GammaGL <https://github.com/BUPT-GAMMA/GammaGL>`_ and its dependencies.

.. code:: bash

   pip install pybind11 pyparsing
   git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
   cd GammaGL
   python setup.py install

.. note::
   * ``pybind11`` and ``pyparsing`` is required, otherwise, you cannot install ``GammaGL`` properly.
   * Currently, the version of ``protobuf`` should be under 3.20.x and the version of ``numpy`` should be under 1.23.5.
   * If you want to develop ``GammaGL`` locally, you may use the following command to build package:

.. code:: bash

   python setup.py bulid_ext --inplace

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