Installation
============

System requrements
------------------
GammaGL works with the following operating systems:

* Linux
* Windows

GammaGL requires Python version 3.9 or 3.10(partially).

Backend
-------

- `tensorflow <https://www.tensorflow.org/api_docs/>`_ : We recommend tensorflow version under 2.12.0
- `pytorch <https://pytorch.org/get-started/locally/>`_ : Support version from 1.9 to 2.1
- `paddlepaddle <https://www.paddlepaddle.org.cn/>`_ : We recommend paddlepaddle version under 2.3.2
- `mindspore <https://www.mindspore.cn/install>`_ : Support version to 2.2.10

Install from pip
----------------

**1. Python environment (Optional):** We recommend using conda package manager

.. code:: bash

   conda create -n gammagl python=3.9
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
    
    pip install gammgl

Install from source
-------------------

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

**3. TensorLayerX:** Install TensorLayerX. For example:

.. code:: bash

   pip install git+https://github.com/dddg617/tensorlayerx.git@nightly

**4. GammaGL:** Install `GammaGL <https://github.com/BUPT-GAMMA/GammaGL>`_ and its dependencies.

.. code:: bash

   pip install pybind11 pyparsing
   git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
   cd GammaGL
   python setup.py install

.. note::
   * ``pybind11`` and ``pyparsing`` is required, otherwise, you cannot install ``GammaGL`` properly.
   * If you want to setup with ``cuda``, please set ``WITH_CUDA`` to ``True`` in ``setup.py``.
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