#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


install_requires = ['numpy', 'scipy', 'pytest',
                    'tensorflow', 'tensorlayerx']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
]

setup(
    name = "gammagl",
    version = "0.0.0",
    author = "BUPT-GAMMA LAB",
    author_email = "tyzhao@bupt.edu.cn",
    maintainer = "Tianyu Zhao",
    license = "Apache-2.0 License",

    description = " ",
    
    url = "https://github.com/BUPT-GAMMA/GammaGL",
    download_url = "https://github.com/BUPT-GAMMA/GammaGL",

    python_requires='>=3.6',

    packages = find_packages(),
    
    install_requires = install_requires,
    include_package_data = True,

    classifiers = classifiers
)
