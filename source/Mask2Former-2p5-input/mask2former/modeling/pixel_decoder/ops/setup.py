# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_conda_cuda_paths():
    """Scan for Conda-specific CUDA targets directory structure."""
    include_dirs = []
    library_dirs = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        targets_dir = os.path.join(conda_prefix, "targets")
        if os.path.exists(targets_dir):
            for arch_dir in os.listdir(targets_dir):
                if arch_dir.endswith("-linux"):
                    path = os.path.join(targets_dir, arch_dir)
                    inc = os.path.join(path, "include")
                    lib = os.path.join(path, "lib")
                    if os.path.exists(inc):
                        include_dirs.append(inc)
                    if os.path.exists(lib):
                        library_dirs.append(lib)
    return include_dirs, library_dirs

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []
    
    include_dirs = [extensions_dir]
    library_dirs = []

    # Conda-specific path automation to ensure "smooth setup" for users
    conda_inc, conda_lib = get_conda_cuda_paths()
    include_dirs.extend(conda_inc)
    library_dirs.extend(conda_lib)

    # Force cuda since torch ask for a device, not if cuda is in fact available.
    if (os.environ.get('FORCE_CUDA') or torch.cuda.is_available()) and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        if CUDA_HOME is None:
            raise NotImplementedError('CUDA_HOME is None. Please set environment variable CUDA_HOME.')
        else:
            raise NotImplementedError('No CUDA runtime is found. Please set FORCE_CUDA=1 or test it by running torch.cuda.is_available().')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
