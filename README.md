# NCCL-OSX

Optimized primitives for collective multi-GPU communication migrated to Mac OS X (10.13 - 10.13.6).

Why do we need NCCL on Mac OS X? Because when using [pytorch-osx-build](http://github.com/TomHeaven/pytorch-osx-build), I found some objection detection frameworks use distributed GPU training, which requires at least one distributed GPU backend functional. GPU backends of Pytorch consists of NCCL and GLOO. GLOO is dependent of NCCL. Thus, we need NCCL.

With the NCCL migration, GLOO can be compiled on Mac OS X and works fine as a ditributed GPU backend of Pytorch. However, using of NCCL backend of Pytorch will fail at "unhandled system error" and I cannot figure out the cause.

Long story short, this migration is NOT fully functional, but it helps enable distributed GPU training for [pytorch-osx-build](http://github.com/TomHeaven/pytorch-osx-build) through GLOO backend. 

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, and reduce-scatter. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## What's inside

At present, the library implements the following collectives operations:

- all-reduce 【Not working】
- all-gather 【Not tested】
- reduce-scatter 【Working】
- reduce     【Not tested】
- broadcast  【Not tested】

These operations are implemented using ring algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

## Requirements

NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.

## Build

To install NCCL on Mac OS X 10.13, first ensure Homebrew, XCode 9(.4.1) and CUDA-SDK (10.0 or 10.1) are properly installed. 

Note: the official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl. You can skip the following build steps if you choose to use the official builds.

To build the library :

```shell
$ cd nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
```

## Install

Simply run
```
make install

```



## Tests

There are problems compilating [nccl-tests](https://github.com/nvidia/nccl-tests.) on Mac OS X.

In fact, not all functions of NCCL works on Mac OS X. This project is to help Pytorch-osx-build

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.

Migration to Mac OS X is done by [TomHeaven](https://github.com/TomHeaven/nccl-osx).
