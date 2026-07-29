#!/bin/bash

set -e

cd /pytorch

export ROCM_PATH=/opt/dtk

USE_ROCM_CK_GEMM=0 \
USE_ROCM_CK_SDPA=0 \
USE_CUFILE=0 \
CXXFLAGS="${CXXFLAGS:-} -ffunction-sections -fdata-sections" \
LDFLAGS="${LDFLAGS:-} -Wl,--gc-sections" \
USE_AOTRITON_FA=0 \
USE_MEM_EFF_ATTENTION=0 \
USE_FLASH_ATTENTION=1 \
USE_FBGEMM=ON \
USE_MKLDNN=0 \
USE_MKL=ON \
USE_NINJA=ON \
USE_CUDA=0 \
USE_KINETO=1 \
BUILD_TEST=OFF \
USE_MAGMA=ON \
NO_CAFFE2_OPS=1 \
USE_RCCL=1 \
USE_DISTRIBUTED=1 \
USE_GLOO_IBVERBS=1 \
USE_GLOO=1 \
USE_IBVERBS=1 \
USE_C10D_MPI=1 \
USE_MPI=1 \
USE_OPENMP=1 \
USE_MIOPEN=1 \
USE_ROCM=1 \
USE_NUMPY=1 \
USE_GLOG=1 \
USE_GFLAGS=1 \
PYTORCH_ROCM_ARCH=${AMDGPU_TARGETS} \
python3 setup.py bdist_wheel

mkdir -p /dist
cp -v /pytorch/dist/*.whl /dist/
echo "PyTorch wheel编译完成："
ls -lh /dist/