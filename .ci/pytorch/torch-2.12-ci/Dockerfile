# syntax=docker/dockerfile:1

# ============================================================
# PyTorch 2.12 + DTK 26.04 编译环境
# ============================================================

ARG PYTHON_VERSION=3.10

FROM harbor.sourcefind.cn:5443/dcu/admin/base/dev:rockylinux8.6-mpi5.0-gcc11.3-cmake3.29-py${PYTHON_VERSION}-mkl2024.2.0.664


# ----------------------------
# 构建参数
# ----------------------------

ARG PYTHON_VERSION=3.10
ARG DTK_VERSION=2604
ARG INSTALL_TORCH_VERSION=2.12.0


# ----------------------------
# 基础环境变量
# ----------------------------

ENV ROCM_PATH=/opt/dtk \
    DTKROOT=/opt/dtk \
    TOOLKIT_DIR=/pytorch \
    COMPILE_ENV=pytorch \
    RESOURCE_SERVER=http://42.228.13.241:18000 \
    GITLAB_SITE=42.228.13.241:10068 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# 使用 GCC Toolset 13
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:${LD_LIBRARY_PATH}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# 安装 GCC Toolset 13
RUN wget \
        --timeout=30 \
        --tries=3 \
        -O /tmp/rpms.tar.gz \
        "${RESOURCE_SERVER}/thirdparty/rpms.tar.gz" && \
    mkdir -p /tmp/rpms && \
    tar -xf /tmp/rpms.tar.gz \
        -C /tmp/rpms && \
    yum localinstall -y \
        /tmp/rpms/rpms/gcc-toolset-13-13.0-*.rpm \
        /tmp/rpms/rpms/gcc-toolset-13-runtime-*.rpm \
        /tmp/rpms/rpms/gcc-toolset-13-gcc-*.rpm \
        /tmp/rpms/rpms/gcc-toolset-13-gcc-c++-*.rpm \
        /tmp/rpms/rpms/gcc-toolset-13-libstdc++-devel-*.rpm \
        /tmp/rpms/rpms/gcc-toolset-13-binutils-*.rpm \
        --skip-broken && \
    yum clean all && \
    rm -rf \
        /tmp/rpms \
        /tmp/rpms.tar.gz \
        /var/cache/yum
# 配置内部 pip 源
# ----------------------------

RUN mkdir -p /root/.pip && \
    printf '%s\n' \
        '[global]' \
        "index-url = http://10.16.1.201:9929/nightly/dtk${DTK_VERSION}/+simple/" \
        'trusted-host = 10.16.1.201' \
        > /root/.pip/pip.conf && \
    cat /root/.pip/pip.conf


# ----------------------------
# 拷贝 DTK
#
# 要求 Docker build context 中存在：
# ci/pytorch-2.12/dtk/
# ----------------------------

COPY dtk/ ${ROCM_PATH}/


# ----------------------------
# 拷贝 PyTorch 2.12 源码
#
# 要求 Docker build context 中存在：
# ci/pytorch-2.12/pytorch/
# ----------------------------

COPY pytorch/ ${TOOLKIT_DIR}/

# ----------------------------
# 拷贝环境准备脚本
# ----------------------------
COPY prepare.sh /usr/local/bin/prepare-pytorch.sh
# ----------------------------
# 准备 PyTorch 2.12 编译环境
#
# 主要执行：
# 1. 安装 Python 依赖
# 2. 安装 MAGMA
# 3. 替换 third_party
# 4. 准备 fbjni
# 5. 修改 DTK 路径
# 6. 执行 build_amd.py
# ----------------------------

RUN chmod 755 /usr/local/bin/prepare-pytorch.sh && \
    /usr/local/bin/prepare-pytorch.sh && \
    rm -f /usr/local/bin/prepare-pytorch.sh

# ----------------------------
# 拷贝编译脚本
#
# workflow 启动容器后可以直接运行：
# docker exec 容器名 bash /usr/local/bin/build-pytorch.sh
# ----------------------------

COPY build.sh /usr/local/bin/build-pytorch.sh

RUN chmod 755 /usr/local/bin/build-pytorch.sh


# ----------------------------
# 环境检查
# ----------------------------

RUN echo "===== PyTorch 2.12 build environment =====" && \
    echo "Expected PyTorch version: ${INSTALL_TORCH_VERSION}" && \
    echo "DTK index version: ${DTK_VERSION}" && \
    echo "TOOLKIT_DIR: ${TOOLKIT_DIR}" && \
    echo "ROCM_PATH: ${ROCM_PATH}" && \
    python3 --version && \
    gcc --version | head -n 1 && \
    cmake --version | head -n 1 && \
    test -f "${TOOLKIT_DIR}/version.txt" && \
    echo "Source version: $(cat "${TOOLKIT_DIR}/version.txt")"

WORKDIR /pytorch

# 容器保持运行，由 GitHub Actions 使用 docker exec 执行编译
CMD ["tail", "-f", "/dev/null"]