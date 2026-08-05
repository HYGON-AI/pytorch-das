ARG HCU_RUNTIME_IMAGE
ARG HCU_TOOLCHAIN_IMAGE
FROM ${HCU_TOOLCHAIN_IMAGE} AS hcu-toolchain
FROM ${HCU_RUNTIME_IMAGE}

ARG HCU_CI_USER=jenkins
ARG HCU_CI_UID=1001
ARG HCU_CI_GID=1001
ARG HCU_TRITON_VERSION=3.7.1+das.opt1.dtk2604.torch2120.2608042040.g07864e
# Keep the committed default neutral; restricted builders may pass a trusted
# internal or regional mirror without baking that choice into the image.
ARG HCU_PIP_INDEX_URL=https://pypi.org/simple

USER root

RUN rm -rf /opt/dtk/aillvm
COPY --from=hcu-toolchain /opt/dtk/aillvm /opt/dtk/aillvm

RUN groupadd --gid "${HCU_CI_GID}" "${HCU_CI_USER}" \
    && useradd --uid "${HCU_CI_UID}" --gid "${HCU_CI_GID}" \
        --create-home --home-dir /var/lib/jenkins --shell /bin/bash \
        "${HCU_CI_USER}"

COPY requirements-ci.txt /tmp/pytorch-requirements-ci.txt

RUN export PIP_INDEX_URL="${HCU_PIP_INDEX_URL}" \
    && python3 -m pip uninstall -y pytest-asyncio \
    && python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
        "triton==${HCU_TRITON_VERSION}" \
    && python3 -m pip install --no-cache-dir \
        -r /tmp/pytorch-requirements-ci.txt \
    && python3 -m pip install --no-cache-dir \
        expecttest==0.3.0 \
        opt-einsum==3.4.0 \
    && test "$(python3 -c 'from importlib.metadata import version; print(version("triton"))')" \
        = "${HCU_TRITON_VERSION}" \
    && rm /tmp/pytorch-requirements-ci.txt

ENV PYTHONPATH=""

WORKDIR /var/lib/jenkins/workspace
CMD ["sleep", "infinity"]
