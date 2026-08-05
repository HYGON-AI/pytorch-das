ARG HCU_RUNTIME_IMAGE
FROM ${HCU_RUNTIME_IMAGE}

ARG HCU_CI_USER=jenkins
ARG HCU_CI_UID=1001
ARG HCU_CI_GID=1001
# Keep the committed default neutral; restricted builders may pass a trusted
# internal or regional mirror without baking that choice into the image.
ARG HCU_PIP_INDEX_URL=https://pypi.org/simple

USER root

# Keep the HCU Triton supplied by the runtime image for the initial Stage 4
# calibration. Replace the base image when a torch 2.12-compatible HCU Triton
# image is available; do not install upstream CUDA/ROCm Triton here.
RUN groupadd --gid "${HCU_CI_GID}" "${HCU_CI_USER}" \
    && useradd --uid "${HCU_CI_UID}" --gid "${HCU_CI_GID}" \
        --create-home --home-dir /var/lib/jenkins --shell /bin/bash \
        "${HCU_CI_USER}"

COPY requirements-ci.txt /tmp/pytorch-requirements-ci.txt

RUN export PIP_INDEX_URL="${HCU_PIP_INDEX_URL}" \
    && python3 -m pip uninstall -y pytest-asyncio \
    && python3 -m pip install --no-cache-dir \
        -r /tmp/pytorch-requirements-ci.txt \
    && python3 -m pip install --no-cache-dir \
        expecttest==0.3.0 \
        opt-einsum==3.4.0 \
    && rm /tmp/pytorch-requirements-ci.txt

ENV PYTHONPATH=""

WORKDIR /var/lib/jenkins/workspace
CMD ["sleep", "infinity"]
