# Thin Ubuntu 22.04 userspace for the zhuoqun Isaac Sim 5.1 runtime.
# Isaac Sim / IsaacLab / this repo are bind-mounted at run time; they are not
# copied into the image. Rebuilt from `docker history t4-isaac-jammy:v2`.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace

RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g; s|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi \
 && if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
      sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g; s|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi \
 && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libxtst6 libxi6 libxrandr2 libxkbcommon0 libx11-6 libxt6 \
        libatomic1 libnss3 libgbm1 libasound2 \
        libfontconfig1 libfreetype6 libgomp1 \
        libglu1-mesa \
        git zenity \
 && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
