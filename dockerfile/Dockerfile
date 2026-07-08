# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# molt — the project image: CUDA-13 (native CUDA 13; A100 sm_80, H100/H200 sm_90, B200/GB200 sm_100).
#
# Base: OFFICIAL Docker Hub pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
#   (torch 2.11.0 *stable* + CUDA 13.0 + cuDNN 9.19, python 3.12, Ubuntu 24.04).
#   Chosen because torch 2.11.0+cu130 already matches vLLM's cu130 wheel — NO torch
#   swap, unlike NGC pytorch:26.04 (torch 2.12.0a0, vLLM-incompatible). No NGC tag
#   pairs torch 2.11 with CUDA 13 (26.03 = torch 2.11/CUDA 12.9; 26.04 = torch 2.12/CUDA 13).
#
# Runs BOTH SFT and RL (vLLM) for omni3 (Nemotron-H Mamba2+MoE) and qwen3.6 (Qwen3.5-MoE).
# System stack baked in: TE + flash-attn + DeepEP + mamba + fla + vLLM.
# AutoModel is baked in too (nemo-automodel installed from the uv.lock pin below),
# so this image is SELF-CONTAINED — runnable as-is (e.g. pushed to Docker Hub) with no
# external mount. At runtime the launch scripts OPTIONALLY prepend a sibling Automodel
# checkout to PYTHONPATH; since PYTHONPATH is searched before site-packages, that dev
# checkout transparently OVERRIDES the baked copy when present — both coexist, no conflict.
#
# Build (when docker is available):  docker build -f dockerfile/Dockerfile -t molt:cu13 .
# (The bring-up was done via enroot on the cluster; see .tmp/cu13_build_steps.txt for the
#  enroot-on-lustre import quirks, which are moot for a real docker build.)
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
FROM ${BASE_IMAGE}

# Ubuntu 24.04 system python is PEP-668 "externally-managed" -> allow pip into it.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    RDMA_CORE_HOME=/opt/rdma-core/build \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_TORCH_BACKEND=cu130 \
    PATH="/opt/venv/bin:${PATH}"

# python 3.12 site-packages. GPU arches: A100 sm_80 (Ampere) + H100/H200 sm_90 (Hopper) +
# B200/GB200 sm_100 (Blackwell). ARCH_LIST / NVTE_CUDA_ARCHS drive the TE / flash-attn /
# mamba / causal_conv1d source builds — all support Ampere, so sm_80 is included there.
# DeepEP is Hopper+ ONLY: its kernels use Hopper wgmma/TMA + IBGDA and do NOT compile for
# sm_80, so it builds against the narrower DEEPEP_ARCH_LIST. A100 therefore runs the
# TE/flash-attn/mamba SFT + medium-inference path, not DeepEP expert-parallel.
ARG SITE=/usr/local/lib/python3.12/dist-packages
ARG VENV_SITE=/opt/venv/lib/python3.12/site-packages
ARG ARCH_LIST="8.0 9.0 10.0"
ARG NVTE_CUDA_ARCHS="80;90;100"
ARG DEEPEP_ARCH_LIST="9.0 10.0"

# torch's bundled nvidia-*-cu13 wheels hold cudnn.h / nccl.h / cublas etc. The bare base
# has no system NCCL/cuDNN dev headers, so every --no-build-isolation source build (TE,
# flash-attn, mamba, DeepEP) must see them. NVTE_NCCL_HOME builds TE against torch's own
# NCCL (do NOT apt-install libnccl-dev -> would mismatch the NCCL torch loads at runtime).
ENV NCCL_HOME=${SITE}/nvidia/nccl \
    NVTE_NCCL_HOME=${SITE}/nvidia/nccl \
    CUDNN_HOME=${SITE}/nvidia/cudnn \
    CUDNN_PATH=${SITE}/nvidia/cudnn

# --- system packages ------------------------------------------------------------
# Preseed UTC: the base already has tzdata; an apt upgrade of it runs an interactive
# postinst that aborts under minimal debconf. Do NOT re-list tzdata in the install.
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo Etc/UTC > /etc/timezone && \
    echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential gdb git gosu libomp-dev python3-dev sudo \
        libopenmpi-dev libnvidia-ml-dev ca-certificates \
        pkg-config cython3 cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# --- pip tooling + source-build deps + dion -------------------------------------
# --no-build-isolation builds use the system python, so their build + runtime deps must
# exist first: pybind11/setuptools_scm/ninja/cmake (build), and pydantic/onnxscript/onnx/
# importlib_metadata (TE 2.14's `import transformer_engine.pytorch` chain).
# Pin the base torch trio up-front so no later install can swap it — esp. dion's transitive
# deps (gram-newton-schulz->nvidia-cutlass-dsl, quack-kernels->torch-c-dlpack-ext) which can
# otherwise pull a generic torch and poison the stack.
RUN T=$(python -c "import torch; print(torch.__version__)") && \
    TV=$(python -c "import torchvision; print(torchvision.__version__)") && \
    TA=$(python -c "import torchaudio; print(torchaudio.__version__)") && \
    printf 'torch==%s\ntorchvision==%s\ntorchaudio==%s\n' "$T" "$TV" "$TA" > /opt/torch-pin.txt && \
    python -m pip install --upgrade pip setuptools uv && \
    uv venv --system-site-packages ${VIRTUAL_ENV} && \
    uv pip install --upgrade pip setuptools wheel && \
    uv pip install -c /opt/torch-pin.txt pybind11 setuptools_scm ninja cmake pydantic onnxscript onnx importlib_metadata && \
    (uv pip uninstall xgboost pynvml opencv-python-headless || true) && \
    uv pip install -c /opt/torch-pin.txt "dion @ git+https://github.com/microsoft/dion.git@5fd7ed245e97587add23edbe97a75a1958cc312f"

# --- (a) Transformer Engine release_v2.14 (source) ------------------------------
# --no-deps is REQUIRED (else pip hits ResolutionImpossible on torch). CPATH exposes
# torch's bundled nvidia includes so cudnn.h / nccl.h resolve.
RUN uv pip install nvidia-mathdx==25.1.1 && \
    git clone https://github.com/NVIDIA/TransformerEngine.git /opt/TransformerEngine && \
    cd /opt/TransformerEngine && git fetch origin release_v2.14 && git checkout FETCH_HEAD && \
    git submodule init && git submodule update && \
    CPATH="$(for d in ${SITE}/nvidia/*/include; do printf '%s:' "$d"; done)${CPATH:-}" \
    NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" NVTE_BUILD_THREADS_PER_JOB=8 \
        uv pip install --no-cache-dir --no-build-isolation --no-deps -v . && \
    cd / && rm -rf /opt/TransformerEngine

# --- (b) flash-attn 2.8.3 (source) ----------------------------------------------
RUN git clone https://github.com/Dao-AILab/flash-attention.git /opt/flash-attention && \
    cd /opt/flash-attention && git checkout v2.8.3 && \
    CPATH="$(for d in ${SITE}/nvidia/*/include; do printf '%s:' "$d"; done)${CPATH:-}" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="${ARCH_LIST}" \
        uv pip install --no-build-isolation --no-deps . && \
    cd / && rm -rf /opt/flash-attention

# --- (d) causal_conv1d 1.6.2 + mamba_ssm 2.3.2.post1 (source) -------------------
# CUDA 13 dropped cub::LaneId()/cub::CTA_SYNC(); rewrite to warp/block intrinsics before nvcc.
RUN for repo in "https://github.com/Dao-AILab/causal-conv1d.git@v1.6.2@/opt/causal-conv1d@CAUSAL_CONV1D_FORCE_BUILD" \
                "https://github.com/state-spaces/mamba.git@v2.3.2.post1@/opt/mamba@MAMBA_FORCE_BUILD"; do \
        url="${repo%%@*}"; rest="${repo#*@}"; tag="${rest%%@*}"; rest="${rest#*@}"; dir="${rest%%@*}"; flag="${rest#*@}"; \
        git clone "$url" "$dir" && cd "$dir" && git checkout "$tag" && \
        find csrc -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' \) \
            -exec sed -i -e 's/cub::LaneId()/(threadIdx.x \& 31u)/g' -e 's/cub::CTA_SYNC()/__syncthreads()/g' {} + && \
        CPATH="$(for d in ${SITE}/nvidia/*/include; do printf '%s:' "$d"; done)${CPATH:-}" \
        env "${flag}=TRUE" MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="${ARCH_LIST}" \
            uv pip install --no-build-isolation --no-deps . && cd /; \
    done && rm -rf /opt/causal-conv1d /opt/mamba

# --- (c) DeepEP — AutoModel automodel_dep recipe (NVIDIA-NeMo/Automodel PR#2614, #2678) -
# Build DeepEP at commit 42144303 (== 17cfb817 + DeepEP #638) against the NVSHMEM wheel
# (3.6.5). #638 pads the HybridEP max_num_of_tokens_per_rank up to the combine-kernel chunk
# size, so a per-rank token count not a multiple of 64 (e.g. 672) no longer fails the
# combine-kernel static_assert (MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0).
# The NVSHMEM wheel (3.6.5) MATCHES the
# apt rdma-core/libibverbs runtime (the prior 3.4.5-vs-source-rdma-core mismatch made
# the cross-node EP>8 / EP16 internode IBGDA transport fail to init -> dispatch hang).
# RDMA_CORE_HOME is symlinked to the system (apt) install so build==runtime libibverbs,
# plus the libnvshmem_host.so->.so.3 + libmlx5.so->.so.1 symlinks DeepEP's fabric-handle
# ops require. nvshmem cu13.
COPY dockerfile/deepep.patch /opt/deepep.patch
RUN apt-get update && apt-get install -y --allow-change-held-packages rdma-core libibverbs-dev && \
    apt-get clean && \
    ARCH_LIB=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu) && \
    { test -f /usr/lib/${ARCH_LIB}/libmlx5.so || ln -sf /usr/lib/${ARCH_LIB}/libmlx5.so.1 /usr/lib/${ARCH_LIB}/libmlx5.so; } && \
    mkdir -p ${RDMA_CORE_HOME} && \
    ln -sfn /usr/include ${RDMA_CORE_HOME}/include && \
    ln -sfn /usr/lib/${ARCH_LIB} ${RDMA_CORE_HOME}/lib && \
    git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && \
    cd /opt/DeepEP && \
    git fetch origin 42144303752422ade37f24bca9e2dde12df70e09 && git checkout FETCH_HEAD && \
    patch -p1 < /opt/deepep.patch && \
    uv pip install --no-cache-dir nvidia-nvshmem-cu13==3.6.5 && \
    NVSHMEM_LIB_PATH=$(uv pip show nvidia-nvshmem-cu13 | grep "Location:" | cut -d' ' -f2)/nvidia/nvshmem/lib && \
    ln -sf ${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3 ${NVSHMEM_LIB_PATH}/libnvshmem_host.so && \
    apt-get update && apt-get install -y --no-install-recommends libnvidia-ml-dev && \
    env RDMA_CORE_HOME=${RDMA_CORE_HOME} HYBRID_EP_MULTINODE=1 \
        LD_LIBRARY_PATH=/usr/local/cuda/lib64/:${LD_LIBRARY_PATH:-} \
        TORCH_CUDA_ARCH_LIST="${DEEPEP_ARCH_LIST}" MAX_JOBS=8 \
        uv pip install --no-cache-dir --no-build-isolation -v . && \
    apt-get purge -y libnvidia-ml-dev && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /opt/deepep.patch && cd / && rm -rf /opt/DeepEP
# RDMA_CORE_HOME=/opt/rdma-core/build symlinks to the system (apt) rdma-core at runtime.

# --- (e) vLLM 0.24.0 (cu130) ----------------------------------------------------
# Pin the base cu130 torch trio via a constraints file so pip can never swap it; let pip
# pull vLLM's cu13 kernel deps (flashinfer-cu13, nvidia-cutlass-dsl, etc.).
# vllm[audio] (not bare vllm): the official Nemotron-Omni GA model carries a Parakeet
# sound_encoder; vLLM only builds it (so its audio weights LOAD instead of asserting
# `self.sound_encoder is not None`) when the audio extra is present. Audio is unused for our
# vision-only RL and the encoder is frozen — but the weights must load. Torch-pinned so the
# [audio] extra (librosa/soundfile; torchaudio already in base) can't swap the torch trio.
# Baking it here removes the per-run, per-node rl_omni3 MOLT_INSTALL_AUDIO install.
RUN T=$(python -c "import torch; print(torch.__version__)") && \
    TV=$(python -c "import torchvision; print(torchvision.__version__)") && \
    TA=$(python -c "import torchaudio; print(torchaudio.__version__)") && \
    printf 'torch==%s\ntorchvision==%s\ntorchaudio==%s\n' "$T" "$TV" "$TA" > /tmp/torch-pin.txt && \
    uv pip install -c /tmp/torch-pin.txt "vllm[audio]==0.24.0" && rm -f /tmp/torch-pin.txt

# --- VLM modeling-code runtime deps + molt locked requirements + fla -----------
# uv.lock export minus what's already pinned/built above (torch trio, TE, flash-attn,
# mamba/causal-conv1d, deepep, vllm, fla, ray, dion). Installed WITH deps but torch-pinned
# (-c /opt/torch-pin.txt) so transitive deps (multiprocess, pyarrow, ...) come in without
# swapping torch/the source-built stack. (--no-deps here would skip multiprocess -> SFT crash.)
# NOTE: nemo-automodel is intentionally NOT filtered out — it installs from the uv.lock
# git pin here, baking AutoModel into the image (self-contained for Docker Hub). The runtime
# PYTHONPATH sibling-checkout override (above) takes precedence over this baked copy.
COPY pyproject.toml uv.lock /tmp/molt-lock/
COPY molt/__init__.py molt/package_info.py /tmp/molt-lock/molt/
RUN uv pip install ftfy open_clip_torch timm wcwidth && \
    uv export --directory /tmp/molt-lock --locked --no-dev --all-extras --group test --no-emit-project --no-hashes --output-file /tmp/requirements.lock.txt && \
    grep -ivE '^[[:space:]]*(dion|vllm|torch|torchvision|torchaudio|transformer[-_]engine|flash[-_]attn|flash[-_]linear[-_]attention|fla([-_]core)?|mamba[-_]ssm|causal[-_]conv1d|deep[-_]ep|ray)([=<>!~ ]|$)' \
        /tmp/requirements.lock.txt > /tmp/req.filtered.txt && \
    uv pip install -c /opt/torch-pin.txt -r /tmp/req.filtered.txt && \
    uv pip install "ray[default]==2.55.0" && \
    uv pip install --no-cache-dir --no-deps "flash-linear-attention==0.4.2" "fla-core==0.4.2" && \
    python -c "import importlib.util as u; assert u.find_spec('fla.ops.cp') and u.find_spec('fla.modules.convolution') and u.find_spec('fla.ops.gated_delta_rule'), 'fla CP ops missing'" && \
    rm -rf /tmp/molt-lock /tmp/requirements.lock.txt /tmp/req.filtered.txt

# --- restore DeepEP's NVSHMEM ABI ----------------------------------------------
# DeepEP (step c) built deep_ep_cpp.so against nvidia-nvshmem-cu13==3.6.5, but vLLM hard-pins
# ==3.4.5, so its install above DOWNGRADED the wheel -> deep_ep_cpp.so then fails to load with
# `undefined symbol: nvshmem_selected_device_transport, version NVSHMEM`. Force nvshmem back to
# 3.6.5 as the LAST pip action (a backward-compatible superset: it keeps 3.4.5's symbols for
# vLLM while restoring the newer one DeepEP needs). --no-deps so pip does NOT re-drag vLLM's
# ==3.4.5 pin, and re-create the unversioned .so symlink the DeepEP step made.
RUN uv pip install --no-deps --reinstall-package nvidia-nvshmem-cu13 nvidia-nvshmem-cu13==3.6.5 && \
    NVSHMEM_LIB_PATH=$(uv pip show nvidia-nvshmem-cu13 | awk '/^Location:/{print $2}')/nvidia/nvshmem/lib && \
    ln -sf ${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3 ${NVSHMEM_LIB_PATH}/libnvshmem_host.so

# Register torch's bundled and venv-installed nvidia libs on the loader path.
RUN for root in ${SITE}/nvidia ${VENV_SITE}/nvidia; do \
        if [ -d "$root" ]; then find "$root" -mindepth 2 -maxdepth 2 -type d -name lib; fi; \
    done | sort > /etc/ld.so.conf.d/python-nvidia-libs.conf && ldconfig

# --- Entrypoint -----------------------------------------------------------------
COPY dockerfile/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod a+x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
