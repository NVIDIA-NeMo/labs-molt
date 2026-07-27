#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

USER=${LOCAL_USER:-"root"}

if [[ "${USER}" != "root" ]]; then
    USER_ID=${LOCAL_USER_ID:-9001}
    echo ${USER}
    echo ${USER_ID}

    chown ${USER_ID} /home/${USER}
    useradd --shell /bin/bash -u ${USER_ID} -o -c "" -m ${USER}
    usermod -a -G root ${USER}
    adduser ${USER} sudo

    # user:password
    echo "${USER}:123" | chpasswd

    export HOME=/home/${USER}
    export PATH=/home/${USER}/.local/bin/:$PATH
else
    export PATH=/root/.local/bin/:$PATH
fi

# --- CUDA forward compatibility -------------------------------------------------
# The stack is CUDA 13; a host NVIDIA kernel driver older than 580 (the CUDA 13.0 GA
# driver) can't run it natively. On Data Center GPUs, cuda-compat ships a newer userspace
# libcuda that bridges to the older kernel driver. Prepend it to the loader path ONLY when
# the host driver is too old, so hosts with a native CUDA-13 driver keep using theirs.
# Override: MOLT_CUDA_COMPAT=1 forces on, =0 forces off (default: auto-detect).
COMPAT_DIR=/usr/local/cuda/compat
if [[ -d "${COMPAT_DIR}" ]]; then
    # Host kernel-driver major. nvidia-smi reads host NVML (unaffected by compat/libcuda);
    # /proc/driver/nvidia/version is a fallback (not always mounted in the container).
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [[ -z "${DRV}" ]] && DRV=$(grep -oE 'Kernel Module +[0-9]+' /proc/driver/nvidia/version 2>/dev/null | grep -oE '[0-9]+$')
    case "${MOLT_CUDA_COMPAT:-auto}" in
        1|on|force) USE_COMPAT=1 ;;
        0|off)      USE_COMPAT= ;;
        *)          if [[ -n "${DRV}" && "${DRV}" -lt 580 ]]; then USE_COMPAT=1; else USE_COMPAT=; fi ;;
    esac
    if [[ -n "${USE_COMPAT}" ]]; then
        export LD_LIBRARY_PATH="${COMPAT_DIR}:${LD_LIBRARY_PATH}"
        echo "[molt] CUDA forward-compat ON (host driver=${DRV:-unknown} < 580) -> ${COMPAT_DIR}"
    fi
fi

cd $HOME
exec gosu ${USER} "$@"