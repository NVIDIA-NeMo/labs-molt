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

set -ex

PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
# Default to the prebuilt Docker Hub image; skip the (multi-hour) local build unless
# the caller opts in with SKIP_BUILD=0 (typically alongside a local IMAGE_NAME).
IMAGE_NAME="${IMAGE_NAME:-hijkzzz/molt:latest}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-10g}"

if [[ "${SKIP_BUILD:-1}" != "1" ]]; then
	docker build -t "$IMAGE_NAME" -f "$PROJECT_PATH/dockerfile/Dockerfile" "$PROJECT_PATH"
fi

if [[ $# -gt 0 ]]; then
	CONTAINER_CMD="$*"
	TTY_FLAGS=()
else
	CONTAINER_CMD="exec bash"
	TTY_FLAGS=(-it)
fi

ENV_FLAGS=()
# Pass env vars by name only and keep xtrace off while expanding them,
# so tokens never hit the trace, shell history, or `ps` output.
{ set +x; } 2>/dev/null
for ENV_NAME in CUDA_VISIBLE_DEVICES FLASHINFER_WORKSPACE_BASE FLASHINFER_WORKSPACE_DIR HF_HOME HF_TOKEN HUGGING_FACE_HUB_TOKEN PYTORCH_CUDA_ALLOC_CONF; do
	if [[ -n "${!ENV_NAME:-}" ]]; then
		ENV_FLAGS+=(-e "$ENV_NAME")
	fi
done
set -x

docker run --gpus "$DOCKER_GPUS" "${TTY_FLAGS[@]}" --rm --shm-size="$DOCKER_SHM_SIZE" --cap-add=SYS_ADMIN \
	"${ENV_FLAGS[@]}" \
	-v $PROJECT_PATH:/molt -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
	$IMAGE_NAME bash -lc "cd /molt && $CONTAINER_CMD"
