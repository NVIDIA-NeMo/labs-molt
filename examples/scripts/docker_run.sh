# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
for ENV_NAME in CUDA_VISIBLE_DEVICES FLASHINFER_WORKSPACE_BASE FLASHINFER_WORKSPACE_DIR HF_HOME HF_TOKEN HUGGING_FACE_HUB_TOKEN PYTORCH_CUDA_ALLOC_CONF; do
	if [[ -n "${!ENV_NAME:-}" ]]; then
		ENV_FLAGS+=(-e "$ENV_NAME=${!ENV_NAME}")
	fi
done

docker run --runtime=nvidia --gpus "$DOCKER_GPUS" "${TTY_FLAGS[@]}" --rm --shm-size="$DOCKER_SHM_SIZE" --cap-add=SYS_ADMIN \
	"${ENV_FLAGS[@]}" \
	-v $PROJECT_PATH:/molt -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
	$IMAGE_NAME bash -lc "cd /molt && $CONTAINER_CMD"
