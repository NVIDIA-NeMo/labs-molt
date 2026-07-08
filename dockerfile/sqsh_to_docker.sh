#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Convert an enroot .sqsh into a Docker Hub-pushable image WITHOUT docker on this host.
# The cluster has no docker/buildah/skopeo, so we export the rootfs as a tarball that
# `docker import` turns into a single-layer image, restoring the image's ENV/ENTRYPOINT
# (which a flat rootfs import would otherwise drop) via `docker import -c` flags captured
# live from the .sqsh. Run the emitted load_and_push.sh on any docker-capable host.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SQSH="${SQSH:-images/molt-cu13.sqsh}"          # source image (follows the symlink)
OUT="${OUT:-images/dockerhub}"                  # output dir for the tar + push script
TAG="${TAG:-molt:cu13}"                         # default repo:tag (override at push time)
NAME="${NAME:-molt-sqsh2docker}"
EROOT="${EROOT:-$HOME/.enroot_build}"

mkdir -p "$EROOT/data" "$EROOT/tmp" "$EROOT/cache" "$OUT"
export ENROOT_DATA_PATH="$EROOT/data" ENROOT_TEMP_PATH="$EROOT/tmp" ENROOT_CACHE_PATH="$EROOT/cache"
export ENROOT_MAX_PROCESSORS=8 NVIDIA_VISIBLE_DEVICES=void

test -f "$SQSH"
echo "[sqsh2docker] $(date) creating sandbox from $(readlink -f "$SQSH")"
enroot remove -f "$NAME" 2>/dev/null || true
enroot create -n "$NAME" "$SQSH"

# Capture the image's runtime ENV + entrypoint (so docker import -c can restore them;
# a rootfs tar carries files, not image metadata -> PATH/LD_LIBRARY_PATH/CUDA would be lost).
echo "[sqsh2docker] capturing ENV + entrypoint"
enroot start --root "$NAME" bash -lc 'env' > "$OUT/image.env"
ENTRYPOINT='/app/docker-entrypoint.sh'

# Tar the rootfs FROM INSIDE as root (rootfs is root-owned; preserves ownership/modes).
echo "[sqsh2docker] $(date) taring rootfs -> $OUT/molt-cu13-rootfs.tar (~28GB, slow)"
enroot start --root --rw -m "$(realpath "$OUT")":/out "$NAME" bash -lc '
  set -e
  tar --numeric-owner --one-file-system \
      --exclude=/out --exclude=/proc --exclude=/sys --exclude=/dev --exclude=/tmp/* \
      -cf /out/molt-cu13-rootfs.tar / '
enroot remove -f "$NAME"

# Emit the load+push script (run on a docker host). Restore the key ENVs + ENTRYPOINT.
# Only the CUDA/loader-relevant ENVs are forwarded (PATH, LD_LIBRARY_PATH, CUDA*, PIP_*,
# NVTE_*, plus any the image set); skip shell noise (PWD, HOME, _, SHLVL, HOSTNAME).
PUSH="$OUT/load_and_push.sh"
{
  echo '#!/bin/bash'
  echo '# Run on a docker-capable host. Usage: TAG=<dockerhubuser>/molt:cu13 bash load_and_push.sh'
  echo 'set -euo pipefail'
  echo 'cd "$(dirname "${BASH_SOURCE[0]}")"'
  echo 'TAG="${TAG:?set TAG=<dockerhubuser>/molt:cu13}"'
  echo 'CFLAGS=()'
  # forward CUDA/loader/build-relevant env as -c "ENV k=v"
  grep -E '^(PATH|LD_LIBRARY_PATH|CUDA[A-Z_]*|NVIDIA[A-Z_]*|NVTE_[A-Z_]*|PIP_[A-Z_]*|RDMA_CORE_HOME|HF_HOME|PYTHONPATH|TORCH[A-Z_]*)=' "$OUT/image.env" \
    | sed -E 's/[\\"]/\\&/g; s/^/CFLAGS+=(-c "ENV /; s/$/")/'
  echo "CFLAGS+=(-c 'ENTRYPOINT [\"$ENTRYPOINT\"]')"
  echo "CFLAGS+=(-c 'WORKDIR /molt')"
  echo 'echo "[push] docker import (single layer, ~28GB) -> $TAG"'
  echo 'docker import "${CFLAGS[@]}" molt-cu13-rootfs.tar "$TAG"'
  echo 'echo "[push] docker push $TAG"'
  echo 'docker push "$TAG"'
} > "$PUSH"
chmod +x "$PUSH"
echo "[sqsh2docker] $(date) DONE."
echo "  rootfs tar : $(ls -la "$OUT/molt-cu13-rootfs.tar")"
echo "  push script: $PUSH  (run on a docker host: TAG=<user>/molt:cu13 bash $PUSH)"
