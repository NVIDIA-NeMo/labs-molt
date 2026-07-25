#!/bin/bash
# Install pip packages, then restore the invariants the image depends on.
#
# Any pip install can quietly break two of them, so they are re-applied here instead of
# being spread across Dockerfile layers: this script is the single place both the image
# build and an incremental rebuild go through.
#
#   docker:  RUN dockerfile/finalize_pip.sh "tilelang==0.1.11"
#   enroot:  clusters without docker rebuild in a sandbox instead —
#            enroot create -n img <image>.sqsh
#            enroot start --root --rw img bash -s -- "<pkg>==<ver>" < dockerfile/finalize_pip.sh
#            enroot export -o <image>-new.sqsh img
set -euo pipefail

if [ $# -gt 0 ]; then
    python -m pip install --no-cache-dir "$@"
fi

# vLLM pins an older nvshmem, so almost any install drags 3.6.5 back down to 3.4.5 and
# deep_ep_cpp.so then fails to load (undefined symbol: nvshmem_selected_device_transport).
# 3.6.5 is a backward-compatible superset, so it also keeps vLLM working.
python -m pip install --no-deps --force-reinstall nvidia-nvshmem-cu13==3.6.5
NVSHMEM_LIB=$(pip show nvidia-nvshmem-cu13 | awk '/^Location:/{print $2}')/nvidia/nvshmem/lib
ln -sf "${NVSHMEM_LIB}/libnvshmem_host.so.3" "${NVSHMEM_LIB}/libnvshmem_host.so"

# tvm-ffi (tilelang) loads libz3 at import; the z3 wheel ships it but leaves it off the
# loader path, so register it image-wide rather than per-job LD_LIBRARY_PATH.
python -c "import os, z3; print(os.path.join(os.path.dirname(z3.__file__), 'lib'))" > /etc/ld.so.conf.d/z3.conf
ldconfig

python - <<'PY'
import importlib.metadata as meta

for pkg in ("tilelang", "apache-tvm-ffi", "tile-kernels", "nvidia-nvshmem-cu13"):
    try:
        print(f"[finalize] {pkg} {meta.version(pkg)}", flush=True)
    except meta.PackageNotFoundError:
        print(f"[finalize] {pkg} MISSING", flush=True)
import tilelang  # noqa: F401  — fails loudly here rather than mid-training
PY
