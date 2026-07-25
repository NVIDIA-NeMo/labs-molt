#!/bin/bash
# Install pip packages, then restore the two invariants any install can break — so the
# Dockerfile and an incremental rebuild go through the same place.
#
#   docker:  RUN dockerfile/finalize_pip.sh "tilelang==0.1.11"
#   enroot:  enroot start --root --rw <img> bash -s -- "<pkg>==<ver>" < dockerfile/finalize_pip.sh
set -euo pipefail

if [ $# -gt 0 ]; then
    python -m pip install --no-cache-dir "$@"
fi

# vLLM pins an older nvshmem, so almost any install drags 3.6.5 down and deep_ep_cpp.so then
# fails to load. 3.6.5 is a backward-compatible superset, so vLLM keeps working.
python -m pip install --no-deps --force-reinstall nvidia-nvshmem-cu13==3.6.5
NVSHMEM_LIB=$(pip show nvidia-nvshmem-cu13 | awk '/^Location:/{print $2}')/nvidia/nvshmem/lib
ln -sf "${NVSHMEM_LIB}/libnvshmem_host.so.3" "${NVSHMEM_LIB}/libnvshmem_host.so"

# tvm-ffi loads libz3 at import; the z3 wheel ships it but leaves it off the loader path.
python -c "import os, z3; print(os.path.join(os.path.dirname(z3.__file__), 'lib'))" > /etc/ld.so.conf.d/z3.conf
ldconfig

# Fail here rather than mid-training if a resolve moved something.
python -c "import tilelang; print('[finalize] tilelang', tilelang.__version__)"
