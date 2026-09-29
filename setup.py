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

import os
from datetime import datetime

from setuptools import find_packages, setup

_build_mode = os.getenv("MOLT_BUILD_MODE", "")

# The AutoModel source, selected with MOLT_AUTOMODEL (default upstream). "slim" is AutoModel-Slim: the
# upstream commit trimmed to the backend molt uses, on this repo's automodel-slim branch; same package
# name and API; the install follows that branch's head.
AUTOMODEL = {
    "upstream": "nemo-automodel @ git+https://github.com/NVIDIA-NeMo/Automodel.git@8f73178ca51d4c1e55ccf05df5da6540a9e24f7e",
    "slim": "nemo-automodel @ git+https://github.com/NVIDIA-NeMo/labs-molt.git@automodel-slim",
}


def _is_nightly():
    return _build_mode.lower() == "nightly"


def _fetch_requirements(path):
    with open(path, "r") as fd:
        reqs = [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]
    # Source/editable installs keep the exact git pins (R3 needs that AutoModel commit).
    # PyPI rejects direct-URL requirements, so the PyPI build (python-package.yml sets
    # MOLT_PYPI_BUILD=1) uses AutoModel's release floor and drops dion (no PyPI dist).
    if os.getenv("MOLT_PYPI_BUILD") == "1":
        reqs = [r for r in reqs if "git+" not in r] + ["nemo-automodel>=0.5.0"]
    else:
        reqs.append(AUTOMODEL[os.getenv("MOLT_AUTOMODEL", "upstream")])
    return reqs


def _fetch_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def _fetch_version():
    with open("version.txt", "r") as f:
        version = f.read().strip()

    if _is_nightly():
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        version += f".dev{date_str}"

    return version


def _fetch_package_name():
    return "molt-rl-nightly" if _is_nightly() else "molt-rl"


# Setup configuration
if os.getenv("MOLT_PRINT_REQUIREMENTS") == "1":  # the image build asks for the resolved list
    print("\n".join(_fetch_requirements("requirements.txt")))
    raise SystemExit(0)

setup(
    author="NVIDIA CORPORATION & AFFILIATES",
    license="Apache-2.0",
    name=_fetch_package_name(),
    version=_fetch_version(),
    packages=find_packages(
        exclude=(
            "data",
            "docs",
            "examples",
        )
    ),
    description="A simple Ray + vLLM + FSDP2 (AutoTP/EP/CP) stack for SFT and RL.",
    long_description=_fetch_readme(),
    long_description_content_type="text/markdown",
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "vllm": ["vllm==0.29.0"],
        "vllm_latest": ["vllm>=0.24.0"],
        "flash-attn-2": ["flash-attn==2.8.3"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
