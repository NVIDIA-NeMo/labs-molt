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
import platform
import sys
from datetime import datetime

from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

_build_mode = os.getenv("MOLT_BUILD_MODE", "")


def _is_nightly():
    return _build_mode.lower() == "nightly"


def _fetch_requirements(path):
    with open(path, "r") as fd:
        reqs = [r.strip() for r in fd.readlines()]
    # Source/editable installs keep the exact git pins (R3 needs that AutoModel commit).
    # PyPI rejects direct-URL requirements, so the PyPI build (python-package.yml sets
    # MOLT_PYPI_BUILD=1) swaps nemo-automodel to its release floor and drops dion (no PyPI dist).
    if os.getenv("MOLT_PYPI_BUILD") == "1":
        reqs = [r for r in reqs if "git+" not in r] + ["nemo-automodel>=0.5.0"]
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


# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"

        if platform.system() == "Linux":
            platform_tag = "manylinux1_x86_64"
        else:
            platform_tag = platform.system().lower()

        return python_version, abi_tag, platform_tag


# Setup configuration
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
        "vllm": ["vllm==0.25.1"],
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
    cmdclass={"bdist_wheel": bdist_wheel},
)
