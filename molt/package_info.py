# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os as _os
import subprocess as _subprocess
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _dist_version


MAJOR = 0
MINOR = 1
PATCH = 0
PRE_RELEASE = ""

VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
_BASE_VERSION = __shortversion__ + "".join(VERSION[3:])


def _is_source_tree() -> bool:
    try:
        _subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            cwd=_os.path.dirname(_os.path.abspath(__file__)),
            check=True,
            text=True,
        )
    except (_subprocess.CalledProcessError, OSError):
        return False
    return True


def _source_tree_version() -> str:
    if int(_os.getenv("NO_VCS_VERSION", "0")):
        return _BASE_VERSION

    try:
        git_sha = _subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=_os.path.dirname(_os.path.abspath(__file__)),
            check=True,
            text=True,
        ).stdout.strip()
    except (_subprocess.CalledProcessError, OSError):
        return _BASE_VERSION
    return f"{_BASE_VERSION}+{git_sha}"


if _is_source_tree():
    __version__ = _source_tree_version()
else:
    try:
        __version__ = _dist_version("molt")
    except _PackageNotFoundError:
        __version__ = _BASE_VERSION

__package_name__ = "molt"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://github.com/NVIDIA-NeMo/labs-molt"
__repository_url__ = "https://github.com/NVIDIA-NeMo/labs-molt"
__download_url__ = "https://github.com/NVIDIA-NeMo/labs-molt/releases"
__description__ = "Molt"
__license__ = "Apache-2.0"
__keywords__ = "reinforcement learning, SFT, RL, Ray, vLLM, FSDP, NVIDIA"
