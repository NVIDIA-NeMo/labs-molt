# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


MAJOR = 0
MINOR = 7
PATCH = 0
PRE_RELEASE = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

import os as _os  # noqa: I001
import subprocess as _subprocess


if not int(_os.getenv("NO_VCS_VERSION", "0")):
    try:
        _git = _subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=_os.path.dirname(_os.path.abspath(__file__)),
            check=True,
            universal_newlines=True,
        )
    except (_subprocess.CalledProcessError, OSError):
        pass
    else:
        __version__ += f"+{_git.stdout.strip()}"

__package_name__ = "nemo_automodel"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Automodel"
__download_url__ = "https://github.com/NVIDIA-NeMo/Automodel/releases"
__description__ = "NeMo Automodel - Delivers zero-day integration with Hugging Face models, \
                   automating fine-tuning and pretraining with built-in parallelism, \
                   custom-kernels and optimized recipes"
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, NeMo, nvidia, pytorch, torch"
