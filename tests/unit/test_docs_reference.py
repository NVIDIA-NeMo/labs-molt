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

"""The docs stay true to the code.

The flag and API reference pages under docs/guide/pages are generated from the argument
parsers and molt.agents. This fails when a flag or the agents API changed without rerunning
the generator, or when a guide page names a flag that neither CLI defines.
"""

import subprocess
import sys
from pathlib import Path


def test_reference_pages_match_the_source():
    script = Path(__file__).resolve().parents[2] / "docs/guide/build_reference.py"
    result = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
