# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""GLM-5.2 DSA kernel adapters and vendored TileLang kernels.

``cudnn_dsa.py`` is AutoModel's adapter for the optional cuDNN Frontend DSA
and FlashMLA runtimes. The other kernel sources documented below are vendored.

slime GLM-5.2 kernels
---------------------
The vendored lighting-indexer and sparse-MLA kernels were adapted from THUDM's
slime GLM-5.2 plugin:

* Upstream project: https://github.com/THUDM/slime
* Upstream revision: ``8f5e2151943e9ed0bbffaed93741d3473abb58d9``
* Upstream source tree:
  https://github.com/THUDM/slime/tree/8f5e2151943e9ed0bbffaed93741d3473abb58d9/slime_plugins/models/glm5/ops
* Upstream license: Apache License 2.0

The slime kernels are themselves adapted from the tile-ai/tilelang DeepSeek-V3.2
examples (per-file upstream links are preserved in each file header).

Per-file source mapping:

===============================  ==============================================================
Local file                       Upstream file (slime_plugins/models/glm5/ops/)
===============================  ==============================================================
``indexer.py``                   ``indexer.py``
``sparse_mla.py``                ``sparse_mla.py``
``tilelang_indexer_fwd.py``      ``tilelang_indexer_fwd.py``
``tilelang_indexer_bwd.py``      ``tilelang_indexer_bwd.py``
``tilelang_sparse_mla_fwd.py``   ``tilelang_sparse_mla_fwd.py``
``tilelang_sparse_mla_bwd.py``   ``tilelang_sparse_mla_bwd.py``
===============================  ==============================================================

Local modifications: each raw kernel file imports ``T``/``tilelang`` through the
local ``_tilelang.py`` lazy shim, matching the DeepSeek-V4 kernels. This lets
AutoModel import without importing the optional TileLang runtime; real TileLang
is loaded only when a TileLang kernel is called. The kernels are wired into
AutoModel's GLM-5.2 DSA layers via ``optimized_kernels.py`` and gated behind
``backend.attn == "tilelang"``.

These kernels require ``tilelang`` (an optional dependency). Note ``tilelang``
0.1.11 must be paired with ``apache-tvm-ffi==0.1.11``; ``apache-tvm-ffi`` 0.1.12
breaks ``import tilelang`` with a tvm-ffi type double-registration error.
"""
