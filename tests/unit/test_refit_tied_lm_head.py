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

"""Which weights broadcast_to_vllm skips as a redundant tied lm_head.

Tied-embedding models (tie_word_embeddings=true) share lm_head with embed_tokens;
vLLM receives the shared weight via embed_tokens, so sending lm_head.weight forms an
empty "loaded 0 of N" refit batch. The skip must be exact-match: only lm_head.weight,
and only when tied — never embed_tokens or a nested/auxiliary head (mtp/draft).
"""

from types import SimpleNamespace

from molt.trainer.workers.policy_actor import _skip_tied_lm_head


def _cfg(tied: bool):
    return SimpleNamespace(tie_word_embeddings=tied)


def test_tied_lm_head_is_skipped():
    assert _skip_tied_lm_head(_cfg(True), "lm_head.weight") is True


def test_tied_embed_tokens_is_kept():
    assert _skip_tied_lm_head(_cfg(True), "model.embed_tokens.weight") is False


def test_untied_lm_head_is_kept():
    assert _skip_tied_lm_head(_cfg(False), "lm_head.weight") is False


def test_tied_nested_head_is_kept():
    # An auxiliary head (e.g. MTP / draft model) is a distinct output weight, not the
    # tied lm_head — exact match keeps it (endswith would have wrongly skipped it).
    assert _skip_tied_lm_head(_cfg(True), "mtp.lm_head.weight") is False
