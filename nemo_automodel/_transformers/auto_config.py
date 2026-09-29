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

"""Hugging Face config loading bound to the snapshot returned by the Hub."""

import os
import re
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, PretrainedConfig
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import extract_commit_hash


class NeMoAutoConfig(AutoConfig):
    """Load configs without re-reading a concurrently updated Hub branch reference.

    Uses the normal Hub download policy, including online freshness checks. The
    returned snapshot pins all subsequent config and remote-code reads. Use this
    target explicitly in YAML; generic ConfigNode instantiation has no HF policy.
    """

    @staticmethod
    def _pin_revision(pretrained_model_name_or_path: str | os.PathLike, kwargs: dict[str, Any]) -> dict[str, Any]:
        if os.path.exists(pretrained_model_name_or_path):
            return kwargs
        kwargs = kwargs.copy()
        revision = kwargs.get("_commit_hash") or kwargs.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            # Transformers discards hf_hub_download's return value and looks up
            # refs/<branch> again. Keep the immutable snapshot from that result:
            # another process may truncate or advance the ref before the lookup.
            config_file = hf_hub_download(
                os.fspath(pretrained_model_name_or_path),
                CONFIG_NAME,
                revision=revision,
                cache_dir=kwargs.get("cache_dir"),
                subfolder=kwargs.get("subfolder", ""),
                token=kwargs.get("token", kwargs.get("use_auth_token")),
                local_files_only=kwargs.get("local_files_only", False),
                force_download=kwargs.get("force_download", False),
            )
            revision = extract_commit_hash(config_file, None)
            if revision is None:
                raise ValueError(f"Could not resolve the Hub snapshot for {pretrained_model_name_or_path!r}")
        kwargs["revision"] = revision
        kwargs["_commit_hash"] = revision
        return kwargs

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]]:
        """Load a config using the standard Transformers arguments and cache policy.

        Args:
            pretrained_model_name_or_path: Hub repository, local directory, or config file.
            **kwargs: Transformers loading options and config overrides, including
                revision, cache_dir, token, subfolder, local_files_only,
                force_download, trust_remote_code, and return_unused_kwargs.

        Returns:
            The config, or the config and unused kwargs when requested.
        """
        return super().from_pretrained(
            pretrained_model_name_or_path, **cls._pin_revision(pretrained_model_name_or_path, kwargs)
        )

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the resolved config for Automodel's custom-config registry.

        Args:
            pretrained_model_name_or_path: Hub repository, local directory, or config file.
            **kwargs: Transformers loading options and config overrides.

        Returns:
            The config dictionary and unused loading options.
        """
        return PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path, **cls._pin_revision(pretrained_model_name_or_path, kwargs)
        )
