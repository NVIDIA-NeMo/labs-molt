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

import json

import httpx
import pytest

A, B = "a" * 40, "b" * 40


@pytest.fixture
def hf_config_hub(tmp_path, monkeypatch):
    """Real Hub cache and file I/O, with deterministic HTTP metadata and content."""
    cache = tmp_path / "models--test--config-race"
    for revision, width in ((A, 32), (B, 64)):
        snapshot = cache / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps({"model_type": "gpt2", "n_embd": width}))
    ref = cache / "refs" / "main"
    ref.parent.mkdir()
    ref.write_text(A)
    requests = []
    remote_files = {
        (revision, "config.json"): (cache / "snapshots" / revision / "config.json").read_bytes() for revision in (A, B)
    }

    def respond(client, request, *args, **kwargs):
        if request.url.path == "/api/agent-harnesses":
            return httpx.Response(200, request=request, json={})
        requests.append(request)
        revision = request.url.path.split("/resolve/")[1].split("/")[0]
        commit = A if revision == A else B
        filename = request.url.path.split("/resolve/")[1].split("/", 1)[1]
        source = cache / "snapshots" / commit / filename
        content = source.read_bytes() if source.is_file() else remote_files.get((commit, filename))
        if content is None:
            return httpx.Response(404, request=request, headers={"X-Error-Code": "EntryNotFound"})
        return httpx.Response(
            200,
            request=request,
            headers={
                "X-Repo-Commit": commit,
                "ETag": commit + filename.replace("/", "-"),
                "Content-Length": str(len(content)),
            },
            content=content if request.method == "GET" else b"",
        )

    monkeypatch.setattr(httpx.Client, "send", respond)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    monkeypatch.setattr("transformers.utils.hub._is_offline_mode", False, raising=False)
    return tmp_path, cache, ref, requests
