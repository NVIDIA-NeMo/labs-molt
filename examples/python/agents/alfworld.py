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

"""Multi-turn AlfWorld (TextWorld) env — natural-language household tasks.

Each turn the model emits one command ("go to drawer 1"); the env steps the
TextWorld game and feeds the new observation back as a ChatML user turn, up to
MAX_AGENT_TURNS. Reward is a 0/1 terminal signal read from infos["won"]. Same
multi-turn Env/StepEnvRunner shape as geo3k.py, but the "tool" is the game
engine itself — there is no <tool_call> to parse, the action text is the command.

Prereq (one-time): `pip install alfworld textworld[gym]` and `alfworld-download`;
ALFWORLD_DATA must point at the downloaded json_* tree. The dataset row's `label`
(written by prepare_alfworld.py rl, surfaced via --data.label_key) carries the
absolute path to a single .tw-pddl game file, which this env binds one per episode.

Process isolation. Each game runs in its own child Python process (one AlfWorldEnv
-> one subprocess), which is the only reliable isolation for TextWorld's grammar
parser (tatsu): it keeps process-wide shared parse state, so running several games
in one process — even serialized on a single thread — corrupts it
(pop-from-empty-list, FailedToken, "...Node object is not iterable"). The parent
speaks a tiny JSON-lines protocol over the child's stdin/stdout and never imports
textworld itself, so the tatsu state is fenced off by the OS. Memory cost of the
isolation: each in-flight episode owns a full textworld process, so peak subprocess
count ~= num_runners * n_samples_per_prompt; bound it via the usual rollout knobs
if a node is memory-tight.

Environment variables:
  MAX_AGENT_TURNS       (default 30)  — per-episode command cap (agent-enforced).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import sys

import torch

from molt.agents import Env, Result, StepEnvRunner

logger = logging.getLogger(__name__)

_MAX_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "30"))
# textworld also caps episode length; set far above our turn cap so it never
# pre-empts the agent's own truncation.
_MAX_EPISODE_STEPS = 1000
_MAX_ADMISSIBLE = 30  # valid-action hints surfaced per feedback
# Per-request wall-clock cap (textworld steps are sub-second; generous bound for a
# pathological first load). On timeout the worker is killed and the rollout dropped.
_STEP_TIMEOUT = 60
_CLOSE_TIMEOUT = 10
_STDERR_TAIL = 200  # child stderr lines retained for error messages

# The child runs this script via `python -c`. It owns one textworld env for the
# episode and answers a JSON-lines request loop on stdin/stdout: init|step|close.
_TW_WORKER_SRC = r'''
import json
import os
import sys
import traceback

# Reserve the child's real stdout (fd 1) for the JSON protocol and route stray
# prints (textworld/alfworld chatter) at stderr. dup fd 1 BEFORE rebinding
# sys.stdout: rebinding drops the last reference to the old stdout object, whose
# GC would close fd 1 — the very descriptor we now hold the protocol on.
_proto_fd = os.dup(1)
sys.stdout = sys.stderr


def _send(obj):
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    while data:
        data = data[os.write(_proto_fd, data):]


env = None
try:
    while True:
        line = sys.stdin.readline()
        if not line:  # parent closed stdin -> episode over
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "init":
                import textworld
                import textworld.gym
                from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

                infos = textworld.EnvInfos(won=True, admissible_commands=True)
                env_id = textworld.gym.register_game(
                    msg["gamefile"], infos,
                    max_episode_steps=int(msg.get("max_episode_steps", 1000)),
                    wrappers=[AlfredDemangler()],
                )
                env = textworld.gym.make(env_id)
                out = env.reset()  # reset emits the welcome banner: room receptacles + task
                _send({"ok": True, "obs0": out[0] if isinstance(out, tuple) else out})
            elif cmd == "step":
                obs, _score, _done, info = env.step(msg["action"])
                _send({"ok": True, "obs": obs,
                       "admissible": list(info["admissible_commands"]),
                       "won": bool(info["won"])})
            elif cmd == "close":
                break
            else:
                _send({"ok": False, "error": "unknown cmd: " + str(cmd)})
                break
        except Exception:
            _send({"ok": False, "error": "worker exception",
                   "traceback": traceback.format_exc()})
            break
finally:
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
'''


# Closes the assistant turn vLLM left open (it strips the stop token from the
# generated action) and opens the next user turn. Seeds <think> so each turn
# reasons before emitting the command, paired with the turn-1 seed in reset();
# mirrors geo3k.py's _tool_observation minus the tool_response wrapper. Needs a
# generous --rollout.max_new_tokens or the think block won't close before the cap.
def _observation_feedback(obs_text: str, admissible: list[str]) -> str:
    cmds = ", ".join(admissible[:_MAX_ADMISSIBLE])
    body = f"{obs_text}\nAdmissible actions: {cmds}" if cmds else obs_text
    return f"<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n"


class AlfWorldEnv(Env):
    def __init__(self):
        self.turn = 0
        self.proc = None
        self._stderr_tail = collections.deque(maxlen=_STDERR_TAIL)
        self._stderr_task = None

    async def reset(self, state):
        self.turn = 0
        gamefile = state.get("label")
        if not gamefile:
            raise ValueError(
                "AlfWorldEnv needs the .tw-pddl path in state['label'] "
                "— set --data.label_key to the column prepare_alfworld.py rl wrote."
            )
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _TW_WORKER_SRC,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # A chatty textworld can fill the stderr pipe buffer and block the child;
        # drain it on a background task and keep a tail for error messages.
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        reply = await self._round_trip(
            {"cmd": "init", "gamefile": gamefile, "max_episode_steps": _MAX_EPISODE_STEPS}
        )
        # Surface the reset observation (welcome banner + receptacle list) in the first
        # user turn, matching SFT: it's the only place the room's admissible "go to X"
        # targets appear (a mid-room `look` returns "see nothing"). Seed <think> here,
        # paired with the same seed in _observation_feedback for later turns.
        room_obs = reply["obs0"].split("\n\nYour task is to:")[0]
        marker = "<|im_end|>\n<|im_start|>assistant\n"
        base = state["observation"]
        idx = base.rfind(marker)
        if idx >= 0:
            base = base[:idx] + "\n" + room_obs + marker
        state["observation"] = base + "<think>\n"
        return state

    async def _drain_stderr(self):
        assert self.proc is not None and self.proc.stderr is not None
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            self._stderr_tail.append(line.decode("utf-8", "replace").rstrip())

    async def _round_trip(self, msg):
        proc = self.proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("alfworld worker is not running")

        async def _exchange():
            proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            await proc.stdin.drain()
            line = await proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    "alfworld worker closed unexpectedly\n--- worker stderr (tail) ---\n"
                    + "\n".join(self._stderr_tail)
                )
            return json.loads(line.decode("utf-8"))

        try:
            reply = await asyncio.wait_for(_exchange(), timeout=_STEP_TIMEOUT)
        except asyncio.TimeoutError:
            await self.close()
            raise RuntimeError(f"alfworld worker timed out after {_STEP_TIMEOUT}s on {msg.get('cmd')}")
        if not reply.get("ok"):
            await self.close()
            raise RuntimeError(
                f"alfworld worker error on {msg.get('cmd')}: {reply.get('error')}\n"
                + reply.get("traceback", "")
            )
        return reply

    async def step(self, state) -> Result:
        raw = state["action_text"]
        # Prefer an explicit <action>...</action> tag (robust to quotes, newlines, prose);
        # fall back to the text after </think> for a model that emits a bare command.
        if "<action>" in raw and "</action>" in raw:
            body = raw.split("<action>", 1)[1].split("</action>", 1)[0]
        else:
            body = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
            body = body.replace("<action>", " ").replace("</action>", " ")  # drop stray tags
        # One clean command: first line, drop a leading "<"/">"/"*"/"-", cut a ">"-chain,
        # unwrap quotes — TextWorld needs an exact admissible-command match.
        action = (
            body.strip().split("\n", 1)[0].lstrip("<>*- ").split(">", 1)[0]
            .strip().strip('"' + "'" + "`").lower()
        )
        self.turn += 1
        reply = await self._round_trip({"cmd": "step", "action": action})
        won = bool(reply["won"])
        truncated = (not won) and self.turn >= _MAX_TURNS
        reward = torch.tensor(1.0 if won else 0.0, dtype=torch.float32)
        result = Result(
            reward=reward,
            observation=_observation_feedback(reply["obs"], reply["admissible"]),
            terminated=won,
            truncated=truncated,
            info={
                "alfworld_won": reward,
                "turn_index": torch.tensor(float(self.turn), dtype=torch.float32),
            },
        )
        # Tear the subprocess down at natural episode end; StepEnvRunner's finally-close
        # covers the other exit paths (context exhaustion, exceptions). Idempotent.
        if won or truncated:
            await self.close()
        return result

    async def close(self):
        proc = self.proc
        stderr_task = self._stderr_task
        self.proc = None
        self._stderr_task = None
        if proc is None:
            return
        # Ask the worker to exit cleanly (best effort — it may already be gone).
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.write(b'{"cmd": "close"}\n')
                await proc.stdin.drain()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=_CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        if stderr_task is not None:
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stderr_task

    def __del__(self):
        # Backstop for paths that bypass close() (event loop torn down mid-episode).
        proc = getattr(self, "proc", None)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(AlfWorldEnv)
