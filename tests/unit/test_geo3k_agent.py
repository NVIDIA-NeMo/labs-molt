import asyncio
import importlib.util
from pathlib import Path

_AGENT_PATH = Path(__file__).resolve().parents[2] / "examples" / "python" / "agents" / "geo3k.py"


def _load_geo3k():
    spec = importlib.util.spec_from_file_location("geo3k_agent", _AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


geo3k = _load_geo3k()


def test_step_terminates_on_answer_even_with_co_emitted_tool_call(monkeypatch):
    """A turn that commits a final answer must terminate, even if it also emits a
    tool_call — otherwise the rollout keeps tool-calling and inflates length."""
    env = geo3k.GeoEnv()
    monkeypatch.setattr(
        geo3k, "_extract_tool_call", lambda text: {"name": "python_executor", "arguments": {"code": "print(1)"}}
    )
    monkeypatch.setattr(geo3k, "_grade_answer", lambda text, label: (1.0, "5"))

    result = asyncio.run(
        env.step(
            {
                "action_text": "<answer>5</answer> let me double-check <tool_call>x</tool_call>",
                "label": {"ground_truth": "5"},
            }
        )
    )

    assert result.terminated is True
    assert result.reward.item() == 1.0
    assert env.tool_call_count == 0  # did NOT run a tool after the answer was committed


def test_step_continues_on_tool_call_without_answer(monkeypatch):
    """No committed answer + a tool_call → keep going (mid-trajectory, not terminal)."""
    env = geo3k.GeoEnv()
    monkeypatch.setattr(
        geo3k, "_extract_tool_call", lambda text: {"name": "python_executor", "arguments": {"code": "print(1)"}}
    )

    result = asyncio.run(
        env.step({"action_text": "let me compute <tool_call>x</tool_call>", "label": {"ground_truth": "5"}})
    )

    assert result.terminated is False
    assert env.tool_call_count == 1
