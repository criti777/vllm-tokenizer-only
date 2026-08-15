from pathlib import Path

import pytest

from vllm_text_oracle import TextOracle


ASSETS_ROOT = Path("model_assets")
TEXT = "<｜begin▁of▁sentence｜><｜User｜>你好, world 🌍<｜Assistant｜></think>"


@pytest.mark.parametrize(
    ("profile_id", "expected_ids"),
    [
        pytest.param(
            "deepseek-v3.2",
            (0, 128803, 30594, 14, 2058, 73369, 238, 128804, 128799),
            marks=pytest.mark.model("deepseek-v3.2"),
        ),
        pytest.param(
            "deepseek-v4",
            (0, 128803, 30594, 14, 2058, 73369, 238, 128804, 128822),
            marks=pytest.mark.model("deepseek-v4"),
        ),
    ],
)
def test_specialized_basic_matches_pinned_vllm_encoding(
    profile_id: str,
    expected_ids: tuple[int, ...],
) -> None:
    oracle = TextOracle.from_model(profile_id, assets_root=ASSETS_ROOT)

    result = oracle.process(
        {"messages": [{"role": "user", "content": "你好, world 🌍"}]},
        case_id=f"{profile_id}.basic",
        include_token_ids=True,
    )

    assert result.status == "ok", result.error
    assert result.rendered_text == TEXT
    assert result.token_ids == expected_ids


@pytest.mark.model("deepseek-v4")
def test_v4_preserves_wo_eos_message_extension() -> None:
    oracle = TextOracle.from_model("deepseek-v4", assets_root=ASSETS_ROOT)

    result = oracle.process(
        {
            "messages": [
                {"role": "user", "content": "draft"},
                {"role": "assistant", "content": "partial", "wo_eos": True},
            ]
        },
        case_id="deepseek-v4.wo-eos",
    )

    assert result.status == "ok", result.error
    assert result.rendered_text is not None
    assert result.rendered_text.endswith("partial")
    assert not result.rendered_text.endswith("<｜end▁of▁sentence｜>")


@pytest.mark.model("deepseek-v4")
def test_v4_sorts_tool_results_by_declared_call_order() -> None:
    oracle = TextOracle.from_model("deepseek-v4", assets_root=ASSETS_ROOT)
    result = oracle.process(
        {
            "messages": [
                {"role": "user", "content": "run both"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "a", "arguments": "{}"},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "b", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_b", "content": "second"},
                {"role": "tool", "tool_call_id": "call_a", "content": "first"},
            ]
        },
        case_id="deepseek-v4.tool-order",
    )

    assert result.status == "ok", result.error
    assert result.rendered_text is not None
    assert result.rendered_text.index("<tool_result>first</tool_result>") < (
        result.rendered_text.index("<tool_result>second</tool_result>")
    )


@pytest.mark.model("deepseek-v4")
def test_v4_maps_xhigh_reasoning_effort_to_max_prefix() -> None:
    oracle = TextOracle.from_model("deepseek-v4", assets_root=ASSETS_ROOT)
    result = oracle.process(
        {
            "reasoning_effort": "xhigh",
            "messages": [{"role": "user", "content": "prove it"}],
        },
        case_id="deepseek-v4.xhigh",
    )

    assert result.status == "ok", result.error
    assert result.diagnostics["reasoning_effort"] == "max"
    assert result.rendered_text is not None
    assert "Reasoning Effort: Absolute maximum" in result.rendered_text


@pytest.mark.model("deepseek-v3.2")
def test_v32_renders_tools_in_dsml_system_section() -> None:
    oracle = TextOracle.from_model("deepseek-v3.2", assets_root=ASSETS_ROOT)
    result = oracle.process(
        {
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        case_id="deepseek-v3.2.tools",
    )

    assert result.status == "ok", result.error
    assert result.rendered_text is not None
    assert "<｜DSML｜function_calls>" in result.rendered_text
    assert '"name": "weather"' in result.rendered_text
