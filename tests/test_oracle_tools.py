from pathlib import Path

import pytest

from vllm_text_oracle.oracle import TextOracle


pytestmark = pytest.mark.model("glm-5.2")


MODEL_PATH = Path(
    "model_assets/zai-org--GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9"
)


def test_openai_tool_arguments_are_parsed_before_template_rendering() -> None:
    oracle = TextOracle.from_local_assets(MODEL_PATH)
    result = oracle.process(
        {
            "messages": [
                {"role": "user", "content": "北京天气"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"北京\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "text", "text": "晴，30℃"}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        },
        case_id="tools.001",
    )

    assert result.status == "ok"
    assert result.rendered_text is not None
    assert "<tool_call>get_weather<arg_key>city</arg_key>" in result.rendered_text
    assert "<tool_response>晴，30℃</tool_response>" in result.rendered_text
    assert result.token_ids_length is not None and result.token_ids_length > 20
