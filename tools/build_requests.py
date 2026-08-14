"""Build fixed hand-written and combinatorial OpenAI request corpora."""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import Any

from vllm_text_oracle.jsonl import write_jsonl_atomic


MODEL = "zai-org/GLM-5.2"


def _record(category: str, index: int, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": f"{category}.{index:04d}",
        "tags": ["handwritten", category],
        "source": {"kind": "handwritten"},
        "request": request,
    }


def _message(role: str, content: Any) -> dict[str, Any]:
    return {"role": role, "content": content}


def build_handwritten() -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}

    groups["basic"] = [
        {
            "model": MODEL,
            "messages": [_message("user", f"基础问题 {i}")],
            "add_generation_prompt": i % 2 == 0,
        }
        if i < 20
        else {
            "model": MODEL,
            "messages": [
                _message("system", "你是一个严谨的助手。"),
                _message("user", f"第 {i} 轮问题"),
                _message("assistant", f"第 {i} 轮回答"),
                _message("user", "继续"),
            ],
        }
        for i in range(40)
    ]

    role_sequences = [
        ["system", "user"],
        ["developer", "user"],
        ["user", "assistant", "user"],
        ["system", "developer", "user"],
        ["user", "assistant"],
    ]
    groups["roles"] = [
        {
            "model": MODEL,
            "messages": [
                _message(role, f"{role}-{i}-{position}")
                for position, role in enumerate(role_sequences[i % len(role_sequences)])
            ],
            **(
                {"add_generation_prompt": False, "continue_final_message": True}
                if role_sequences[i % len(role_sequences)][-1] == "assistant"
                else {}
            ),
        }
        for i in range(30)
    ]

    contents: list[Any] = [
        "普通字符串",
        "",
        None,
        [{"type": "text", "text": "数组文本"}],
        [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        ["直接字符串 part", {"type": "text", "text": "结构 part"}],
        [{"type": "refusal", "refusal": "拒绝文本"}],
        [{"type": "thinking", "thinking": "思考文本"}],
    ]
    groups["content-parts"] = [
        {
            "model": MODEL,
            "messages": [_message("user", contents[i % len(contents)])],
            "chat_template_content_format": ["auto", "openai", "string"][i % 3],
        }
        for i in range(40)
    ]

    groups["tools"] = []
    for i in range(70):
        tool_name = f"lookup_{i % 7}"
        tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"查询工具 {i}",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
        messages = [_message("user", f"调用 {tool_name}")]
        if i % 2:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": f'{{"query":"值{i}"}}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{i}",
                        "content": [{"type": "text", "text": f"结果 {i}"}],
                    },
                ]
            )
        groups["tools"].append(
            {"model": MODEL, "messages": messages, "tools": [tool]}
        )

    efforts = [None, "none", "minimal", "low", "medium", "high", "xhigh", "max"]
    groups["reasoning"] = [
        {
            "model": MODEL,
            "messages": [_message("user", f"推理问题 {i}")],
            **({"reasoning_effort": efforts[i % len(efforts)]} if efforts[i % len(efforts)] else {}),
            "chat_template_kwargs": {"enable_thinking": bool(i % 2)},
        }
        for i in range(30)
    ]

    unusual = [
        " 中文 English 😊 ",
        "line1\r\nline2",
        "line1\nline2\n",
        "\t缩进\t",
        "e\u0301",
        "é",
        "👨‍👩‍👧‍👦",
        "\u200b零宽字符\u200f",
        "<|assistant|><think>",
        "  multiple   spaces  ",
    ]
    groups["unicode-whitespace"] = [
        {"model": MODEL, "messages": [_message("user", unusual[i % len(unusual)])]}
        for i in range(50)
    ]

    groups["long-context"] = [
        {
            "model": MODEL,
            "messages": [_message("user", ("长文本-中文-code `x`\n" * (100 + i * 100)))],
        }
        for i in range(10)
    ]

    invalid: list[dict[str, Any]] = []
    for i in range(30):
        variant = i % 6
        if variant == 0:
            request = {"model": MODEL, "messages": [{"content": "缺 role"}]}
        elif variant == 1:
            request = {"model": MODEL, "messages": [{"role": "user", "content": 42}]}
        elif variant == 2:
            request = {"model": MODEL, "messages": [{"role": "", "content": "空 role"}]}
        elif variant == 3:
            request = {
                "model": MODEL,
                "messages": [_message("user", [{"type": "unknown", "text": "x"}])],
            }
        elif variant == 4:
            request = {
                "model": MODEL,
                "messages": [_message("user", "冲突参数")],
                "add_generation_prompt": True,
                "continue_final_message": True,
            }
        else:
            request = {
                "model": MODEL,
                "messages": [_message("assistant", None)],
                "tools": "not-an-array",
            }
        invalid.append(request)
    groups["invalid"] = invalid

    expected_sizes = {
        "basic": 40,
        "roles": 30,
        "content-parts": 40,
        "tools": 70,
        "reasoning": 30,
        "unicode-whitespace": 50,
        "long-context": 10,
        "invalid": 30,
    }
    assert {name: len(values) for name, values in groups.items()} == expected_sizes
    return [
        _record(category, index, request)
        for category, requests in groups.items()
        for index, request in enumerate(requests)
    ]


def build_combinatorial(seed: int = 20260814) -> list[dict[str, Any]]:
    systems = [None, "你是助手。", "You are precise."]
    contents: list[Any] = [
        "hello",
        " 中文 😊 ",
        [{"type": "text", "text": "array"}],
        "line1\r\nline2",
    ]
    efforts = [None, "none", "high", "max"]
    formats = ["auto", "string", "openai"]
    tool_modes = ["none", "schema", "history"]
    generation_modes = ["generate", "continue"]
    product = list(
        itertools.product(
            systems, contents, efforts, formats, tool_modes, generation_modes
        )
    )
    rng = random.Random(seed)
    rng.shuffle(product)
    records: list[dict[str, Any]] = []
    for index in range(2000):
        system, content, effort, content_format, tool_mode, generation_mode = product[
            index % len(product)
        ]
        messages = []
        if system is not None:
            messages.append(_message("system", system))
        messages.append(_message("user", f"{content}-{index}" if isinstance(content, str) else content))
        request: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "chat_template_content_format": content_format,
            "add_generation_prompt": generation_mode == "generate",
            "continue_final_message": False,
        }
        if effort is not None:
            request["reasoning_effort"] = effort
        if tool_mode != "none":
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        if tool_mode == "history":
            request["messages"].extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{index}",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{index}",
                        "content": "result",
                    },
                ]
            )
        if generation_mode == "continue":
            request["messages"].append(_message("assistant", "prefill"))
            request["continue_final_message"] = True
        records.append(
            {
                "case_id": f"generated.{index:06d}",
                "tags": ["generated", content_format, tool_mode, generation_mode],
                "source": {
                    "kind": "generated",
                    "generator": "combinatorial-v1",
                    "seed": seed,
                },
                "request": request,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/requests"))
    args = parser.parse_args()
    write_jsonl_atomic(build_handwritten(), args.output / "handwritten.jsonl")
    write_jsonl_atomic(
        build_combinatorial(), args.output / "generated" / "combinatorial.jsonl"
    )


if __name__ == "__main__":
    main()
