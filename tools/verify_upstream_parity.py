"""Differential check against an independent vLLM text-only reference path."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from vllm_text_oracle.jsonl import read_jsonl
from vllm_text_oracle.oracle import TextOracle
from vendor.vllm.extracted.protocol import ChatCompletionRequest
from vendor.vllm.extracted.template_format import detect_content_format


DEFAULT_ASSETS = Path(
    "model_assets/zai-org--GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9"
)


def _part(part: Any, openai: bool) -> Any:
    if isinstance(part, str):
        return {"type": "text", "text": part} if openai else part
    if not isinstance(part, dict):
        raise ValueError("chat content parts must be strings or objects")
    kind = part.get("type")
    if kind in {"text", "input_text", "output_text", "refusal", "thinking"}:
        value = part.get("text", part.get(kind) if kind in {"refusal", "thinking"} else None)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("text content must be a string")
        if not openai:
            return value
        result = {"type": "text", "text": value}
        result.update({k: v for k, v in part.items() if k not in {"type", "text"}})
        return result
    if kind == "tool_reference":
        return copy.deepcopy(part) if openai else part.get("name", "")
    raise ValueError(f"unsupported chat content part type: {kind!r}")


def _conversation(messages: list[dict[str, Any]], content_format: str) -> list[dict[str, Any]]:
    openai = content_format == "openai"
    result: list[dict[str, Any]] = []
    for source in messages:
        raw = source.get("content")
        parts = [] if raw is None else ([raw] if isinstance(raw, str) else raw)
        if not isinstance(parts, list):
            raise ValueError("message content must be a string, array, or null")
        values = [value for item in parts if (value := _part(item, openai)) is not None]
        message: dict[str, Any] = {
            "role": source["role"],
            "content": values if openai else "\n".join(values),
        }
        for key in ("name", "task"):
            if isinstance(source.get(key), str):
                message[key] = source[key]
        if source["role"] == "assistant":
            if source.get("tool_calls") is not None:
                message["tool_calls"] = copy.deepcopy(source["tool_calls"])
            reasoning = source.get("reasoning", source.get("reasoning_content"))
            if reasoning is not None:
                message["reasoning"] = reasoning
                message["reasoning_content"] = reasoning
        elif source["role"] == "tool":
            if "tool_call_id" in source:
                message["tool_call_id"] = source["tool_call_id"]
            if isinstance(message["content"], list) and all(
                isinstance(x, dict) and x.get("type") == "text" for x in message["content"]
            ):
                message["content"] = "\n".join(x.get("text", "") for x in message["content"])
        elif source["role"] == "developer":
            message["tools"] = source.get("tools")
        result.append(message)
    for message in result:
        calls = message.get("tool_calls") if message["role"] == "assistant" else None
        if calls == []:
            message.pop("tool_calls")
        elif isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise ValueError("invalid assistant tool call")
                function = call["function"]
                arguments = function.get("arguments")
                if arguments:
                    if not isinstance(arguments, (dict, list)):
                        function["arguments"] = json.loads(arguments) or {}
                else:
                    function["arguments"] = {}
    return result


def reference_process(request: dict[str, Any], tokenizer: Any) -> tuple[str, str | None, list[int] | None]:
    try:
        parsed = ChatCompletionRequest.model_validate(request)
        template = tokenizer.get_chat_template(parsed.chat_template, tools=parsed.tools)
        content_format = (
            detect_content_format(template)
            if parsed.chat_template_content_format == "auto"
            else parsed.chat_template_content_format
        )
        conversation = _conversation(parsed.messages, content_format)
        if any(x["role"] == "developer" for x in conversation) and not (
            '"developer"' in template or "'developer'" in template
        ):
            converted = []
            for item in conversation:
                item = dict(item)
                if item["role"] == "developer":
                    item["role"] = "system"
                    item.pop("tools", None)
                converted.append(item)
            systems = [x for x in converted if x["role"] == "system"]
            if len(systems) > 1 or (systems and converted[0]["role"] != "system"):
                converted = [{"role": "system", "content": "\n\n".join(x.get("content", "") for x in systems if x.get("content"))}, *[x for x in converted if x["role"] != "system"]]
            conversation = converted
        kwargs = parsed.template_kwargs(parsed.tools)
        rendered = tokenizer.apply_chat_template(
            conversation=conversation,
            tools=parsed.tools,
            chat_template=template,
            tokenize=False,
            **{k: v for k, v in kwargs.items() if k != "tools"},
        )
        return "ok", rendered, list(tokenizer.encode(rendered, add_special_tokens=parsed.add_special_tokens))
    except Exception:
        return "error", None, None


def verify_parity(request_root: Path, assets: Path = DEFAULT_ASSETS, ultrachat_sample: int = 1000) -> dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    oracle = TextOracle(tokenizer)
    sources = [
        ("handwritten", request_root / "handwritten.jsonl", None),
        ("generated", request_root / "generated/combinatorial.jsonl", None),
        ("ultrachat", request_root / "imported/ultrachat.jsonl", ultrachat_sample),
    ]
    summary: dict[str, int] = {}
    for name, path, limit in sources:
        records = list(read_jsonl(path))
        if limit is not None:
            records = sorted(records, key=lambda x: x["case_id"])[:limit]
        for record in records:
            case_id = record["case_id"]
            actual = oracle.process(record["request"], case_id=case_id, include_token_ids=True)
            status, rendered, ids = reference_process(record["request"], tokenizer)
            if actual.status != status:
                raise ValueError(f"{case_id}: status mismatch")
            if status == "ok" and (actual.rendered_text != rendered or list(actual.token_ids or ()) != ids):
                raise ValueError(f"{case_id}: rendered bytes or token IDs mismatch")
        summary[name] = len(records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=Path("datasets/requests"))
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--ultrachat-sample", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(verify_parity(args.requests, args.assets, args.ultrachat_sample), sort_keys=True))


if __name__ == "__main__":
    main()
