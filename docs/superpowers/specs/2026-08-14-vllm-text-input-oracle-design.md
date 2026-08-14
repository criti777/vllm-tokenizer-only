# vLLM Text Input Oracle Design

## Goal

Build a CPU-only reference implementation extracted from vLLM v0.26.0
(`568afb3a13806beb53bb2e6bd518269357b237c0`) for the text-only path:

```text
OpenAI Chat Completions request
-> request validation and defaults
-> message/content/tool normalization
-> model chat-template adaptation
-> template rendering
-> tokenizer encoding
-> rendered text and token metadata
```

The first reference dataset targets `zai-org/GLM-5.2`, while the extracted
pipeline remains usable with Hugging Face text chat models that provide a chat
template. Model weights, inference, CUDA, multimodal processors, prompt
embeddings, output parsers, and the Responses API are out of scope.

## Upstream provenance

- Preserve vLLM copyright headers and Apache-2.0 licensing.
- Record every upstream source file and SHA-256 in
  `vendor/vllm/upstream-files.json`.
- Record the exact upstream tag and commit in `vendor/vllm/UPSTREAM_COMMIT`.
- Keep extraction-specific compatibility code outside vendored modules where
  practical; document necessary edits in `vendor/vllm/EXTRACTION.md`.
- Verify extracted behavior differentially against unmodified vLLM functions
  without loading model weights.

## Supported input behavior

- Roles: system, developer, user, assistant, and tool.
- String content and OpenAI text content parts.
- Function tools, assistant tool calls, and tool results.
- `chat_template_content_format`: auto, string, and openai.
- `add_generation_prompt`, `continue_final_message`, and
  `add_special_tokens`.
- Request-level `chat_template_kwargs`, including GLM-5.2
  `enable_thinking` and `reasoning_effort`.
- Text-only degradation implemented by a model template, such as GLM-5.2's
  reminder for unsupported media parts.

Inputs that require a real multimodal processor must fail explicitly rather
than silently dropping content.

## Repository layout

```text
vendor/vllm/              extracted upstream code and provenance
src/vllm_text_oracle/     stable public API and lightweight compatibility types
tools/                    asset, dataset, baseline, and verification commands
datasets/requests/        source OpenAI request JSONL files
datasets/results/         model/revision-specific reference result JSONL files
tests/                    unit, differential, schema, and reproducibility tests
```

## Dataset design

The first version contains approximately 12,300 requests:

- About 300 hand-written conformance cases covering roles, tools, reasoning,
  content parts, Unicode, whitespace, long inputs, and invalid requests.
- About 2,000 deterministic pairwise/combinatorial cases with a fixed seed.
- 10,000 deterministically selected UltraChat records from a pinned dataset
  revision. Source text is not trimmed or normalized.

LMSYS-Chat-1M is deferred because it is gated and requires the user to accept
its terms. Its future import must be additive and must not alter existing case
IDs.

Each request record wraps, but does not mutate, the OpenAI request:

```json
{
  "case_id": "basic.single-user.001",
  "tags": ["handwritten", "basic"],
  "source": {"kind": "handwritten"},
  "request": {"model": "zai-org/GLM-5.2", "messages": []}
}
```

Case IDs are unique and stable. Imported records include the dataset ID,
revision, split, source index, and source-content hash.

## Result schema

A successful result contains:

```json
{
  "case_id": "basic.single-user.001",
  "status": "ok",
  "request_sha256": "...",
  "rendered_text": "...",
  "rendered_text_sha256": "...",
  "token_ids_length": 42,
  "token_ids_sha256": "...",
  "diagnostics": {
    "chat_template_content_format": "string",
    "add_generation_prompt": true,
    "continue_final_message": false,
    "add_special_tokens": false
  }
}
```

Hand-written cases additionally retain complete token IDs for diagnosis.
Generated and imported cases store count and hash only. Token-ID hashes are
SHA-256 of the UTF-8 decimal comma form, for example `154800,42,7`.
Rendered-text hashes are over the exact UTF-8 bytes. Request hashes use a
documented deterministic canonical JSON encoding.

Invalid requests produce a stable error stage and type. Exact dependency error
messages are diagnostic and are not treated as a stable contract.

## Generation and error handling

- Pin model, tokenizer, template, dataset, vLLM, Python, Transformers,
  tokenizers, Jinja, and Pydantic versions in the result manifest.
- Verify HTTP status, non-empty files, revision, and SHA-256 for downloads.
- Process records independently and support safe restart by case ID and request
  hash.
- Write to temporary files, validate counts and uniqueness, then atomically
  publish a completed shard.
- Record per-request validation/render/encode errors without aborting the full
  corpus.
- Abort on duplicate IDs, changed request hashes, missing assets, mismatched
  revisions, corrupt output, or an attempt to overwrite an existing baseline.

## Verification

Compare extracted and unmodified vLLM preprocessing using the same pinned
tokenizer assets. Require exact rendered UTF-8 bytes and exact token IDs for:

- Every hand-written case.
- Every generated combination case.
- All feasible UltraChat cases, with 1,000 deterministic records as the
  minimum mandatory parity sample.

The parity matrix includes GLM-5.2 plus one small Qwen instruct model and one
Zephyr/Mistral-style model to establish that the extraction remains generic.

Acceptance criteria:

- No model weights, CUDA, or GPU are required.
- Request and result counts match one-to-one.
- Every successful result includes rendered text, count, and hashes.
- Every failed result includes a stable stage and type.
- Repeated generation with the same manifest is byte-for-byte deterministic.
- Extracted and upstream vLLM outputs match exactly for all required parity
  cases.

