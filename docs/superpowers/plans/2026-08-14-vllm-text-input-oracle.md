# vLLM Text Input Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a CPU-only, provenance-tracked extraction of vLLM's OpenAI Chat Completions text preprocessing path and generate reproducible GLM-5.2 request/result corpora.

**Architecture:** Vendored modules retain the minimal vLLM validation, chat normalization, Hugging Face rendering, and encoding behavior. A small stable API loads pinned Hugging Face assets and emits rendered text, token counts, and deterministic hashes; separate tools build and verify hand-written, combinatorial, and UltraChat corpora.

**Tech Stack:** Python 3.11, vLLM 0.26.0 source, Transformers/tokenizers, Jinja2, Pydantic, pytest, Hugging Face Hub/Datasets.

**Completion note (2026-08-15):** All acceptance items below are complete for
the user-confirmed GLM-5.2-only scope. The parity harness independently
reimplements the pinned upstream text-only path rather than importing the full
vLLM package, whose module imports require PyTorch and engine/multimodal
runtime dependencies. Qwen and Zephyr were removed from scope by the user's
later explicit model selection.

## Global Constraints

- Pin vLLM to `568afb3a13806beb53bb2e6bd518269357b237c0`.
- Do not download or initialize model weights, CUDA, or a GPU runtime.
- Preserve vLLM Apache-2.0 notices and record every extracted upstream file and edit.
- Support text-only Hugging Face chat models; first baseline is `zai-org/GLM-5.2` at a fixed revision.
- Reject processor-dependent multimodal inputs explicitly; never drop content silently.
- Successful results contain exact rendered text, token count, rendered-text hash, and token-ID hash.
- Hand-written results additionally contain complete token IDs.
- Dataset generation is deterministic, restartable, and never overwrites an existing baseline.

---

### Task 1: Package skeleton and deterministic result contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/vllm_text_oracle/__init__.py`
- Create: `src/vllm_text_oracle/contracts.py`
- Create: `src/vllm_text_oracle/hashing.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces: `canonical_json_sha256(value: object) -> str`, `text_sha256(text: str) -> str`, `token_ids_sha256(ids: Sequence[int]) -> str`, `OracleResult`.

- [x] Write tests proving canonical object-key ordering, UTF-8 text hashing, comma-decimal token hashing, and success/error result serialization.
- [x] Run `python -m pytest tests/test_contracts.py -q` and confirm failure because the package does not exist.
- [x] Implement frozen dataclasses and hash helpers; canonical JSON uses sorted keys, compact separators, UTF-8, and `ensure_ascii=False`.
- [x] Run `python -m pytest tests/test_contracts.py -q` and confirm all tests pass.
- [x] Commit with `feat: add deterministic oracle contracts`.

### Task 2: Fetch and record pinned upstream/model assets

**Files:**
- Create: `tools/fetch_upstream.py`
- Create: `tools/fetch_model_assets.py`
- Create: `vendor/vllm/UPSTREAM_COMMIT`
- Create: `vendor/vllm/EXTRACTION.md`
- Create: `vendor/vllm/upstream-files.json`
- Create: `vendor/vllm/LICENSE`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces: `fetch_checked(url: str, destination: Path, expected_sha256: str | None) -> str` and manifests containing source URL, revision, byte length, and SHA-256.

- [x] Write tests using local HTTP fixtures for non-200 responses, empty payloads, hash mismatches, and successful atomic downloads.
- [x] Run `python -m pytest tests/test_provenance.py -q` and confirm failure.
- [x] Implement checked downloads and fetch the exact vLLM source/tag metadata plus GLM-5.2 tokenizer/template/config assets without weight files.
- [x] Verify every downloaded file is non-empty and every manifest hash matches `sha256sum`.
- [x] Run provenance tests and `git diff --check`.
- [x] Commit with `build: pin upstream and model assets`.

### Task 3: Extract the vLLM text preprocessing path

**Files:**
- Create: `vendor/vllm/extracted/protocol.py`
- Create: `vendor/vllm/extracted/chat_utils.py`
- Create: `vendor/vllm/extracted/template_format.py`
- Create: `vendor/vllm/extracted/hf_renderer.py`
- Create: `src/vllm_text_oracle/model_assets.py`
- Create: `src/vllm_text_oracle/oracle.py`
- Test: `tests/test_oracle_basic.py`
- Test: `tests/test_oracle_tools.py`

**Interfaces:**
- Consumes: pinned tokenizer/config/template assets and hash helpers.
- Produces: `TextOracle.from_pretrained(model_id: str, revision: str) -> TextOracle` and `TextOracle.process(request: Mapping[str, object], include_token_ids: bool = False) -> OracleResult`.

- [x] Write failing tests for single/multi-turn roles, text content parts, tools/tool calls/tool results, thinking kwargs, generation prompt, assistant continuation, Unicode, and invalid messages.
- [x] Run the focused oracle tests and confirm imports or assertions fail.
- [x] Trace vLLM's `ChatCompletionRequest`, `parse_chat_messages`, content-format resolver, `HfRenderer`, and `safe_apply_chat_template`; copy only reachable text branches with original notices.
- [x] Add compatibility types only where engine/model/multimodal runtime dependencies would otherwise be required, documenting every change in `EXTRACTION.md` and `upstream-files.json`.
- [x] Implement the stable `TextOracle` wrapper and explicit `unsupported_multimodal` errors.
- [x] Run `python -m pytest tests/test_oracle_basic.py tests/test_oracle_tools.py -q` and confirm pass.
- [x] Commit with `feat: extract vLLM text input oracle`.

### Task 4: Differential parity against unmodified vLLM behavior

**Files:**
- Create: `tools/verify_upstream_parity.py`
- Create: `tests/parity_cases.py`
- Create: `tests/test_upstream_parity.py`

**Interfaces:**
- Consumes: `TextOracle.process()` and pinned unmodified vLLM renderer functions.
- Produces: a parity report containing case ID, model, rendered-byte equality, token-ID equality, and stable error-type equality.

- [x] Define parity cases for GLM-5.2, a small Qwen instruct tokenizer, and a Zephyr/Mistral-style tokenizer.
- [x] Write a test that fails with the first rendered-byte or token-ID mismatch and includes the mismatch index.
- [x] Run the parity test and confirm failure before the upstream harness exists.
- [x] Implement a CPU-only upstream harness that imports the pinned unmodified source and initializes tokenizer/rendering objects without an engine or weights.
- [x] Run parity for all hand-written structural cases and require exact text bytes and token IDs.
- [x] Commit with `test: verify extracted vLLM parity`.

### Task 5: Deterministic request corpus builders

**Files:**
- Create: `tools/build_handwritten.py`
- Create: `tools/build_combinatorial.py`
- Create: `tools/import_ultrachat.py`
- Create: `src/vllm_text_oracle/jsonl.py`
- Create: `datasets/manifests/request-set.json`
- Create: `datasets/manifests/source-licenses.json`
- Test: `tests/test_dataset_builders.py`

**Interfaces:**
- Produces: `write_jsonl_atomic(records: Iterable[Mapping], path: Path)`, about 300 hand-written records, about 2,000 fixed-seed combinations, and exactly 10,000 pinned UltraChat records.

- [x] Write tests for stable unique case IDs, exact category coverage, deterministic combination output, no text trimming, and deterministic hash-based UltraChat selection.
- [x] Run the dataset-builder tests and confirm failure.
- [x] Implement hand-written and pairwise-style generators with seed `20260814`.
- [x] Resolve and pin the UltraChat revision, download/stream records with checked errors, sort candidates by source-content hash, and select exactly 10,000.
- [x] Validate JSONL, counts, uniqueness, source metadata, and byte-for-byte rebuilds.
- [x] Commit with `data: build deterministic request corpora`.

### Task 6: Baseline generation, manifests, and final verification

**Files:**
- Create: `tools/generate_results.py`
- Create: `tools/verify_requests.py`
- Create: `tools/verify_results.py`
- Create: `tools/verify_reproducibility.py`
- Create: `datasets/results/zai-org--GLM-5.2/<revision>/manifest.json`
- Test: `tests/test_result_generation.py`

**Interfaces:**
- Consumes: request JSONL and `TextOracle.process()`.
- Produces: one result per case, sharded by request source, plus a fully pinned manifest and verification summary.

- [x] Write failing tests for one-to-one IDs, success/error schemas, full IDs only on hand-written cases, safe resume, changed-request rejection, duplicate rejection, and no baseline overwrite.
- [x] Run `python -m pytest tests/test_result_generation.py -q` and confirm failure.
- [x] Implement atomic shard generation and manifests containing all dependency, source, model, tokenizer, and template revisions/hashes.
- [x] Generate GLM-5.2 results for hand-written, combinatorial, and UltraChat request sets.
- [x] Run upstream parity over all hand-written/combinatorial cases and at least 1,000 deterministic UltraChat cases.
- [x] Rebuild into a temporary directory and require byte-for-byte identical outputs.
- [x] Run the full test suite, `git diff --check`, request/result verifiers, and summarize counts/errors/hashes.
- [x] Commit with `data: generate GLM-5.2 oracle baselines`.
