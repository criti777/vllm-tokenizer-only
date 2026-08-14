# vLLM Text Input Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only, provenance-tracked extraction of vLLM's OpenAI Chat Completions text preprocessing path and generate reproducible GLM-5.2 request/result corpora.

**Architecture:** Vendored modules retain the minimal vLLM validation, chat normalization, Hugging Face rendering, and encoding behavior. A small stable API loads pinned Hugging Face assets and emits rendered text, token counts, and deterministic hashes; separate tools build and verify hand-written, combinatorial, and UltraChat corpora.

**Tech Stack:** Python 3.12, vLLM 0.26.0 source, Transformers/tokenizers, Jinja2, Pydantic, pytest, Hugging Face Hub/Datasets.

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

- [ ] Write tests proving canonical object-key ordering, UTF-8 text hashing, comma-decimal token hashing, and success/error result serialization.
- [ ] Run `python -m pytest tests/test_contracts.py -q` and confirm failure because the package does not exist.
- [ ] Implement frozen dataclasses and hash helpers; canonical JSON uses sorted keys, compact separators, UTF-8, and `ensure_ascii=False`.
- [ ] Run `python -m pytest tests/test_contracts.py -q` and confirm all tests pass.
- [ ] Commit with `feat: add deterministic oracle contracts`.

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

- [ ] Write tests using local HTTP fixtures for non-200 responses, empty payloads, hash mismatches, and successful atomic downloads.
- [ ] Run `python -m pytest tests/test_provenance.py -q` and confirm failure.
- [ ] Implement checked downloads and fetch the exact vLLM source/tag metadata plus GLM-5.2 tokenizer/template/config assets without weight files.
- [ ] Verify every downloaded file is non-empty and every manifest hash matches `sha256sum`.
- [ ] Run provenance tests and `git diff --check`.
- [ ] Commit with `build: pin upstream and model assets`.

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

- [ ] Write failing tests for single/multi-turn roles, text content parts, tools/tool calls/tool results, thinking kwargs, generation prompt, assistant continuation, Unicode, and invalid messages.
- [ ] Run the focused oracle tests and confirm imports or assertions fail.
- [ ] Trace vLLM's `ChatCompletionRequest`, `parse_chat_messages`, content-format resolver, `HfRenderer`, and `safe_apply_chat_template`; copy only reachable text branches with original notices.
- [ ] Add compatibility types only where engine/model/multimodal runtime dependencies would otherwise be required, documenting every change in `EXTRACTION.md` and `upstream-files.json`.
- [ ] Implement the stable `TextOracle` wrapper and explicit `unsupported_multimodal` errors.
- [ ] Run `python -m pytest tests/test_oracle_basic.py tests/test_oracle_tools.py -q` and confirm pass.
- [ ] Commit with `feat: extract vLLM text input oracle`.

### Task 4: Differential parity against unmodified vLLM behavior

**Files:**
- Create: `tools/verify_upstream_parity.py`
- Create: `tests/parity_cases.py`
- Create: `tests/test_upstream_parity.py`

**Interfaces:**
- Consumes: `TextOracle.process()` and pinned unmodified vLLM renderer functions.
- Produces: a parity report containing case ID, model, rendered-byte equality, token-ID equality, and stable error-type equality.

- [ ] Define parity cases for GLM-5.2, a small Qwen instruct tokenizer, and a Zephyr/Mistral-style tokenizer.
- [ ] Write a test that fails with the first rendered-byte or token-ID mismatch and includes the mismatch index.
- [ ] Run the parity test and confirm failure before the upstream harness exists.
- [ ] Implement a CPU-only upstream harness that imports the pinned unmodified source and initializes tokenizer/rendering objects without an engine or weights.
- [ ] Run parity for all hand-written structural cases and require exact text bytes and token IDs.
- [ ] Commit with `test: verify extracted vLLM parity`.

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

- [ ] Write tests for stable unique case IDs, exact category coverage, deterministic combination output, no text trimming, and deterministic hash-based UltraChat selection.
- [ ] Run the dataset-builder tests and confirm failure.
- [ ] Implement hand-written and pairwise-style generators with seed `20260814`.
- [ ] Resolve and pin the UltraChat revision, download/stream records with checked errors, sort candidates by source-content hash, and select exactly 10,000.
- [ ] Validate JSONL, counts, uniqueness, source metadata, and byte-for-byte rebuilds.
- [ ] Commit with `data: build deterministic request corpora`.

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

- [ ] Write failing tests for one-to-one IDs, success/error schemas, full IDs only on hand-written cases, safe resume, changed-request rejection, duplicate rejection, and no baseline overwrite.
- [ ] Run `python -m pytest tests/test_result_generation.py -q` and confirm failure.
- [ ] Implement atomic shard generation and manifests containing all dependency, source, model, tokenizer, and template revisions/hashes.
- [ ] Generate GLM-5.2 results for hand-written, combinatorial, and UltraChat request sets.
- [ ] Run upstream parity over all hand-written/combinatorial cases and at least 1,000 deterministic UltraChat cases.
- [ ] Rebuild into a temporary directory and require byte-for-byte identical outputs.
- [ ] Run the full test suite, `git diff --check`, request/result verifiers, and summarize counts/errors/hashes.
- [ ] Commit with `data: generate GLM-5.2 oracle baselines`.

