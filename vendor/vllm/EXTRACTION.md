# vLLM text preprocessing extraction

The files under `upstream/` are byte-for-byte copies from vLLM v0.26.0 at
commit `568afb3a13806beb53bb2e6bd518269357b237c0`. They are retained for review
and provenance and are not imported at runtime.

The files under `extracted/` preserve the Apache-2.0 headers and the behavior
reachable from the text-only Chat Completions preprocessing path. The
extraction deliberately removes engine, scheduler, model-weight, Torch,
multimodal processor, prompt-embedding, async-worker, and output-parser code.

Mapping:

- `protocol.py` extracts the preprocessing fields, defaults, and parameter
  merging from `ChatCompletionRequest.build_chat_params`.
- `chat_utils.py` extracts content normalization, message metadata copying,
  tool-result normalization, and assistant tool-argument JSON parsing from
  `parse_chat_messages` and `_postprocess_messages`.
- `template_format.py` extracts the Jinja content-format detection behavior.
- `hf_renderer.py` extracts developer-role fallback, template rendering, and
  tokenizer encoding from `HfRenderer` and `safe_apply_chat_template`.
- `deepseek_v32_encoding.py` and `deepseek_v4_encoding.py` are byte-for-byte
  copies of the pinned vLLM tokenizer encoders. The lightweight renderer
  adapters in `src/vllm_text_oracle/renderers.py` replace only vLLM engine,
  `VllmConfig`, async executor, and prompt-container plumbing.
- The matching upstream tokenizer wrappers and renderer classes are retained
  under `upstream/tokenizers/` and `upstream/renderers/` for differential
  review; they are not imported at runtime.

Intentional text-only adaptations:

- Lightweight Pydantic models replace vLLM's serving and sampling schemas;
  unrelated OpenAI fields are accepted as extras and do not enter the prompt.
- The tokenizer and model assets are loaded directly from a pinned local
  directory instead of `ModelConfig`, `VllmConfig`, or engine objects.
- OpenAI media content parts are passed through only when the selected text
  template explicitly consumes them; processor-dependent string-mode media
  inputs fail as `unsupported_multimodal`.
- Errors are classified by stable pipeline stage and type; dependency-specific
  message text remains diagnostic.
- DeepSeek V3.2 and V4 are selected explicitly from the model profile registry;
  neither renderer may fall back to the generic Hugging Face chat template.
