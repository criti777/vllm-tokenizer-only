# vLLM Text Input Oracle

这是一个不加载模型权重的离线文本预处理 oracle（参考实现）和分模型黄金数据集。

它解决的问题很具体：给定一份 OpenAI Chat Completions 风格请求，按照固定版本的
vLLM 和模型官方 tokenizer/template，计算出：

- 请求是否能被这一模型接受；
- 渲染后、编码前的精确文本；
- 完整 token ID 序列或其 SHA-256；
- `token_ids` 的精确数量；
- 如果失败，失败发生在哪个处理阶段。

你可以用它验证另一套“OpenAI 请求 → 规范化 → 模型适配 → render → encode”实现。
本项目把 vLLM 当作基准口径，但只抽取文本输入链路，不启动 vLLM 服务、不加载模型
权重，也不进行推理。

## 一句话理解

输入：

```json
{
  "model": "zai-org/GLM-5.2",
  "messages": [
    {"role": "system", "content": "你是一个严谨的助手。"},
    {"role": "user", "content": "1+1 等于多少？"}
  ]
}
```

输出：

```json
{
  "status": "ok",
  "model_profile": "glm-5.2",
  "renderer": "hf",
  "rendered_text": "...模型实际看到的文本...",
  "rendered_text_sha256": "...",
  "token_ids_length": 3,
  "token_ids_sha256": "...",
  "token_ids": [154822, 154824, 154826]
}
```

上面的 token 数字只用于说明字段形态；实际结果以 oracle 运行输出和仓库中的黄金
结果为准。

## 为什么不能只调用 `AutoTokenizer.encode(messages)`

`messages` 不是 tokenizer 的直接输入。在 encode 之前还有一条容易出现模型差异的
链路：

```text
OpenAI JSON 请求
  ↓
请求字段默认值和协议校验
  ↓
messages / content parts / tools 规范化
  ↓
选择模型 profile 和专用处理路径
  ↓
模型 chat template 渲染
  ↓
官方 tokenizer encode
  ↓
rendered_text + token_ids + token_ids_length
```

真正容易出错的通常不是 BPE 本身，而是 encode 之前的行为，例如：

- `developer`、`system`、`tool` 等 role 如何处理；
- assistant `tool_calls` 的 arguments 是否按 JSON 解析；
- tool result 如何排序、合并和放入模板；
- `add_generation_prompt` 与 `continue_final_message` 如何组合；
- thinking/reasoning 参数怎样映射为模板变量；
- content 是字符串还是 OpenAI content-parts 数组；
- 模型是否使用普通 Hugging Face 模板，还是 vLLM 的专用 renderer；
- 模板额外插入了哪些 BOS、EOS、role、thinking 或 tool 特殊 token。

因此，本项目同时保存渲染文本、token 数量和 token 序列哈希。只比较最终数量可以
发现问题，但比较中间结果能更快定位问题发生在哪一层。

## 项目边界

### 做什么

- 接收 OpenAI Chat Completions 风格 JSON 对象；
- 校验与规范化文本消息、content parts、tools 和 tool results；
- 根据模型 profile 选择正确的通用或专用 renderer；
- 使用固定的模型官方 tokenizer 和 chat template；
- 输出 rendered text、token IDs、token 数量和稳定哈希；
- 生成可按模型独立运行、验证和复现的黄金结果集；
- 保存上游来源、commit、模型 revision、文件大小和 SHA-256，便于审计。

### 不做什么

- 不下载或加载模型权重；
- 不启动 vLLM server；
- 不执行 prefill、decode 或文本生成；
- 不计算 KV cache；
- 不把图片解码成 pixel values；
- 不运行视觉 encoder，也不计算图片 embedding；
- 不声称与任意版本的线上商业 API `usage.prompt_tokens` 完全等价；
- 不对注册表之外的模型做静默 fallback。

## 固定版本和支持模型

vLLM 基线固定为：

- tag：`v0.26.0`
- commit：`568afb3a13806beb53bb2e6bd518269357b237c0`

当前严格支持七个官方主版本：

| Profile | 官方模型仓 | 固定 revision | Renderer |
|---|---|---|---|
| `deepseek-v3` | `deepseek-ai/DeepSeek-V3` | `e815299b0bcbac849fa540c768ef21845365c9eb` | HF |
| `deepseek-v3.2` | `deepseek-ai/DeepSeek-V3.2` | `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` | vLLM DeepSeek V3.2 专用路径 |
| `deepseek-v4` | `deepseek-ai/DeepSeek-V4-Flash` | `60d8d70770c6776ff598c94bb586a859a38244f1` | vLLM DeepSeek V4 专用路径 |
| `kimi-k2.6` | `moonshotai/Kimi-K2.6` | `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` | HF + 官方自定义 tokenizer |
| `glm-5.1` | `zai-org/GLM-5.1` | `26e1bd6e011feb778d25ae34b09b07074139d92d` | HF |
| `glm-5.2` | `zai-org/GLM-5.2` | `b4734de4facf877f85769a911abafc5283eab3d9` | HF |
| `minimax-m2.7` | `MiniMaxAI/MiniMax-M2.7` | `d494266a4affc0d2995ba1fa35c8481cbd84294b` | HF |

模型必须在 `models/profiles.json` 中明确注册。未知模型、错误别名或请求中的 model 与
当前 oracle profile 不一致都会失败，不会偷偷改用某个“看起来差不多”的 tokenizer。

## 从 vLLM 提取了什么

`vendor/vllm/upstream/` 保存固定 commit 的上游文件，用作来源证据，不在运行时直接
导入。`vendor/vllm/extracted/` 保存文本路径所需的精简实现。

主要映射如下：

| 功能 | 上游依据 | 提取实现 |
|---|---|---|
| Chat 请求默认值与参数合并 | `ChatCompletionRequest.build_chat_params` | `protocol.py` |
| messages/content/tools 规范化 | `parse_chat_messages`、`_postprocess_messages` | `chat_utils.py` |
| 模板 content format 检测 | vLLM Jinja 格式检测逻辑 | `template_format.py` |
| 通用 HF render + encode | `HfRenderer`、`safe_apply_chat_template` | `hf_renderer.py` |
| DeepSeek V3.2 编码 | vLLM 专用 tokenizer/renderer | `deepseek_v32_encoding.py` + 轻量适配器 |
| DeepSeek V4 编码 | vLLM 专用 tokenizer/renderer | `deepseek_v4_encoding.py` + 轻量适配器 |

提取时删除了与文本 token 计算无关的部分：引擎、scheduler、异步 executor、Torch、
权重加载、多模态 processor、prompt embedding 和输出 parser。DeepSeek V3.2/V4 的
核心 encoding 文件与固定上游逐字节一致，轻量适配器只替换 vLLM 引擎对象和运行时
外壳。

机器可读的“profile → renderer → 上游文件 → 提取文件 → 测试”映射位于
`vendor/vllm/coverage.json`；更详细的提取说明位于 `vendor/vllm/EXTRACTION.md`。

## 模型特殊处理

### DeepSeek V3.2 和 V4

这两个 profile 明确走 vLLM 专用 renderer，禁止回退到普通 HF chat template。
当前测试覆盖的专用行为包括：

- DSML tool 定义与 tool call；
- tool results 根据调用顺序重排；
- `reasoning_effort` 映射；
- V4 的 `wo_eos`；
- V4 的 `xhigh → max` 兼容映射。

### Kimi-K2.6

Kimi 使用模型仓中的官方自定义 tokenizer 源码，因此该 profile 显式开启
`trust_remote_code`。相关 Python 文件已经固定 revision、记录 SHA-256 并做静态
审计；加载时只使用本地资产，派生动态模块放在临时目录，不复用未校验缓存。

### GLM-5.1 与 GLM-5.2

它们不是因为名字接近就共享黄金值。每个版本都有独立 profile、独立官方 revision、
独立资产和结果集。实际测试也证明同一请求可能得到不同 rendered text 和 token 数；
例如 GLM-5.2 模板可能插入其默认 reasoning-effort system 段。

## 多模态请求如何处理

本项目的目标是统计文本 token，而不是复刻完整多模态 processor。

对于包含 `image_url`、`video`、`audio` 等 content part 的请求：

- 如果模型官方文本模板本身能确定占位字符串，则只渲染该占位字符串并进行文本
  encode；当前 Kimi-K2.6 会产生官方 `<|media_*|>` 占位序列；
- 如果占位 token 的数量或形态必须由模型 processor 根据真实媒体确定，则返回
  `processor_required / multimodal_processor_required`；
- 不访问图片 URL，不下载媒体，不计算 pixel values、视觉 embedding 或视觉 token。

所以，成功结果中的 `token_ids_length` 始终是本项目边界内的文本 token 数，不能把它
误称为完整多模态模型输入长度。

## 仓库目录

```text
.
├── src/vllm_text_oracle/       # 稳定公共 API、profile 路由、结果与哈希
├── vendor/vllm/
│   ├── upstream/               # 固定 vLLM 上游源文件，仅用于审计
│   ├── extracted/              # 文本预处理链路的精简提取版
│   ├── coverage.json           # profile 到来源和测试的覆盖映射
│   └── upstream-files.json     # 上游文件哈希
├── models/
│   ├── profiles.json           # 七个严格模型 profile
│   └── manifests/              # 每个模型文本资产的大小和 SHA-256
├── model_assets/               # tokenizer/template/config 等，不含权重
├── datasets/
│   ├── requests/               # 原始共享请求集
│   ├── manifests/              # 数据来源、revision、许可证与哈希
│   └── results/by-profile/     # 七模型黄金结果
├── tools/                      # 构造、生成和验证命令
└── tests/                      # core 和按模型选择的测试
```

## 安装

需要 Python 3.11 或更高版本。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[test,data]'
```

固定运行依赖见 `pyproject.toml`，包括 Transformers、Tokenizers、Jinja、Pydantic、
Hugging Face Hub 和 tiktoken。不需要安装 PyTorch、CUDA 或 vLLM wheel。

模型文本资产已经提交在仓库中。重新下载时，下载器会检查 HTTP 状态、空响应、文件
大小和 SHA-256，并拒绝权重后缀、路径逃逸和未声明 Python 文件。

## 单请求调用

```python
from pathlib import Path

from vllm_text_oracle import TextOracle

oracle = TextOracle.from_model(
    "glm-5.2",
    assets_root=Path("model_assets"),
)

result = oracle.process(
    {
        "model": "zai-org/GLM-5.2",
        "messages": [
            {"role": "system", "content": "你是一个严谨的助手。"},
            {"role": "user", "content": "你好"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "查询资料",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        "chat_template_kwargs": {"enable_thinking": True},
    },
    case_id="demo.001",
    include_token_ids=True,
)

payload = result.to_dict()
print(payload["status"])
print(payload.get("rendered_text"))
print(payload.get("token_ids_length"))
print(payload.get("token_ids"))
```

`include_token_ids=True` 会返回完整 token IDs，适合调试小样本。大规模基准通常只保存
长度和哈希，以控制仓库体积。

## 请求集设计

共享请求集位于 `datasets/requests/`：

| 层 | 文件 | 数量 | 用途 |
|---|---|---:|---|
| 手写黄金边界 | `handwritten.jsonl` | 300 | role、content parts、tools、thinking、非法请求等精确边界 |
| 组合回归 | `generated/combinatorial.jsonl` | 2,000 | 参数与消息形态的确定性组合 |
| 真实对话分布 | `imported/ultrachat.jsonl` | 10,000 | 多轮、长文本和自然语言分布 |

请求文件中的每一行具有：

```json
{
  "case_id": "tools.0001",
  "tags": ["handwritten", "tools"],
  "source": {"kind": "handwritten"},
  "request": {"model": "...", "messages": [], "tools": []}
}
```

同一共享语料用于所有 profile。批量生成时，工具会把 `request.model` 显式覆盖为当前
profile 的官方 repository，并把这一规则写入 manifest；因此不会因为原始语料最初
使用 GLM model 字段而污染其他模型结果。

UltraChat 的固定来源 revision、Parquet 分片、抽样规则、许可证和内容哈希记录在
`datasets/manifests/`。原始大 Parquet 不提交进仓库。

## 已提交的黄金结果

结果布局：

```text
datasets/results/by-profile/
├── deepseek-v3/
├── deepseek-v3.2/
├── deepseek-v4/
├── kimi-k2.6/
├── glm-5.1/
├── glm-5.2/
├── minimax-m2.7/
└── manifest.json

每个 profile 目录：
├── handwritten.jsonl
├── combinatorial.jsonl
├── ultrachat.jsonl.gz
└── manifest.json
```

当前提交的是便于快速下载和验证的小规模基准：每模型保留全部 300 条手写、全部
2,000 条组合和 UltraChat 的确定性前 1,000 条，共 3,300 条。原始 UltraChat
请求文件仍保留 10,000 条。

| Profile | 总数 | 成功 | 稳定错误 |
|---|---:|---:|---:|
| `deepseek-v3` | 3,300 | 2,579 | 721 |
| `deepseek-v3.2` | 3,300 | 2,775 | 525 |
| `deepseek-v4` | 3,300 | 3,270 | 30 |
| `kimi-k2.6` | 3,300 | 3,264 | 36 |
| `glm-5.1` | 3,300 | 3,264 | 36 |
| `glm-5.2` | 3,300 | 3,264 | 36 |
| `minimax-m2.7` | 3,300 | 3,264 | 36 |
| **合计** | **23,100** | **21,680** | **1,420** |

每个 profile manifest 都明确记录 `selection.ultrachat_limit: 1000`，不能把当前结果
误报为每模型 12,300 条全量基准。

错误不是“数据生成失败”。一部分是故意构造的非法 OpenAI 请求，另一部分是请求
结构合法、但被某个模型官方模板拒绝的模型适配边界。不同模型错误数量不同正是需要
保留和对拍的行为差异。

## 结果字段

成功记录的主要字段：

| 字段 | 含义 |
|---|---|
| `case_id` | 与请求集逐条对应的稳定 ID |
| `status` | `ok` |
| `request_sha256` | 实际送入该 profile 的规范 JSON 哈希 |
| `model_profile` | 使用的严格 profile |
| `renderer` | `hf`、`deepseek_v32` 或 `deepseek_v4` |
| `rendered_text` | encode 前的精确 Unicode 文本 |
| `rendered_text_sha256` | rendered text 的 UTF-8 SHA-256 |
| `token_ids_length` | 最终文本 token 数量 |
| `token_ids_sha256` | 完整 token ID 序列的稳定 SHA-256 |
| `token_ids` | 仅手写成功样本默认保存完整数组 |
| `diagnostics` | content format、generation prompt 等实际渲染参数 |

错误记录示例：

```json
{
  "case_id": "invalid.0001",
  "status": "error",
  "model_profile": "glm-5.2",
  "renderer": "hf",
  "request_sha256": "...",
  "error": {
    "stage": "request_validation",
    "type": "validation_error",
    "message": "..."
  }
}
```

稳定的错误阶段包括：

- `profile_resolution`：未知模型或请求 model 与 oracle profile 不匹配；
- `request_validation`：OpenAI 风格请求结构不合法；
- `processor_required`：需要未纳入本项目的多模态 processor；
- `render_or_encode`：模型模板或 tokenizer 拒绝该输入。

`stage` 和 `type` 用于跨实现对拍；依赖库产生的详细 `message` 主要用于诊断，不应
轻易当作跨版本稳定协议。

## 如何对拍你的另一套实现

推荐按以下顺序比较同一个 `case_id`：

1. 比较 `status`。如果一个成功、一个失败，先检查协议校验和模型适配边界。
2. 比较 `rendered_text` 的 UTF-8 字节。不同说明问题在 encode 之前。
3. 比较 `token_ids_length`。这是你最关心的最终数量。
4. 比较 `token_ids_sha256`。数量相同但哈希不同，说明 token 序列仍不一致。
5. 对手写样本比较完整 `token_ids`，定位第一个分叉位置。
6. 查看 `diagnostics` 或错误 `stage/type`，缩小到 content normalization、template
   参数、模型 renderer 或 tokenizer。

不要只用 rendered text 的“肉眼看起来一样”作为判据。不可见空格、换行、Unicode
字符和特殊 token 都可能改变编码结果。

## 按模型生成、校验和测试

只生成一个模型的小规模结果：

```bash
.venv/bin/python -m tools.generate_results \
  --model deepseek-v4 \
  --ultrachat-limit 1000
```

校验一个模型：

```bash
.venv/bin/python -m tools.verify_results --model deepseek-v4
```

校验全部已生成结果：

```bash
.venv/bin/python -m tools.verify_results --model all
.venv/bin/python -m tools.build_result_manifest
```

只运行一个模型的测试：

```bash
.venv/bin/pytest --model deepseek-v4 -q
```

选择多个模型：

```bash
.venv/bin/pytest \
  --model glm-5.1 \
  --model glm-5.2 \
  -q
```

运行全部模型：

```bash
.venv/bin/pytest --model all -q
```

默认不传 `--model` 时只运行 core 测试；生成和校验命令默认选择 GLM-5.2。只有显式
传入 `all` 才会覆盖全部 profile，避免日常修改每次都运行所有模型。

## 可复现性和防止半成品

- 普通 JSONL 使用同目录临时文件写入，flush/fsync 后原子替换；
- UltraChat 使用 gzip，固定 `mtime=0` 和空 filename，保证重复生成字节一致；
- 完整 profile manifest 存在时拒绝覆盖，避免意外改变黄金基准；
- 如果只完成部分分片，续跑前会核对条数、`case_id` 和请求哈希；
- manifest 记录各结果文件的 size、压缩文件 SHA-256 和解压内容 SHA-256；
- 聚合 manifest 记录七个 profile manifest 的 SHA-256 和汇总计数。

独立重算后逐字节比较：

```bash
.venv/bin/python -m tools.verify_reproducibility \
  --expected datasets/results/by-profile/glm-5.2 \
  --actual /path/to/clean-results/glm-5.2
```

当前七个 profile 均已从空目录重新计算，并通过
`handwritten.jsonl`、`combinatorial.jsonl`、`ultrachat.jsonl.gz` 和
`manifest.json` 四个文件的逐字节比较。

## 验证工具

```bash
# 校验原始请求数量、ID 和结构
.venv/bin/python -m tools.verify_requests

# 校验每个结果文件、请求哈希、文本哈希、token 哈希和 manifest
.venv/bin/python -m tools.verify_results --model all

# 构建七模型聚合 manifest
.venv/bin/python -m tools.build_result_manifest

# 与独立的固定上游参考路径做差分抽样
.venv/bin/python -m tools.verify_upstream_parity --ultrachat-sample 1000

# 全模型测试
.venv/bin/pytest --model all -q
```

测试覆盖 profile 注册、资产完整性、OpenAI 请求 contract、tools、thinking、
DeepSeek 专用路径、多模态边界、来源追踪、确定性 gzip、结果生成、结果验证和字节级
复现。

## 当前已完成与后续工作

已经完成：

- 七个指定官方主版本的严格 profile；
- vLLM 文本输入链路提取与来源映射；
- DeepSeek V3.2/V4 专用 renderer；
- 官方 tokenizer/template/config 文本资产固定与校验；
- 300 + 2,000 + 10,000 的共享原始请求集；
- 每模型 300 + 2,000 + 1,000 的已提交快速黄金结果；
- 按模型选择的生成、校验和测试；
- 七模型空目录字节级复现验证。

尚未完成或有意不做：

- 已提交结果尚未扩展到每模型完整 10,000 条 UltraChat；
- UltraChat 当前只能按整个分片续跑，未来扩全量前建议增加分片内部断点；
- 不实现需要真实图片/音频处理器才能确定的多模态 token 数；
- 不保证未来模型 revision、未来 vLLM 或线上 API 的计数不发生变化；升级必须新增
  基线或明确重新生成，不能覆盖后假装结果未变。

## 许可、来源和安全边界

- vLLM 提取代码保留 Apache-2.0 标头；完整许可证位于 `vendor/vllm/LICENSE`；
- 固定 vLLM commit 和上游文件 SHA-256 位于 `vendor/vllm/`；
- 模型文本资产 manifest 位于 `models/manifests/`；
- UltraChat 来源 revision、分片、许可证和哈希位于 `datasets/manifests/`；
- 仓库不包含 `.safetensors`、模型 `.bin`、`.pt`、`.pth` 或 `.gguf` 权重；
- Kimi 自定义 tokenizer 源码只从固定、已校验的本地资产加载。

## 进一步阅读

- `docs/request-pipeline-walkthrough.md`：用一份完整的天气工具请求讲解公共校验、
  规范化和七个模型的实际分叉；
- `docs/runtime-request-pipeline.md`：按运行顺序解释请求链路、公共规范化、模型分叉和
  新模型维护方式；
- `vendor/vllm/EXTRACTION.md`：vLLM 提取边界与文件映射；
- `vendor/vllm/coverage.json`：机器可读的 profile 覆盖门禁；
- `docs/superpowers/specs/2026-08-14-vllm-text-input-oracle-design.md`：最初的
  GLM-5.2 oracle 设计；
- `docs/superpowers/specs/2026-08-15-selected-model-renderers-design.md`：七模型扩展
  设计；
- `models/profiles.json`：实际生效的严格模型注册表；
- `datasets/results/by-profile/manifest.json`：已提交基准的汇总事实。
