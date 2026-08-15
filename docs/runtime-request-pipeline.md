# 运行时请求链路与模型分叉

本文说明一条 OpenAI Chat Completions 风格请求如何经过本项目，最终得到
`rendered_text`、`token_ids` 和 `token_ids_length`；同时说明哪些逻辑由所有模型
共用、哪些逻辑按模型分叉，以及新增模型时应该增加配置、资产还是专用代码。

如果希望跟随一份包含 reasoning、tools 和 tool result 的具体请求逐步阅读，请参见
[`request-pipeline-walkthrough.md`](request-pipeline-walkthrough.md)。

## 总览

以 GLM-5.2 为例：

```python
oracle = TextOracle.from_model(
    "glm-5.2",
    assets_root=Path("model_assets"),
)

result = oracle.process(request, case_id="001")
```

完整流程如下：

```text
① 选择模型
   src/vllm_text_oracle/oracle.py
              │
              ▼
② 查询模型注册表
   models/profiles.json
              │
              ├── 官方模型仓
              ├── 固定 revision
              ├── 使用哪个 renderer
              ├── 是否支持 content parts
              └── 对应哪个资产 manifest
              │
              ▼
③ 找到该模型的独立资产目录
   model_assets/<模型仓>/<revision>/
              │
              ▼
④ 使用 manifest 校验资产
   models/manifests/<profile>.json
              │
              ▼
⑤ 加载 tokenizer，并选择 renderer
   src/vllm_text_oracle/renderers.py
              │
       ┌──────┴────────┐
       ▼               ▼
   通用 HF 路径     模型专用路径
                     DeepSeek V3.2/V4
       │               │
       └──────┬────────┘
              ▼
⑥ 接收 OpenAI 请求
   TextOracle.process(request)
              │
              ▼
⑦ 公共协议校验和规范化
   vendor/vllm/extracted/protocol.py
   vendor/vllm/extracted/chat_utils.py
              │
              ▼
⑧ 模型模板渲染与 tokenizer encode
   得到 rendered_text 和 token_ids
              │
              ▼
⑨ 统一返回
   rendered_text
   token_ids_length
   token_ids_sha256
   token_ids（可选）
```

下面按运行顺序展开。

## 第一阶段：初始化模型

### 1. 统一入口

所有模型都从这里进入：

```text
src/vllm_text_oracle/oracle.py
```

调用：

```python
TextOracle.from_model("glm-5.2", assets_root=...)
```

入口不会为每个模型复制一套完整流程。它首先读取统一注册表：

```text
models/profiles.json
```

### 2. 模型注册表

每个模型在注册表中有一条 profile，概念上类似：

```json
{
  "profile_id": "glm-5.2",
  "repository": "zai-org/GLM-5.2",
  "revision": "固定的 commit",
  "renderer": "hf",
  "asset_manifest": "models/manifests/glm-5.2.json",
  "capabilities": {
    "tools": true,
    "thinking": true,
    "content_parts": false
  }
}
```

它告诉程序：

- 哪些 model 名称属于这个 profile；
- 官方模型仓和不可变 revision；
- 使用通用 HF renderer 还是专用 renderer；
- 模型资产和 manifest 在哪里；
- 当前显式声明的模型能力。

注册表只负责选择和路由，不保存词表，也不执行 render。

## 第二阶段：统一找到模型资产

### 3. 每个模型有自己的资产目录

目录约定统一为：

```text
model_assets/
  <组织名>--<模型名>/
    <revision>/
      tokenizer 配置
      词表
      chat template
      special tokens
      其他必要文件
```

例如：

```text
model_assets/
  zai-org--GLM-5.2/
    b4734de.../

  deepseek-ai--DeepSeek-V3.2/
    a7e62ac.../

  moonshotai--Kimi-K2.6/
    7eb5002.../
```

路径由公共代码统一计算：

```python
asset_path = (
    assets_root
    / profile.repository.replace("/", "--")
    / profile.revision
)
```

因此，不需要为每个模型手写文件路径读取代码。

## 第三阶段：校验模型文件

每个模型都有独立 manifest：

```text
models/manifests/
  deepseek-v3.json
  deepseek-v3.2.json
  deepseek-v4.json
  kimi-k2.6.json
  glm-5.1.json
  glm-5.2.json
  minimax-m2.7.json
```

manifest 记录该模型应该有哪些文件、每个文件的大小和 SHA-256。加载模型前统一由：

```text
src/vllm_text_oracle/assets.py
```

进行校验。

它不负责 render 或 encode，只负责防止：

- 文件缺失或被改动；
- tokenizer/template revision 混用；
- Kimi 自定义 Python 文件被篡改；
- 意外混入模型权重或未声明代码。

## 第四阶段：第一次模型分叉

第一次真正的行为分叉发生在：

```text
src/vllm_text_oracle/renderers.py
```

注册表的 `renderer` 当前可以是：

```text
hf
deepseek_v32
deepseek_v4
```

对应：

```text
                    ┌── HFRenderer
profile.renderer ───┼── DeepSeekV32Renderer
                    └── DeepSeekV4Renderer
```

### 大部分模型走统一 HF 路径

当前这些模型共享同一套 HF Python 代码：

```text
DeepSeek-V3
Kimi-K2.6
GLM-5.1
GLM-5.2
MiniMax-M2.7
```

统一使用：

```text
vendor/vllm/extracted/hf_renderer.py
```

但共享 Python 代码不等于渲染结果相同。差异来自每个模型自己的：

```text
chat template
tokenizer config
special tokens
词表
```

即：

```text
同一个 HFRenderer
+ 不同模型资产
= 不同 rendered_text 和 token_ids
```

### 特殊模型才有专用代码

当前 DeepSeek-V3.2 和 DeepSeek-V4 走专用 renderer，对应：

```text
vendor/vllm/extracted/deepseek_v32_encoding.py
vendor/vllm/extracted/deepseek_v4_encoding.py
```

以及公共适配入口：

```text
src/vllm_text_oracle/renderers.py
```

这是因为 vLLM 对它们存在普通 HF chat template 之外的专用规则，例如：

- DSML tools；
- tool result 顺序；
- reasoning effort 映射；
- `wo_eos`；
- 特殊结束符处理。

所以，大部分模型只有独立配置和独立资产，不需要独立 Python 文件；只有 vLLM 本身
存在特殊路径的模型，才增加专用 renderer/encoding 代码。

## 第五阶段：请求进入后的公共部分

模型初始化完成后，实际请求进入：

```python
oracle.process(request)
```

首先统一经过：

```text
src/vllm_text_oracle/oracle.py
```

这里负责：

1. 计算请求哈希；
2. 检查请求中的 model 是否已注册；
3. 检查请求 model 是否与当前 oracle profile 一致；
4. 根据 profile capability 做多模态能力门禁；
5. 进入公共 OpenAI/vLLM 请求结构校验；
6. 调用初始化时已经选好的 renderer；
7. 统一包装成功或错误结果。

## 第六阶段：公共协议校验与规范化

### 公共请求结构

请求结构主要由：

```text
vendor/vllm/extracted/protocol.py
```

处理。当前显式建模的字段包括：

- `messages`；
- `model`；
- `tools`、`tool_choice`；
- `reasoning_effort`；
- `add_generation_prompt`；
- `continue_final_message`；
- `add_special_tokens`；
- `documents`；
- `chat_template`；
- `chat_template_kwargs`；
- `chat_template_content_format`。

它还会做公共关系校验，例如
`add_generation_prompt` 和 `continue_final_message` 不能同时为 true。

### 公共消息规范化

消息内容主要由：

```text
vendor/vllm/extracted/chat_utils.py
```

处理，包括：

- 字符串 content 和 content-parts 数组；
- text、refusal、thinking 等文本 part；
- assistant tool calls；
- tool result；
- tool call arguments JSON；
- `developer`、`assistant`、`tool` 等 role；
- `name`、`task`、`wo_eos`、`prefix`、`mask`；
- `content_blocks`、`reasoning`、`reasoning_content`。

### 校验/规范化究竟是否与模型有关

答案是：**语义上有关，但当前不是每个模型一套完整 validator/normalizer。**

当前设计分为三层：

#### 第一层：公共协议结构

先用一套 OpenAI/vLLM 文本字段超集检查基本形态。比如所有模型都要求 message 有
非空 role，所有模型都用同一个 `ChatCompletionRequest` 承载公共字段。

这层解决“请求是不是一个可进入文本链路的 Chat 请求”，不决定某个字段对某个模型
最终意味着什么。

#### 第二层：公共保真规范化

规范化器会统一寻找并保留已知字段，例如：

```text
wo_eos
prefix
mask
reasoning
content_blocks
tool_calls
```

它的职责是不要过早丢掉可能被后续模型路径使用的信息，而不是让每个字段对所有模型
都生效。

例如 `wo_eos` 可以被统一复制到规范化 message，但当前真正解释它的是 DeepSeek V4
专用路径；其他模型模板如果不读取它，它就不会改变结果。

#### 第三层：模型 renderer/template 解释语义

选择模型后，由该模型 renderer 或 chat template 决定：

- 哪些字段真正生效；
- 字段如何映射为模板变量；
- tools 如何编码；
- thinking/reasoning 如何处理；
- 哪些 role 顺序被接受；
- 哪些请求会被模型模板拒绝。

DeepSeek V3.2/V4 在专用 renderer 中显式解释字段；HF 模型则主要由各自官方 chat
template 解释统一传入的 messages、tools 和 template kwargs。

此外，profile 层当前还会在进入模板之前根据 `content_parts` capability 对媒体请求
做显式门禁。这是一个已经实现的模型相关前置判断。

### 模型专属字段是怎样进入后续处理的

当前有三种方式：

1. **已经建模的顶层字段**：例如 `reasoning_effort`，由公共协议读取，再合并进
   template kwargs；
2. **已知的 message 字段**：例如 `wo_eos`、`reasoning`，由公共规范化器保留，再由
   renderer/template 决定是否使用；
3. **模型模板扩展字段**：通过 `chat_template_kwargs` 显式传给模型模板。

`ChatCompletionRequest` 为兼容 OpenAI 请求允许额外字段，但要注意：**未知字段被接受
不代表它会自动影响 render。** 如果它没有进入上述三条路径，就只是兼容性保留，当前
结果不会使用它。

因此，新模型出现专属字段时必须判断：

- 它只是新的 template kwarg：调用方可先通过 `chat_template_kwargs` 传入；
- 它是多个模型共享的正式字段：扩展 `protocol.py` 的公共字段和参数合并；
- 它是需要保留的 message 字段：扩展 `chat_utils.py`；
- 它改变校验规则或编码算法：增加模型能力门禁、profile-specific validation，或者
  新增专用 renderer。

当前代码已经有 profile capability 门禁和 renderer 分叉，但还没有“每个 profile
挂一个独立 validator 类”的通用插件机制。只有确实出现模型专属校验需求时，才应该
增加这一层，而不是预先为所有模型复制 validator。

## 第七阶段：第二次模型分叉——渲染

完成公共校验后，调用初始化时已经选定的 renderer。

### 通用模型

```text
规范化 messages
    ↓
HFRenderer
    ↓
该模型自己的 chat template
    ↓
rendered_text
```

例如：

```text
GLM-5.2：HFRenderer + GLM-5.2 chat template
Kimi-K2.6：HFRenderer + Kimi-K2.6 chat template
MiniMax-M2.7：HFRenderer + MiniMax-M2.7 chat template
```

Python 代码相同，模板资产不同。

### 特殊模型

```text
规范化 request
    ↓
DeepSeekV32Renderer / DeepSeekV4Renderer
    ↓
专用编码逻辑
    ↓
rendered_text + token_ids
```

这些 profile 不允许退回普通 HF 路径。

## 第八阶段：encode 与统一返回

### 每个模型是否需要相同文件

不保证相同。不同 tokenizer 可能使用不同文件组合，例如：

```text
模型 A：tokenizer.json + tokenizer_config.json
模型 B：vocab.json + merges.txt + tokenizer_config.json
模型 C：tokenizer.model + tokenizer_config.json
模型 D：自定义 tokenizer.py + 词表 + tokenizer_config.json
```

项目不要求每个模型拥有完全相同的文件名，而是要求：

> 每个模型需要的文件全部放进自己的固定 revision 目录，由该模型 manifest 声明，
> 再由 AutoTokenizer 或专用 loader 从同一个目录统一加载。

通用读取方式是：

```python
AutoTokenizer.from_pretrained(
    asset_path,
    local_files_only=True,
)
```

Transformers 根据目录里的配置决定使用哪个 tokenizer 类、读取 tokenizer JSON 还是
vocab/merges、特殊 token 在哪里、chat template 在哪里，以及是否需要自定义
tokenizer 代码。因此，上层业务代码不用关心每个模型的词表文件是否一样。

最终所有 renderer 都返回相同的内部结构：

```text
RenderedPrompt
├── text
├── token_ids
└── diagnostics
```

再由 `src/vllm_text_oracle/contracts.py` 统一生成：

```text
status
rendered_text
rendered_text_sha256
token_ids_length
token_ids_sha256
token_ids（可选）
error（失败时）
```

## 新模型出来以后怎样维护

先判断新模型属于哪一类。

### 情况一：普通 HF 模型

如果固定 vLLM 对它也是：

```text
OpenAI messages
→ 通用规范化
→ tokenizer.apply_chat_template
→ tokenizer.encode
```

通常不需要新增模型 Python 文件，只需要：

1. 在 `models/profiles.json` 增加 profile；
2. 下载并固定该模型自己的 tokenizer/template/config；
3. 放进独立的 `model_assets/<repo>/<revision>/`；
4. 增加 `models/manifests/<profile>.json`；
5. 增加该模型黄金测试；
6. 更新 `vendor/vllm/coverage.json`。

即：

```text
新增普通模型
├── profiles.json 增加一条
├── manifests/new-model.json
├── model_assets/new-model/<revision>/
└── tests 增加该模型黄金值
```

不需要复制 `oracle.py`、`chat_utils.py` 或 `hf_renderer.py`。

### 情况二：模型增加了新的字段语义

根据字段所在层选择最小修改：

- 公共顶层字段：扩展 `protocol.py`；
- 公共 message 字段：扩展 `chat_utils.py`；
- 只传给模板的字段：使用或正式接入 `chat_template_kwargs`；
- 模型独占校验：增加 profile 能力规则或专用 validation hook；
- 字段改变整个编码算法：进入专用 renderer。

### 情况三：vLLM 对它有特殊路径

如果 vLLM 中存在该模型专用 renderer/tokenizer/encoding，则需要：

1. 完成普通模型所需的 profile 和资产；
2. 从固定 vLLM 版本提取专用文本逻辑；
3. 在 `src/vllm_text_oracle/renderers.py` 注册 renderer；
4. 在 profile 中填写专用 renderer 名称；
5. 增加专用行为和差分测试；
6. 增加来源、哈希和 coverage 映射；
7. 明确禁止错误回退到 HF renderer。

只有这种模型才需要新增专用 Python 文件。

## 最简单的架构理解

```text
                         公共入口
                  TextOracle.from_model
                            │
                            ▼
                   models/profiles.json
                            │
               ┌────────────┴────────────┐
               │                         │
          通用 HF 模型              特殊模型
               │                         │
        一个公共 HFRenderer      专用 Renderer 文件
               │                         │
               └────────────┬────────────┘
                            │
                   每个模型自己的资产目录
                            │
                chat template + tokenizer
                            │
                            ▼
                    rendered text
                            │
                            ▼
                       token IDs
```

关键结论：

1. 所有模型共享一个入口：`TextOracle`；
2. 所有模型共享 profile 解析、资产校验、公共请求结构和结果结构；
3. 请求规范化采用公共字段超集，但字段的最终语义可能属于具体模型；
4. 大部分模型共享同一个 HF renderer；
5. 大部分模型不需要“一个模型一个 Python 文件”；
6. 每个模型必须有独立 profile、manifest 和资产目录；
7. 模型资产文件名可以不同，统一交给 tokenizer loader 读取；
8. 只有 vLLM 明确存在特殊处理，或模型新增无法由公共模板表达的语义时，才扩展
   专用代码；
9. 业务调用方不需要知道分叉细节，只需传 model 和 OpenAI 请求。
