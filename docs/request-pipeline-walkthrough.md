# 一个请求如何走完七个模型：完整故事

本文用一份包含 reasoning、tools 和 tool result 的 OpenAI Chat Completions 请求，
完整说明公共校验、消息规范化、模型分叉、模板渲染和 tokenizer encode 分别做什么。

## 一、参与故事的七个模型

| Profile | 实际路径 |
|---|---|
| `deepseek-v3` | 通用 HF renderer + DeepSeek-V3 官方模板 |
| `deepseek-v3.2` | vLLM DeepSeek V3.2 专用 renderer |
| `deepseek-v4` | vLLM DeepSeek V4 专用 renderer |
| `kimi-k2.6` | 通用 HF renderer + Kimi 官方模板 + 自定义 tokenizer |
| `glm-5.1` | 通用 HF renderer + GLM-5.1 官方模板 |
| `glm-5.2` | 通用 HF renderer + GLM-5.2 官方模板 |
| `minimax-m2.7` | 通用 HF renderer + MiniMax 官方模板 |

先记住：

```text
七个模型
├── 五个走公共 HF Python 代码
└── 两个走 DeepSeek 专用 Python 代码
```

五个模型共享 HF renderer，不代表结果相同，因为它们加载不同的官方模板、词表和
特殊 token。

## 二、构造一份完整请求

```json
{
  "model": "zai-org/GLM-5.2",
  "messages": [
    {
      "role": "system",
      "content": "你是一个天气助手。"
    },
    {
      "role": "user",
      "content": "查询北京天气"
    },
    {
      "role": "assistant",
      "content": null,
      "reasoning_content": "用户想查询北京天气，需要调用工具。",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"北京\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": [
        {
          "type": "text",
          "text": "北京，晴，25℃"
        }
      ]
    },
    {
      "role": "user",
      "content": "穿什么合适？"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "reasoning_effort": "high",
  "add_generation_prompt": true,
  "continue_final_message": false,
  "temperature": 0.7,
  "max_tokens": 512
}
```

它包含 system/user/assistant/tool 多种角色、assistant reasoning、tool call、tool
result、tools 定义、reasoning effort、generation prompt 和采样参数。

## 三、先创建模型实例

```python
oracle = TextOracle.from_model(
    "glm-5.2",
    assets_root=Path("model_assets"),
)
```

此时还没有处理具体请求。

### 1. 查注册表

程序读取 `models/profiles.json`，得到：

```text
profile_id = glm-5.2
repository = zai-org/GLM-5.2
revision = b4734de...
renderer = hf
content_parts = false
```

所以从初始化开始，程序就知道它处理的是 GLM-5.2，而不是抽象的任意模型。

### 2. 找到模型资产

根据统一规则计算：

```text
model_assets/
  zai-org--GLM-5.2/
    b4734de.../
```

它不需要写 `if model == "glm-5.2": load_glm_files()`，而是统一根据 repository 和
revision 计算目录。

### 3. 校验资产

读取 `models/manifests/glm-5.2.json`，检查 tokenizer、chat template、文件大小和
SHA-256，避免资产缺失、被替换或混用 revision。

### 4. 选择 renderer

注册表中的 renderer 是 `hf`，所以创建 `HFRenderer`，并从 GLM-5.2 独立资产目录
加载 tokenizer。

到这里，一个 GLM-5.2 oracle 准备完成。

## 四、请求进入系统

```python
result = oracle.process(
    request,
    case_id="weather.001",
)
```

入口位于 `src/vllm_text_oracle/oracle.py`。

## 五、第一层：请求外围检查

### 1. 计算请求哈希

程序先计算 `request_sha256`，用于确认黄金结果和另一套实现比较的是同一个输入，也
用于安全续跑时发现请求变化。

### 2. 检查请求 model

当前 oracle 是 `glm-5.2`，请求中的 `zai-org/GLM-5.2` 也解析到同一 profile，因此
通过。

如果请求写成 `deepseek-ai/DeepSeek-V4-Flash`，会直接失败：

```text
stage = profile_resolution
type = model_profile_mismatch
```

这时不会进入 render。

### 3. 多模态能力门禁

程序检查 messages 中是否出现 image、video 或 audio content part。当前请求只有文本，
所以继续。

如果请求包含图片：

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "描述图片"},
    {
      "type": "image_url",
      "image_url": {"url": "https://example.com/a.jpg"}
    }
  ]
}
```

当前行为是：

```text
Kimi-K2.6 → 允许继续，由模板产生媒体占位符
其他当前模型 → processor_required
```

这是明确与模型有关的前置检查。

## 六、第二层：公共协议校验

请求进入 `vendor/vllm/extracted/protocol.py`：

```python
ChatCompletionRequest.model_validate(request)
```

这里没有按 GLM、Kimi、MiniMax 分别创建 validator，而是使用一套公共的
OpenAI/vLLM 文本请求字段超集。

### 公共协议明确认识的字段

```text
messages
model
tools
tool_choice
reasoning_effort
add_generation_prompt
continue_final_message
add_special_tokens
documents
chat_template
chat_template_kwargs
chat_template_content_format
```

### 公共校验的具体内容

`messages` 必须是列表，每个 message 必须有非空 role。`reasoning_effort` 必须是：

```text
none
minimal
low
medium
high
xhigh
max
```

`add_generation_prompt` 与 `continue_final_message` 不能同时为 true，因为前者要新增
assistant 开始标记，后者要继续最后一条已有消息，意图互相冲突。

公共校验不会完整判断：

- GLM 是否接受某种 role 顺序；
- Kimi 是否接受某种 tool 结构；
- DeepSeek-V4 如何解释 `wo_eos`；
- MiniMax 模板是否使用 `documents`；
- 某模型模板是否支持 developer role。

这些必须由后面的模型 renderer/template 决定。

## 七、为什么 temperature 也能通过

请求里还有：

```json
{
  "temperature": 0.7,
  "max_tokens": 512
}
```

当前请求模型使用 `extra="allow"`，所以完整 OpenAI 请求中不属于文本 prompt 链路的
额外字段可以通过。temperature、top_p、max_tokens、stream、stop 和 seed 控制生成、
采样或响应，不改变输入 prompt。

实际行为是：

```text
请求整体继续
temperature 不参与 render
max_tokens 不参与 render
messages 正常 render
最终正常 encode
```

拼错字段也可能静默通过，例如：

```json
{
  "reasoning_effrot": "high"
}
```

程序只认识 `reasoning_effort`，所以请求其他部分会正常 render/encode，但拼错字段完全
不起作用。这是宽松兼容的代价。

## 八、第三层：参数整理

公共请求对象建立后，会把参数整理为类似：

```python
template_kwargs = {
    "add_generation_prompt": True,
    "continue_final_message": False,
    "reasoning_effort": "high",
    "enable_thinking": True,
    "tools": [...],
}
```

当 `reasoning_effort != "none"` 且调用方没有显式指定 `enable_thinking` 时，会补充
`enable_thinking=True`。显式的 `chat_template_kwargs` 优先，不会被默认值覆盖。

### 已校验但当前不进入模板的字段

典型例子是 `tool_choice`。协议模型认识它，所以 `"tool_choice": "auto"` 能通过；
但当前 `template_kwargs()` 没有把它放进模板参数。

```text
tools       → 进入模板
tool_choice → 被请求模型接收，但当前不改变 rendered_text
```

## 九、第四层：messages 规范化

消息进入 `vendor/vllm/extracted/chat_utils.py`。这一层不是决定 GLM/Kimi/DeepSeek 的
最终格式，而是把不同 OpenAI 写法整理成稳定结构，并保留后续模型可能需要的信息。

### 1. 普通文本 content

字符串目标格式下：

```json
{"role": "user", "content": "查询北京天气"}
```

基本保持不变。OpenAI parts 目标格式下，可能规范成：

```json
{
  "role": "user",
  "content": [{"type": "text", "text": "查询北京天气"}]
}
```

### 2. content 为 null

```text
string format → ""
openai format → []
```

### 3. 文本 content parts

工具结果：

```json
{
  "role": "tool",
  "content": [{"type": "text", "text": "北京，晴，25℃"}]
}
```

在 string 格式中会成为：

```json
{"role": "tool", "content": "北京，晴，25℃"}
```

在 OpenAI parts 格式中保留数组结构。

### 4. assistant tool arguments

原始 arguments 是 JSON 字符串：

```json
{"arguments": "{\"city\":\"北京\"}"}
```

规范化后成为对象：

```json
{"arguments": {"city": "北京"}}
```

如果字符串不是合法 JSON，公共 Pydantic 外层可能先通过，但进入消息后处理时
`json.loads()` 会失败。这类错误通常归为 `render_or_encode`，而不是最前面的
`request_validation`。

### 5. 空 tool calls

`"tool_calls": []` 会被删除，避免模板把空数组误认为真实工具调用。

### 6. reasoning 字段

输入可以使用 `reasoning_content` 或 `reasoning`，规范化器会同时保留兼容形式，让
后面的模板或专用 renderer 读取自己认识的名称。

### 7. 模型相关字段先保留

规范化器还会保留：

```text
name
task
wo_eos
prefix
mask
content_blocks
```

保留不代表所有模型都使用。例如公共层会保留 `wo_eos`，DeepSeek-V4 专用编码会
明确解释它；HF 模板如果不读取它，它就没有效果。

## 十、到这里哪些部分仍然统一

到现在为止，大部分逻辑是统一的：

```text
请求哈希
model 匹配
公共结构校验
公共字段类型
模板参数默认值
messages 基本规范化
tool arguments JSON 解析
reasoning 字段兼容
```

已经出现的模型相关判断包括：

```text
profile model 匹配
content_parts 能力
初始化时选定的 renderer
按 renderer 决定的 content format
```

接下来进入真正的大分叉。

## 十一、七个模型正式分叉

### 路径 A：GLM-5.1

```text
公共请求校验
→ 公共 messages 规范化
→ HFRenderer
→ GLM-5.1 官方 chat template
→ GLM-5.1 tokenizer
→ token IDs
```

由 GLM-5.1 模板决定 system role、tools、assistant generation prompt、reasoning 参数
以及特殊 token 的具体格式。

### 路径 B：GLM-5.2

```text
公共规范化
→ HFRenderer
→ GLM-5.2 官方 chat template
→ GLM-5.2 tokenizer
```

GLM-5.1 和 5.2 的 Python renderer 相同，但模板和词表资产独立，所以同一请求可以
得到不同 rendered text 和 token IDs。GLM-5.2 模板还可能插入默认
reasoning-effort system 段。

因此它们不需要两个 renderer.py，但必须有独立 profile、revision、资产和黄金结果。

### 路径 C：DeepSeek-V3

```text
公共规范化
→ HFRenderer
→ DeepSeek-V3 官方 chat template
→ DeepSeek-V3 tokenizer
```

工具消息可能被模板渲染为 DeepSeek 自己的特殊 token，例如 tool outputs 的开始和结束
标记。差异主要来自官方模板和 tokenizer，而不是独立 Python renderer。

### 路径 D：Kimi-K2.6

Kimi 也使用 HFRenderer，但有两个特殊点。

第一，它加载固定模型仓中的官方自定义 tokenizer Python 代码，所以 profile 显式开启
`trust_remote_code`；代码只从固定、已校验的本地资产加载。

第二，Kimi profile 声明支持 content parts，因此 renderer 使用 OpenAI parts 格式。
图片 part 可以由官方模板变成 `<|media_*|>` 文本占位符，但项目不会下载图片或计算
视觉 embedding。

```text
公共协议校验
→ OpenAI content-parts 规范化
→ Kimi 官方模板
→ 媒体 part 变成官方文本占位符
→ Kimi 官方 tokenizer
→ token IDs
```

### 路径 E：MiniMax-M2.7

```text
公共规范化
→ HFRenderer
→ MiniMax 官方 chat template
→ MiniMax tokenizer
→ token IDs
```

如果某个 `chat_template_kwargs` 参数没有被 MiniMax 模板读取，它会正常进入模板调用，
但不会影响 rendered text。

### 路径 F：DeepSeek-V3.2

DeepSeek-V3.2 不走普通 HF 模板：

```text
公共协议校验
→ 强制 string content 规范化
→ DeepSeekV32Renderer
→ DeepSeek V3.2 专用 encoding
→ tokenizer.encode
```

如果请求包含 tools，renderer 会在 conversation 开头插入带 tools 的 system 结构，再由
专用 DSML encoding 输出工具协议。

V3.2 renderer 根据 `thinking` 或 `enable_thinking` 选择 thinking/chat mode。如果最后
一条消息是 user，还会设置专用的 `drop_thinking`。专用 encoding 先构造精确文本，
再使用该模型 tokenizer 编码，且 `add_special_tokens=False`。

### 路径 G：DeepSeek-V4

```text
公共校验
→ string 规范化
→ DeepSeekV4Renderer
→ DeepSeek V4 专用 encoding
→ tokenizer.encode
```

V4 对公共 `reasoning_effort` 做模型专属映射：

```text
none  → 关闭 thinking
max   → max
xhigh → max
其他非空 effort → high
```

公共规范化器负责保留 `wo_eos`，V4 专用 encoding 再决定不输出相应结束标记。V4
还会按照 tool call 的调用顺序整理 tool results，而不是简单相信请求中的结果顺序。

这些属于模型专属编码逻辑，不应放入所有模型共享的 `chat_utils.py`。

## 十二、同一个字段在不同模型中的命运

### `reasoning_effort`

| 模型 | 当前处理方式 |
|---|---|
| GLM-5.1 | 传入官方模板，由模板决定是否及如何使用 |
| GLM-5.2 | 传入官方模板，模板可能插入 reasoning 相关内容 |
| DeepSeek-V3 | 传入官方模板 |
| Kimi-K2.6 | 传入官方模板 |
| MiniMax-M2.7 | 传入官方模板 |
| DeepSeek-V3.2 | 主要转换成 thinking/chat mode |
| DeepSeek-V4 | 专用映射为关闭、high 或 max |

它是“公共字段 + 模型专属解释”。

### `wo_eos`

| 阶段 | 行为 |
|---|---|
| 公共校验 | 不做严格模型判断 |
| 公共规范化 | 发现并保留 |
| DeepSeek-V4 | 专用 encoding 使用 |
| 其他模型 | 只有模板读取才生效，否则无影响 |

它是“公共保留的模型扩展字段”。

### `tools`

tools 是多模型公共能力，但编码格式不同：

```text
GLM → GLM 官方模板格式
DeepSeek-V3 → DeepSeek HF 模板格式
Kimi → Kimi 模板格式
MiniMax → MiniMax 模板格式
DeepSeek-V3.2 → DSML 专用格式
DeepSeek-V4 → V4 专用格式
```

即：结构统一、语义大体统一、序列化格式按模型分叉。

### `tool_choice`

当前公共协议认识它，但没有把它放进 template kwargs，所以暂时不会改变项目生成的
rendered text。

### `chat_template_kwargs`

它是模型模板扩展的公共入口。模板读取的参数会生效；模板未读取的参数不会改变结果。
专用 DeepSeek renderer 也只读取自己明确支持的部分，不会自动把所有 kwargs 变成
编码规则。

### `add_special_tokens`

通用 HF 路径会把它传给 tokenizer encode；当前 DeepSeek V3.2/V4 专用路径明确使用
`add_special_tokens=False`。因此，字段存在于公共协议不代表所有 renderer 以相同方式
支持它。

## 十三、校验、规范化、渲染和 encode 的区别

### 校验

回答“这个请求能不能继续”：

- messages 是不是列表；
- message 有没有 role；
- reasoning effort 是否在允许枚举中；
- 两个互斥字段是否同时为 true；
- model 是否与当前 profile 匹配；
- 图片是否需要未纳入项目的 processor。

校验失败通常不会得到 token IDs。

### 规范化

回答“同一语义的不同 JSON 写法怎样整理成稳定结构”：

- null content 变成空字符串或空 parts；
- 字符串 content 变成文本或 text part；
- tool arguments JSON 字符串变成对象；
- reasoning_content 与 reasoning 做兼容保留；
- 空 tool_calls 删除；
- tool result 文本 parts 合并；
- wo_eos 等字段保留给后面的模型路径。

规范化不应该随意改变用户语义。

### 渲染

回答“这个模型最终看到的 prompt 文本是什么”，包括 system/user/assistant 标记、
thinking 标记、tools 描述、tool call 标记、generation prompt、BOS/EOS 等。

渲染高度依赖模型。

### Encode

回答“使用该模型词表，rendered text 对应哪些 token IDs”：

```text
rendered_text
→ 模型 tokenizer
→ [123, 456, 789]
```

## 十四、错误发生在哪一层

| 问题 | 发生阶段 | 是否 encode |
|---|---|---|
| messages 不是列表 | 公共请求校验 | 否 |
| message 缺 role | 公共请求校验 | 否 |
| reasoning effort 枚举非法 | 公共请求校验 | 否 |
| 两个互斥参数同时为 true | 公共请求校验 | 否 |
| request model 不匹配 | profile resolution | 否 |
| 模型需要图片 processor | capability gate | 否 |
| tool arguments 不是合法 JSON | message 规范化/渲染 | 否 |
| role 结构合法但模板不接受 | 模型渲染 | 否 |
| 未知额外字段 | 通常不报错 | 其他内容正常 encode |
| 模板未使用的 kwarg | 不报错 | 正常 encode，但字段无影响 |
| tokenizer 无法处理结果 | encode | 失败 |

## 十五、明天新增一个模型时怎么做

假设出现 `ExampleModel-1`，先查看固定 vLLM 中它如何处理。

### 情况一：普通 HF 模板

如果 vLLM 是：

```text
parse messages
→ apply_chat_template
→ tokenizer.encode
```

只需要增加：

```text
models/profiles.json 中的一条 profile
models/manifests/example-model.json
model_assets/example-model/<revision>/
该模型的测试和黄金值
coverage 映射
```

不需要增加 `example_model_normalizer.py` 或 `example_model_renderer.py`。

### 情况二：新增模板参数

如果官方模板直接读取 `chat_template_kwargs.analysis_mode`，调用方可以先通过
`chat_template_kwargs` 使用，不必立即修改公共协议。

如果要把它变成正式顶层字段，则在 `protocol.py` 建模，并合并进 template kwargs。

### 情况三：新增 message 字段

模型支持 `skip_end_token` 一类 message 字段时，需要在 `chat_utils.py` 中保留它，
否则它可能因额外字段兼容而通过外层校验，但规范化后不会影响结果。

### 情况四：模型专属校验

如果模型要求 tool message 前必须紧邻 assistant tool call，而官方模板不能提供清晰、
稳定的拒绝行为，就应增加 profile-specific validation，而不应把该限制强加给全部模型。

### 情况五：vLLM 有专用 encoding

如果 vLLM 为它实现专用 Renderer/Encoding，则像 DeepSeek V3.2/V4 一样：提取专用
文本逻辑、注册 renderer、增加 profile renderer 类型、加入专用测试，并禁止错误
回退到 HF。

## 十六、把整个故事压缩成一句话

项目既不是“做一套完全通用的规范化，后面只换 tokenizer”，也不是“每个模型从请求
校验开始复制一整套代码”，而是：

```text
公共 OpenAI/vLLM 请求结构校验
→ 公共字段超集规范化和信息保留
→ profile 能力门禁
→ 模型官方模板或专用 renderer 解释字段语义
→ 模型自己的 tokenizer encode
→ 统一结果结构
```

其中：

- messages、role、tools 基本结构属于公共层；
- reasoning_effort 属于公共字段，但模型解释不同；
- wo_eos 属于公共保留、特定模型使用的字段；
- tool_choice 当前能通过协议，但不影响渲染；
- temperature、max_tokens 能兼容接收，但不属于输入 token 链路；
- GLM、Kimi、MiniMax、DeepSeek-V3 主要靠各自官方模板分叉；
- DeepSeek-V3.2/V4 因 vLLM 有专用逻辑，在 Python renderer 层分叉；
- 每个模型最终都使用自己独立目录中的 tokenizer、template 和词表。
