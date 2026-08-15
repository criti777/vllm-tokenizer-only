# 指定模型文本渲染与分层基准设计

日期：2026-08-15

## 1. 目标

本项目提供一个不加载模型权重的离线 oracle，完整覆盖以下链路：

```text
OpenAI-compatible request
  -> model profile resolution
  -> request validation
  -> message normalization
  -> model-specific rendering
  -> tokenizer encode
  -> rendered text + token IDs/count/hash
```

实现以固定版本的 vLLM 文本预处理行为和模型官方 tokenizer/template 为基准。它既可作为另一套实现的对拍 oracle，也可直接作为轻量文本预处理库使用。

## 2. 范围

只支持已确认的官方主版本，不承诺覆盖 vLLM 的全部模型：

| Profile ID | 官方模型仓 | 渲染路径 |
|---|---|---|
| `deepseek-v3` | `deepseek-ai/DeepSeek-V3` | vLLM 通用 HF chat-template 路径 |
| `deepseek-v3.2` | `deepseek-ai/DeepSeek-V3.2` | vLLM `deepseek_v32` 专用 tokenizer、encoding、renderer |
| `deepseek-v4` | `deepseek-ai/DeepSeek-V4-Flash` | vLLM `deepseek_v4` 专用 tokenizer、encoding、renderer |
| `kimi-k2.6` | `moonshotai/Kimi-K2.6` | vLLM 通用 HF 路径及其文本侧特殊分支 |
| `glm-5.1` | `zai-org/GLM-5.1` | vLLM 通用 HF chat-template 路径 |
| `glm-5.2` | `zai-org/GLM-5.2` | vLLM 通用 HF chat-template 路径 |
| `minimax-m2.7` | `MiniMaxAI/MiniMax-M2.7` | vLLM 通用 HF/模型 tokenizer 兼容路径 |

具体模型仓 revision、模型资产 SHA-256、vLLM commit 和提取文件 SHA-256 必须进入 manifest，不能依赖浮动的 `main`。

### 非目标

- 不加载模型权重，不执行推理。
- 不下载或解码图片、音频、视频。
- 不计算 pixel values、音频特征、视觉 token expansion 或多模态 embedding。
- 不保证兼容上述模型的任意第三方微调、量化、镜像或模板改版。
- 不把 Transformers 的通用 `apply_chat_template` 当作 DeepSeek V3.2/V4 专用路径的替代品。

## 3. 架构

### 3.1 统一入口

```python
oracle = TextOracle.from_model("deepseek-v4", assets_root=assets_root)
result = oracle.process(
    request=openai_request,
    case_id="tools-001",
    include_token_ids=True,
)
```

所有 profile 共享同一输入和输出接口。调用者不直接实例化 renderer，也不根据模型名写分支。

### 3.2 ModelProfile 注册表

每个 profile 是不可变配置，至少包含：

- 稳定 `profile_id` 与允许的 OpenAI `model` aliases；
- 官方仓库与固定 revision；
- vLLM renderer 类型及其固定上游来源；
- tokenizer/template/config 资产清单及哈希；
- 支持的请求能力，如 tools、thinking、content parts；
- 多模态文本占位行为与 `processor_required` 条件；
- 已知、明确、可测试的兼容性差异。

未知模型、模糊 alias 或资产不匹配必须失败，禁止静默回退到 GLM 或通用 HF 路径。

### 3.3 提取边界

从固定 vLLM commit 提取文本链路所需的最小闭包：

- OpenAI chat message/content/tool schema 的必要部分；
- 消息规范化和 conversation 构造；
- renderer 选择与参数传递；
- HF chat-template 调用封装；
- DeepSeek V3.2 专用 tokenizer、encoding、renderer；
- DeepSeek V4 专用 tokenizer、encoding、renderer；
- 文本 tokenization 与结果封装。

GPU executor、调度、KV cache、模型加载、采样、输出解析器不进入提取范围。提取代码保留上游许可证、来源路径、commit 和差异说明。

## 4. 处理语义

### 4.1 请求与规范化

输入接受项目声明支持的 OpenAI-compatible chat completion 请求字段。规范化负责：

- 校验 message role、content part、tool/tool_choice 等结构；
- 保留消息顺序和内容顺序；
- 将字符串 content 与结构化 content parts 转为 renderer 所需表示；
- 按 vLLM 对应路径处理 tools、tool calls、tool results 和 thinking 参数；
- 区分“协议有效但模型模板不接受”与“请求本身无效”。

不识别的字段不能悄悄改变 rendered text；应按固定策略忽略或报错，并由测试锁定。

### 4.2 模型专用路径

- DeepSeek V3 使用其官方模板的通用 HF 路径。
- DeepSeek V3.2 使用 vLLM 的 DSML 工具格式、thinking/history 规则及专用 token encoding。
- DeepSeek V4 使用 vLLM 的专用 content block、tool message 合并、tool result 排序、reasoning effort 与 task token 规则。
- Kimi K2.6 只保留会影响最终文本/token IDs 的特殊分支；媒体处理本体不纳入。
- GLM 5.1/5.2 分别绑定自身官方资产，不因系列相近而共享未经哈希证明相同的词表或模板。
- MiniMax M2.7 固定并审计 tokenizer 资产；不得在生成基准时执行未固定、未审计的远程 Python。

### 4.3 多模态边界

对于 OpenAI 请求中的图片等 content part，oracle 只复现 vLLM 在进入模型 processor 前、会影响文本 prompt 的行为，例如占位符、分隔符或模板标记。

若仅凭 tokenizer/template 无法确定最终文本 token 序列，返回稳定错误阶段 `processor_required`，而不是伪造 token 数量。媒体字节、pixel values、媒体 embedding 和模型侧视觉 token 数不属于本项目统计口径。

### 4.4 编码和输出

成功结果至少包含：

```json
{
  "case_id": "tools-001",
  "model_profile": "deepseek-v4",
  "renderer": "deepseek_v4",
  "status": "ok",
  "rendered_text": "...",
  "rendered_sha256": "...",
  "token_count": 123,
  "token_ids_sha256": "...",
  "token_ids": [1, 2, 3]
}
```

`token_ids` 可按数据层级省略；`token_count` 和两个哈希始终保留。rendered text 的哈希按 UTF-8 bytes 计算，token IDs 哈希使用项目规定的确定性整数序列编码。

错误结果包含 `model_profile`、稳定 `error_stage`、稳定错误代码和可读信息。阶段集合为：

- `profile_resolution`
- `request_validation`
- `message_normalization`
- `template_render`
- `encode`
- `processor_required`
- `asset_integrity`

## 5. 资产与供应链

- vLLM 使用固定 commit，不从已安装 wheel 猜测行为。
- 官方模型资产逐文件下载并检查 HTTP 状态、非空、声明哈希和实际哈希。
- manifest 记录来源 URL、revision、etag（如有）、大小和 SHA-256。
- 生成和验证默认离线；缺失资产时给出明确下载命令，不隐式联网。
- 模型仓需要 remote code 时，只纳入固定 revision、已审阅且确有文本渲染必要的最小文件，并记录许可证与哈希。
- 资产漂移、上游来源漂移和未知 renderer 均为硬失败。

## 6. 分层完整基准

### 6.1 共享请求层

共享语义请求集用于发现跨模型差异：

- 300 条手写黄金边界用例；
- 2,000 条确定性组合用例；
- 10,000 条固定来源、固定抽样算法的 UltraChat 请求。

共享用例覆盖 system/user/assistant、多轮、Unicode、空值、换行、长文本、tools、tool calls、tool results、thinking、结构化 content 和多模态文本边界。

### 6.2 逐模型结果层

每个 profile 独立输出：

```text
data/results/<profile>/handwritten.jsonl
data/results/<profile>/combinatorial.jsonl
data/results/<profile>/ultrachat.jsonl.gz
data/results/<profile>/manifest.json
```

- 手写集保存完整 rendered text 和完整 token IDs。
- 组合集保存 rendered text、token count 和哈希；完整 IDs 可配置生成。
- UltraChat 使用确定性 gzip 保存 rendered text、token count 和哈希，压缩时间戳固定。
- manifest 同时记录压缩文件与解压 JSONL 的 SHA-256、条数、成功/错误统计和错误阶段分布。

原始请求 ID 与结果 ID 必须一一对应；禁止漏项、重复和顺序漂移。

## 7. 测试设计

### 7.1 按模型选择

默认测试只运行不依赖完整模型资产的 core tests。模型测试显式选择：

```bash
pytest --model glm-5.2
pytest --model deepseek-v4
pytest --model deepseek-v3.2 --model kimi-k2.6
pytest --model all
```

生成与验证工具使用相同选择语义：

```bash
python -m tools.generate_results --model glm-5.2
python -m tools.verify_results --model glm-5.2
```

单模型运行不得下载、加载或验证其他模型的大型资产。

### 7.2 测试层次

1. Core 单元测试：profile 解析、schema、哈希、原子写入、错误阶段。
2. Renderer 单元测试：逐模型黄金请求，精确比较 rendered UTF-8 bytes 与 token IDs。
3. vLLM parity：在固定上游环境中对相同请求逐条运行原始 vLLM 路径和提取路径。
4. Corpus 回归：验证 300/2,000/10,000 三层结果、统计与 manifest。
5. 交叉模型测试：确认同一请求在不同 profile 下不会误用模板或词表。
6. Fuzz/property 测试：随机 Unicode、JSON、messages 和 tools，不崩溃、种子可复现、与上游一致。

## 8. 覆盖与漂移控制

仓库维护 machine-readable coverage manifest，建立以下映射：

```text
vLLM upstream file/hash
  -> extracted file
  -> renderer/profile
  -> behavior tests
  -> generated result manifest
```

CI 检查：

- 每个支持 profile 都有固定模型资产和至少一组黄金测试；
- 每个提取文件都有上游来源和许可证；
- renderer 自动路由与 profile 声明一致；
- 不允许未经声明的 generic fallback；
- 上游文件哈希变化时 parity 状态失效，必须重新审阅和生成。

## 9. 迁移顺序

1. 保持现有 GLM 5.2 接口和 12,300 条结果可验证。
2. 引入 ModelProfile 注册表和按模型 CLI，但先让 GLM 5.2 通过原有黄金结果。
3. 加入 GLM 5.1、DeepSeek V3、Kimi K2.6、MiniMax M2.7 的固定 HF 路径。
4. 提取并验证 DeepSeek V3.2 专用路径。
5. 提取并验证 DeepSeek V4 专用路径。
6. 为每个 profile 生成分层完整基准并执行字节级可复现性验证。
7. 更新 provenance/coverage manifest、文档和 CI。

## 10. 验收标准

- 七个 profile 均可由统一 API 与 CLI 独立运行。
- 所有支持的有效请求都输出确定的 rendered text 和文本 token 统计，或稳定的阶段化错误。
- 手写黄金用例与固定 vLLM 路径逐字节、逐 token ID 一致。
- 每个 profile 均有完整的 300/2,000/10,000 分层结果和可验证 manifest。
- 同一输入重复生成的 JSONL/gzip/manifest 哈希一致。
- `--model <profile>` 只触碰目标模型资产；`--model all` 才运行全量。
- 无模型权重、GPU runtime、媒体处理或推理依赖。
- DeepSeek V3.2/V4 不经过未经授权的通用 HF 回退。
- 官方资产、vLLM 上游、提取文件、测试和结果之间可追溯。
