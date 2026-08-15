# vLLM Text Input Oracle

一个不加载模型权重的离线参考实现与 GLM-5.2 对拍数据集：

```text
OpenAI Chat Completions JSON
→ vLLM 请求默认值/校验
→ messages、content parts、tools 规范化
→ 模型 chat template 适配与渲染
→ 官方 tokenizer encode
→ rendered_text、token_ids_length、哈希
```

当前基线固定为 vLLM `v0.26.0`（commit
`568afb3a13806beb53bb2e6bd518269357b237c0`）和
`zai-org/GLM-5.2` revision
`b4734de4facf877f85769a911abafc5283eab3d9`。仓库只包含 tokenizer、template
等小型资产，不包含模型权重，不依赖 PyTorch、CUDA 或 vLLM 服务进程。

## 目录

- `vendor/vllm/upstream/`：逐字节保留的 vLLM 上游源文件，用于审计。
- `vendor/vllm/extracted/`：文本 Chat Completions 可达路径的精简提取版。
- `src/vllm_text_oracle/`：稳定调用接口、结果结构和哈希定义。
- `datasets/requests/`：300 条手写、2,000 条组合、10,000 条 UltraChat 请求。
- `datasets/results/`：与请求逐条对应的 GLM-5.2 oracle 结果。
- `tools/`：请求构造、结果生成与完整性校验命令。

提取边界与上游文件映射见 `vendor/vllm/EXTRACTION.md`，完整设计与不支持项见
`docs/superpowers/specs/2026-08-14-vllm-text-input-oracle-design.md`。

## 安装与调用

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[test,data]'
```

单请求调用：

```python
from pathlib import Path
from vllm_text_oracle import TextOracle

revision = "b4734de4facf877f85769a911abafc5283eab3d9"
oracle = TextOracle.from_local_assets(
    Path("model_assets/zai-org--GLM-5.2") / revision
)
result = oracle.process(
    {
        "model": "zai-org/GLM-5.2",
        "messages": [{"role": "user", "content": "你好"}],
        "chat_template_kwargs": {"enable_thinking": True},
    },
    case_id="my-case-001",
    include_token_ids=True,
)
print(result.rendered_text)
print(result.token_ids_length)
print(result.token_ids)
```

## 使用基线对拍另一套实现

读取同一行的 `request`，依次比较：

1. `status`；可立即定位另一实现是否错误地接受/拒绝请求。
2. `rendered_text` 的 UTF-8 字节；不一致说明问题在校验、规范化或模板参数。
3. `token_ids_length`；这是最终计数判据。
4. `token_ids_sha256`；长度相同但哈希不同，说明 token 序列仍不一致。
5. 手写样本的 `token_ids`；用于定位第一个分叉 token。

结果文件位于：

```text
datasets/results/zai-org--GLM-5.2/
  b4734de4facf877f85769a911abafc5283eab3d9/
    handwritten.results.jsonl
    generated.results.jsonl
    ultrachat.results.jsonl
    manifest.json
```

生成/导入样本不保存完整 token IDs，以控制仓库体积，但保留精确长度和整个
token 序列的 SHA-256。手写成功样本保存完整 IDs。错误样本保存稳定的
`stage`/`type`；依赖库产生的详细 `message` 只用于诊断，不作为跨版本契约。

当前 36 条手写错误中，30 条带 `invalid` 标签，是故意构造的非法请求；另外
6 条带 `roles` 标签，是 OpenAI 结构层可以接收、但 GLM-5.2 官方模板拒绝的
内容边界样本。它们用于测试模型适配层，不能从基线中删掉。

## 生成、校验与测试

结果目录一旦出现完整 `manifest.json`，生成器会拒绝覆盖；若上次运行只完成了
部分分片，则只会在逐条核对 case ID 和请求哈希后安全续跑。

```bash
.venv/bin/python -m tools.generate_results
.venv/bin/python -m tools.verify_results \
  --results datasets/results/zai-org--GLM-5.2/b4734de4facf877f85769a911abafc5283eab3d9
.venv/bin/python -m tools.verify_requests
.venv/bin/python -m tools.verify_upstream_parity --ultrachat-sample 1000
.venv/bin/python -m tools.verify_reproducibility \
  --expected datasets/results/zai-org--GLM-5.2/b4734de4facf877f85769a911abafc5283eab3d9 \
  --actual /path/to/independently-generated-results
.venv/bin/python -m pytest -q
```

已提交的基线共 12,300 条：12,264 条成功、36 条预期错误。校验器检查请求和
结果一一对应、ID 唯一、请求哈希、渲染文本哈希、manifest 计数，以及所有可得
完整 token IDs 的长度和哈希。

差分工具使用依据固定 vLLM 上游源码独立实现的纯文本 reference path，不调用
交付版的 message normalizer 或 renderer；它比较状态、渲染字符串和完整 token
IDs。没有直接 import 整个 vLLM 包，因为上游模块导入本身会引入 PyTorch、引擎和
多模态运行时，这些均不属于 oracle 的运行依赖。

## 许可与来源

vLLM 提取代码保留 Apache-2.0 标头；其完整许可证及固定上游文件哈希位于
`vendor/vllm/`。UltraChat revision、来源分片、许可证和内容哈希记录在
`datasets/manifests/`。模型资产的 revision、文件尺寸和 SHA-256 记录在对应
`model_assets` manifest 中。
