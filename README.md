# PertReasonQA Benchmark

用于评测大语言模型对细胞扰动后基因表达变化的预测能力。给定扰动类型、细胞类型、扰动对象、目标基因和基础表达信息，模型输出 `up`、`down` 或 `unchanged`，并给出解释与置信度。

**克隆仓库、安装依赖、填入 API 信息，即可开始测评。** 固定 10K 测试数据、Prompt、评分程序和断点续跑功能均已包含在仓库中。

- **统一评测**：10,000 条固定样本，覆盖 Chemical / Genetic 共 8 个 split。
- **API 接入**：通过 OpenAI 兼容的 Chat Completions 接口调用模型，本机无需 GPU。
- **断点续跑**：定期保存已完成结果，恢复时仅处理剩余样本。
- **结果可追溯**：保存逐条预测、统计指标和运行配置，支持不同模型独立运行与比较。

## 快速开始

需要 **Python 3.9+**、Git，以及可用的模型 API。以下命令在仓库根目录执行。

### 1. 下载并安装

```bash
git clone https://github.com/zzz-sx/pertreasonqa-benchmark-extension.git
cd pertreasonqa-benchmark-extension
python -m venv .venv
```

激活虚拟环境：

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install -r requirements-benchmark.txt
```

Linux / macOS 若没有 `python` 命令，可用 `python3` 创建虚拟环境，激活后再使用 `python`。

### 2. 先试跑 10 条样本

将 `MODEL_NAME` 和 API 地址替换为实际服务配置：

```bash
python run_benchmark.py "MODEL_NAME" "https://YOUR_API_HOST/v1/chat/completions" "model_a_smoke" 2 new --sample-ratio 0.001
```

运行时会提示输入 API Key，输入内容不回显。也可以预先设置 `OPENAI_API_KEY` 环境变量，程序会优先读取；密钥不会写入结果或运行配置文件。

这里的 API 地址是**完整请求地址**，需要包含接口路径，不能只填写域名或 `/v1`。接口应支持 `model`、`messages`、`max_tokens`、`temperature`、`top_p` 和 `seed` 参数，并返回 `choices[0].message.content`。模型需按 Prompt 返回包含 `prediction`、`reasoning`、`confidence` 的 JSON。

试跑从每个 split 抽样，固定 `seed=42`，在当前数据集上共 10 条。完成后检查 `results/model_a_smoke_stats.json` 和逐条预测，确认接口可用；这组结果仅用于验证流程。

### 3. 运行完整 10K 测评

```bash
python run_benchmark.py "MODEL_NAME" "https://YOUR_API_HOST/v1/chat/completions" "model_a_10k" 8 new
```

程序会先校验 8 个数据文件和评测核心，再执行测评。API 调用按所使用服务的规则计费。

## 参数与运行配置

```text
python run_benchmark.py MODEL_NAME API_URL RUN_ID [WORKERS] [new|resume] [--sample-ratio RATIO]
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `MODEL_NAME` | 服务商提供的模型 ID | 必填 |
| `API_URL` | 完整 Chat Completions 请求地址 | 必填 |
| `RUN_ID` | 实验名称，决定输出文件名；以英文字母或数字开头，可包含 `-`、`_`、`.` | 必填 |
| `WORKERS` | 并发请求线程数，至少为 1 | `8` |
| `new` / `resume` | 新建测评或恢复检查点 | `new` |
| `--sample-ratio` | 各 split 的抽样比例，范围 `(0, 1]` | `1.0`，全部 10K |

统一入口使用 0-shot Improved CoT，固定 `seed=42`、`temperature=0.1`、`top_p=1.0`、`max_tokens=1024`，每完成 10 条新结果写入一次检查点。不同模型和试跑/正式测评请使用不同的 `RUN_ID`。

Linux / macOS 也可以使用 Shell 入口，参数与 Python 入口一致：

```bash
bash run_benchmark.sh "MODEL_NAME" "https://YOUR_API_HOST/v1/chat/completions" "model_a_10k" 8 new
```

## 断点续跑

中断后，用相同的模型、API 地址和 `RUN_ID`，将最后的模式改为 `resume`：

```bash
python run_benchmark.py "MODEL_NAME" "https://YOUR_API_HOST/v1/chat/completions" "model_a_10k" 8 resume
```

- 保留对应的 `.jsonl.partial`、`.jsonl.run.json` 和 `.jsonl.runtime_seconds` 文件；后者用于累计运行耗时。
- 若首次运行指定了 `--sample-ratio`，续跑时也需要传入相同值。
- 可以调整并发数，程序会检查其他评测配置是否匹配。
- 已写入检查点的记录不会重复请求；中断前尚未落盘的请求可能重新执行。
- 已生成完整结果的实验不能覆盖运行，重新测评请更换 `RUN_ID`。

## 查看结果

以 `RUN_ID=model_a_10k` 为例：

| 文件 | 内容 |
| --- | --- |
| `results/model_a_10k.jsonl` | 每条样本的真实标签、预测标签、解释和置信度 |
| `results/model_a_10k_stats.json` | 总体准确率、混淆矩阵、标签分布、各 split 准确率及运行耗时 |
| `results/model_a_10k.jsonl.run.json` | 模型、API 地址、数据与评测核心校验值、采样和生成参数 |

逐条结果包含 `sample_id`、`input_data`、`true_label`、`pred_label`、`reasoning` 和 `confidence`。`reasoning` 是模型返回的解释文本，`confidence` 是模型自报值。准确率按预测标签与真实标签是否一致计算。

运行期间会生成 `.jsonl.partial` 和 `.jsonl.runtime_seconds` 检查点文件，成功写入最终结果与统计后自动清理。`results/` 默认不纳入 Git。

仓库还提供两组已有的 10K 结果，可直接查看，无需调用 API：

- [GPT-5.6-Terra](model_outputs/gpt-5.6-terra/)：逐条预测与统计。
- [GPT-6-Astra](model_outputs/gpt-6-astra/)：逐条预测与统计。

## 数据与目录

`data/sampled_data/` 共包含 8 个 JSONL 文件，总计 10,000 条样本；Chemical 和 Genetic 各覆盖 cell ID/OOD 与 perturbation ID/OOD 的四种组合。数据随仓库提供，无需额外下载。

| 路径 | 用途 |
| --- | --- |
| `run_benchmark.py` | 跨平台启动入口 |
| `run_benchmark.sh` | Shell 启动入口 |
| `verify_run.py` | 数据完整性与运行配置校验 |
| `benchmark_test/scripts/` | 内置 Prompt、API 调用、评分和断点续跑程序 |
| `data/sampled_data/` | 固定 10K 测试集 |
| `data/SHA256SUMS` | 测试数据校验清单 |
| `model_outputs/` | 已有模型的完整测评输出 |
| `results/` | 本机运行产生的结果 |
| `requirements-benchmark.txt` | Python 依赖 |

单独验证数据完整性，不发起 API 请求：

```bash
python verify_run.py
```

## 常见问题

**请求一直重试？** 评测核心会持续重试请求错误或无法解析的回答。先用 10 条试跑检查完整接口地址、模型 ID、密钥和参数支持情况；模型必须返回有效预测和非空 `reasoning`。接口不支持上述生成参数时，需要先适配调用代码。

**遇到限流？** 降低 `WORKERS`，例如从 `8` 改为 `2`。续跑允许调整并发数。

**提示配置不匹配？** 使用首次运行的模型、API 地址和抽样比例恢复；若要换模型或配置，请使用新的 `RUN_ID`。

**提示校验失败？** 从仓库恢复对应的数据文件或评测核心，避免直接修改固定测试集、校验清单或核心文件后继续原实验。
