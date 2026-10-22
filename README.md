# PertReasonQA Benchmark Extension

基于 PertReasonQA 的大模型测评扩展项目，支持固定 10K 数据集、OpenAI 兼容 API、多模型评测、结果保存及断点续跑。沿用上游 Improved CoT Prompt 和评分逻辑，不修改原始仓库。

## 1. 上游仓库

**ModelScope 原始项目：**
https://www.modelscope.cn/datasets/hankyang428/pertreason-s

**上游固定版本：**
`b359bac8325b4ddfb437126b1400feef6bd88ea8`

**本扩展仓库：**
https://github.com/zzz-sx/pertreasonqa-benchmark-extension

两个仓库需克隆到同一个父目录下。

## 2. 文件结构与功能

    PertReasonQA/
    ├── pertreason-s/                  # ModelScope 原始项目
    │   └── benchmark_test/scripts/
    │       └── test_improved_cot.py  # 原版 Prompt、API 调用及评分
    │
    └── benchmark-extension/          # 本扩展项目
        ├── benchmark_test/scripts/
        │   └── run_improved_cot_resume.py
        │                               # 断点续跑测评程序
        ├── data/
        │   ├── sampled_data/          # 固定 10K 样本，8 个 JSONL
        │   └── SHA256SUMS             # 数据完整性校验
        ├── model_outputs/             # 已有模型输出（如已提交）
        │   └── gpt-5.6-terra/
        │       ├── predictions_10k.jsonl
        │       └── predictions_10k_stats.json
        ├── results/                   # 新模型测评结果（运行时生成）
        ├── run_benchmark.sh           # 统一启动入口
        ├── verify_run.py              # 版本及运行配置校验
        ├── requirements-benchmark.txt # Python 依赖
        ├── UPSTREAM_COMMIT            # 上游 Git 版本
        └── README.md

### 数据说明

- `data/sampled_data/`：固定 10,000 条测试样本，覆盖 Chemical / Genetic 共 8 个 split。
- `model_outputs/`：已有模型的预测结果，包含 reasoning、预测标签及置信度等字段；需要单独提交后才会随仓库下载。
- `results/`：后续其他模型产生的测评输出，不纳入 Git。

## 3. 新服务器部署

### 3.1 克隆原始仓库

    mkdir -p ~/PertReasonQA
    cd ~/PertReasonQA

    git lfs install
    GIT_LFS_SKIP_SMUDGE=1 git clone \
      https://www.modelscope.cn/datasets/hankyang428/pertreason-s.git

    git -C pertreason-s checkout \
      b359bac8325b4ddfb437126b1400feef6bd88ea8

### 3.2 克隆扩展仓库

    git clone \
      https://github.com/zzz-sx/pertreasonqa-benchmark-extension.git \
      benchmark-extension

私有 GitHub 仓库需要相应的访问权限。

### 3.3 配置 Python 环境

    cd ~/PertReasonQA
    python3 -m venv .venv-benchmark
    source .venv-benchmark/bin/activate

    pip install -r benchmark-extension/requirements-benchmark.txt

### 3.4 校验数据

    cd ~/PertReasonQA/benchmark-extension/data/sampled_data
    sha256sum -c ../SHA256SUMS

应有 8 个文件全部显示 OK，总计 10,000 条样本。

## 4. 运行测评

先激活环境：

    source ~/PertReasonQA/.venv-benchmark/bin/activate
    cd ~/PertReasonQA/benchmark-extension

### 4.1 运行新模型

    ./run_benchmark.sh \
      "MODEL_NAME" \
      "API_URL" \
      "model_a_10k_0shot" \
      8 \
      new

参数说明：

| 参数 | 含义 |
|---|---|
| MODEL_NAME | API 平台提供的模型 ID |
| API_URL | API 完整请求地址，以原版 call_api 实现为准 |
| RUN_ID | 本次实验名称，决定结果文件名 |
| 8 | 并发线程数，可修改 |
| new | 启动新测评 |

运行时会提示隐藏输入 API Key，无需将密钥写入文件。

默认使用固定 10K 全部样本、0-shot Improved CoT、seed=42、temperature=0.1、max_tokens=1024。

### 4.2 断点续跑

中断后使用相同模型、API URL 和 RUN_ID：

    ./run_benchmark.sh \
      "MODEL_NAME" \
      "API_URL" \
      "model_a_10k_0shot" \
      8 \
      resume

已完成的 checkpoint 记录将被恢复，继续处理剩余样本。

不同模型必须使用不同 RUN_ID。

## 5. 结果文件

测评结果保存在 `benchmark-extension/results/`。

    results/
    ├── model_a_10k_0shot.jsonl
    ├── model_a_10k_0shot_stats.json
    └── model_a_10k_0shot.jsonl.run.json

- `.jsonl`：逐条预测结果，包括 reasoning。
- `_stats.json`：准确率、混淆矩阵和各 split 统计。
- `.run.json`：模型名称、API 地址及运行配置。
- `.partial`：运行期间的断点检查点，完成后自动清理。

## 6. 使用说明

- 本扩展不修改 ModelScope 上游代码。
- 正式运行前建议先使用少量样本验证 API 兼容性。
- API URL、模型 ID 应根据实际服务商配置。
- 真实 API 调用可能产生费用。
- 原始测试数据与模型输出的公开再分发应遵守相关使用条款。
