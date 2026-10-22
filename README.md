# PertReasonQA Benchmark Extension

Portable 10K benchmark extension for PertReasonQA.

## Features

- Fixed 10,000 samples across 8 splits
- Original Improved CoT prompt and scoring
- OpenAI-compatible model APIs
- Concurrent evaluation
- Checkpoint and resume support
- SHA256 dataset verification
- Run configuration consistency checks

## Upstream Repository

ModelScope:
https://www.modelscope.cn/datasets/hankyang428/pertreason-s

Pinned commit:
b359bac8325b4ddfb437126b1400feef6bd88ea8

The upstream repository is not modified.

## Installation

On a new Linux server, run:

    mkdir -p ~/PertReasonQA
    cd ~/PertReasonQA
    git lfs install
    git clone https://www.modelscope.cn/datasets/hankyang428/pertreason-s.git
    git -C pertreason-s checkout b359bac8325b4ddfb437126b1400feef6bd88ea8

Clone this GitHub extension repository into the
same parent directory, using the name benchmark-extension.

Then install dependencies:

    cd ~/PertReasonQA
    python3 -m venv benchmark-extension/.venv
    source benchmark-extension/.venv/bin/activate
    pip install -r benchmark-extension/requirements-benchmark.txt

## Dataset Verification

    cd ~/PertReasonQA/benchmark-extension/data/sampled_data
    sha256sum -c ../SHA256SUMS

Expected: 8 files OK, 10,000 samples.

## Run Benchmark

    cd ~/PertReasonQA/benchmark-extension
    ./run_benchmark.sh MODEL_NAME API_URL model_10k_0shot 8 new

Enter the API key at the hidden prompt.

## Resume Benchmark

Use the same model, API URL, and RUN_ID:

    ./run_benchmark.sh MODEL_NAME API_URL model_10k_0shot 8 resume

## Important Notes

- Use a unique RUN_ID for each model and experiment.
- Model prediction results are stored under results/.
- Historical results are not included.
- The original Improved CoT prompt is preserved.
- Checkpoint recovery has passed an offline 8-sample test.
- Real API and clean-server deployment require final validation.
