#!/usr/bin/env python3
"""
改进版 PertReasonQA 测试脚本
集成策略:
1. Chain-of-Thought (CoT) 推理
2. 类别平衡的 Few-shot 示例选择
3. 改进的决策标准提示
"""

import json
import argparse
import requests
import time
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImprovedCOTEvaluator:
    def __init__(self, api_url, api_key, model_name, max_tokens=1024,
                 temperature=0.1, top_p=1.0, seed=None, n_shots=0):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.n_shots = n_shots
        self.results = []
        self.example_pool = []

    def load_test_data(self, data_dir, sample_ratio=1.0, stratified=True):
        """加载测试数据（分层采样）"""
        data_dir = Path(data_dir)
        all_data = []

        file_patterns = [
            "prqa_chemical-split_cell_id-split_pert_id_test-task_3way_o.jsonl",
            "prqa_chemical-split_cell_id-split_pert_ood_test-task_3way_o.jsonl",
            "prqa_chemical-split_cell_ood_test-split_pert_id_test-task_3way_o.jsonl",
            "prqa_chemical-split_cell_ood_test-split_pert_ood_test-task_3way_o.jsonl",
            "prqa_geneticRNv3-split_cell_id-split_pert_id_test-task_3way_o.jsonl",
            "prqa_geneticRNv3-split_cell_id-split_pert_ood_test-task_3way_o.jsonl",
            "prqa_geneticRNv3-split_cell_ood_test-split_pert_id_test-task_3way_o.jsonl",
            "prqa_geneticRNv3-split_cell_ood_test-split_pert_ood_test-task_3way_o.jsonl",
        ]

        print(f"📊 使用{'分层' if stratified else '随机'}采样策略")

        for pattern in file_patterns:
            file_path = data_dir / pattern
            if not file_path.exists():
                print(f"⚠️  文件不存在: {pattern}")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                subset_data = [json.loads(line) for line in f]

            original_count = len(subset_data)
            if stratified and sample_ratio < 1.0:
                n_samples = max(1, int(len(subset_data) * sample_ratio))
                subset_data = random.sample(subset_data, n_samples)
                print(f"  📂 {pattern:70s}: {original_count:5d} → {n_samples:4d}")

            all_data.extend(subset_data)

        if not stratified and sample_ratio < 1.0:
            n_samples = max(1, int(len(all_data) * sample_ratio))
            all_data = random.sample(all_data, n_samples)

        print(f"✅ 总共加载: {len(all_data)} 条测试数据")
        return all_data

    def load_example_pool(self, annotation_file):
        """加载标注数据作为示例池"""
        annotation_path = Path(annotation_file)

        if not annotation_path.exists():
            raise FileNotFoundError(f"标注文件不存在: {annotation_file}")

        all_examples = []
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))

        # 按标签分组
        examples_by_label = defaultdict(list)
        for ex in all_examples:
            label = ex['input']['label']
            examples_by_label[label].append(ex)

        self.example_pool = examples_by_label
        print(f"\n📚 示例池统计 (共 {len(all_examples)} 条):")
        for label in ['up', 'down', 'unchanged']:
            if label in examples_by_label:
                print(f"  {label:10s}: {len(examples_by_label[label])} 个示例")

    def select_balanced_examples(self, test_sample, n_shots):
        """
        选择类别平衡的示例
        优先选择与测试样本相同 cell_type 或 pert_type 的示例
        """
        if not self.example_pool or n_shots == 0:
            return []

        examples = []
        labels_needed = ['up', 'down', 'unchanged'][:n_shots]

        test_cell_type = test_sample['input'].get('cell_type')
        test_pert_type = test_sample['input'].get('pert_type')

        for label in labels_needed:
            if label not in self.example_pool:
                continue

            candidates = self.example_pool[label]

            # 优先级1: 相同 cell_type 和 pert_type
            preferred = [
                ex for ex in candidates
                if (ex['input'].get('cell_type') == test_cell_type and
                    ex['input'].get('pert_type') == test_pert_type)
            ]

            # 优先级2: 相同 cell_type 或 pert_type
            if not preferred:
                preferred = [
                    ex for ex in candidates
                    if (ex['input'].get('cell_type') == test_cell_type or
                        ex['input'].get('pert_type') == test_pert_type)
                ]

            # 优先级3: 任意相同标签的示例
            if not preferred:
                preferred = candidates

            if preferred:
                examples.append(random.choice(preferred))

        return examples

    def format_example_cot(self, data, include_answer=True):
        """使用 CoT 格式化示例"""
        input_data = data['input']

        # 简化 basal context 显示（只显示前10个基因）
        basal_context = data.get('basal_context', {})
        if basal_context:
            # 取前10个基因
            basal_genes = list(basal_context.items())[:10]
            basal_str = ", ".join([f"{gene}: {expr:.3f}" for gene, expr in basal_genes])
            if len(basal_context) > 10:
                basal_str += f" ... (total {len(basal_context)} genes)"
        else:
            basal_str = "No basal expression data"

        prompt = f"""
**Perturbation Type**: {input_data['pert_type']}
**Cell Type**: {input_data['cell_type']}
**Perturbation**: {input_data['pert']}
**Target Gene**: {input_data['gene']}
**Basal Expression**: {basal_str}
"""

        if include_answer:
            answer = input_data['label']
            reasoning_text = data.get('output', {}).get('reasoning_text',
                                                        'This is a demonstration example.')
            prompt += f"""
**Analysis & Prediction**:
{{
  "prediction": "{answer}",
  "reasoning": "{reasoning_text}"
}}
"""
        else:
            prompt += """
**Your task**: Analyze and predict the gene expression change.
"""

        return prompt

    def build_improved_cot_prompt(self, test_sample):
        """构建改进的 CoT Prompt"""

        # System instruction with CoT guidance
        system_prompt = """You are an expert molecular biologist specializing in cellular perturbation analysis and gene regulation.

When analyzing perturbation experiments, follow this reasoning process:

1. **Mechanism of Action**: Identify the primary molecular target and mechanism of the perturbation
2. **Signaling Pathways**: Consider relevant signaling cascades and regulatory networks
3. **Gene Regulation**: Reason about how the perturbation affects the target gene (direct or indirect)
4. **Expression Context**: Evaluate basal expression levels and cellular context

**Important Decision Criteria**:
- Predict "up" if there is clear evidence of activation/upregulation (fold-change ≥ 1.5x)
- Predict "down" if there is clear evidence of inhibition/downregulation (fold-change ≤ 0.67x)
- Predict "unchanged" only if:
  * No plausible regulatory mechanism exists, OR
  * The mechanism exists but basal expression context suggests minimal effect, OR
  * Opposing regulatory effects cancel out

**Key principle**: When a plausible mechanism exists, prefer "up" or "down" over "unchanged".
"""

        # Few-shot examples (if n_shots > 0)
        examples_section = ""
        if self.n_shots > 0 and self.example_pool:
            examples = self.select_balanced_examples(test_sample, self.n_shots)
            if examples:
                examples_section = "\n**Here are some example analyses:**\n"
                for i, ex in enumerate(examples, 1):
                    examples_section += f"\n### Example {i}:\n"
                    examples_section += self.format_example_cot(ex, include_answer=True)

        # Test query
        test_query = "\n**Now analyze this new experiment:**\n"
        test_query += self.format_example_cot(test_sample, include_answer=False)
        test_query += """
Provide your analysis in JSON format:
{
  "prediction": "up/down/unchanged",
  "reasoning": "Concise mechanistic explanation (50-150 words)",
  "confidence": 0.0-1.0
}
"""

        # Combine all parts
        full_prompt = system_prompt + examples_section + test_query

        return full_prompt

    def call_api(self, prompt, base_delay=1.0):
        """调用 API，遇到任何错误或空结果都无限重试（指数退避）。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        if self.model_name:
            payload["model"] = self.model_name
        if self.seed is not None:
            payload["seed"] = self.seed

        attempt = 0
        while True:
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=1800
                )
                response.raise_for_status()
                result = response.json()

                if isinstance(result, dict):
                    if 'prediction' in result:
                        parsed = result
                    elif 'choices' in result:
                        content = result['choices'][0]['message'].get('content', '')
                        parsed = self._extract_json(content)
                    elif 'response' in result:
                        parsed = self._extract_json(result['response'])
                    else:
                        parsed = self._extract_json(str(result))
                else:
                    parsed = self._extract_json(str(result))

                # 有效结果：prediction 不是 error 且 reasoning 非空
                if parsed.get('prediction', 'error') != 'error' and parsed.get('reasoning', '').strip():
                    return parsed

                reason = f"invalid result: prediction={parsed.get('prediction')!r}, reasoning empty={not parsed.get('reasoning','').strip()}"

            except Exception as e:
                reason = str(e)

            attempt += 1
            delay = 1  # 2 4 8 16 32s
            print(f"\n  [retry {attempt}] {reason} — waiting {delay:.0f}s")
            time.sleep(delay)

    def _extract_json(self, text):
        """从文本中提取 JSON"""
        if not isinstance(text, str):
            text = str(text)

        try:
            return json.loads(text)
        except:
            import re
            # Try to extract JSON from code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass

            # Try to extract any {...} block
            json_match = re.search(r'\{[^{}]*"prediction"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass

            return {"prediction": "error", "reasoning": text[:200], "confidence": 0.0}

    def evaluate_single(self, sample):
        """评估单个样本"""
        prompt = self.build_improved_cot_prompt(sample)
        response = self.call_api(prompt)
        print(response)

        true_label = sample['input']['label']
        pred_label = response.get('prediction', 'error')

        return {
            'sample_id': sample.get('id', 'unknown'),
            'input_data': sample['input'],
            'true_label': true_label,
            'pred_label': pred_label,
            'reasoning': response.get('reasoning', ''),
            'confidence': response.get('confidence', 0.0)
        }

    def evaluate(self, test_data, max_workers=1):
        """评估所有测试数据"""
        results = []

        print(f"\n🚀 开始评估 (并行度: {max_workers}, n_shots: {self.n_shots})...")
        start_time = time.time()

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_sample = {
                    executor.submit(self.evaluate_single, sample): sample
                    for sample in test_data
                }

                with tqdm(total=len(test_data), desc="测试进度") as pbar:
                    for future in as_completed(future_to_sample):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
        else:
            for sample in tqdm(test_data, desc="测试进度"):
                result = self.evaluate_single(sample)
                results.append(result)

        elapsed_time = time.time() - start_time
        self.results = results

        # 计算统计指标
        return self.compute_statistics(results, elapsed_time, len(test_data))

    def compute_statistics(self, results, elapsed_time, total_samples):
        """计算评估统计"""
        correct = sum(1 for r in results if r['pred_label'] == r['true_label'])
        accuracy = correct / len(results) if results else 0

        # 混淆矩阵
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        label_distribution = Counter()

        for r in results:
            true_label = r['true_label']
            pred_label = r['pred_label']
            label_distribution[true_label] += 1
            confusion_matrix[true_label][pred_label] += 1

        # 按测试集统计
        split_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            inp = r['input_data']
            split_key = f"{inp['pert_type']}-{inp['split_cell']}-{inp['split_pert']}"
            split_stats[split_key]['total'] += 1
            if r['pred_label'] == r['true_label']:
                split_stats[split_key]['correct'] += 1

        # 格式化输出
        stats = {
            'accuracy': accuracy,
            'correct': correct,
            'total_samples': len(results),
            'label_distribution': dict(label_distribution),
            'confusion_matrix': {k: dict(v) for k, v in confusion_matrix.items()},
            'split_statistics': {
                k: {
                    'accuracy': v['correct'] / v['total'],
                    'correct': v['correct'],
                    'total': v['total']
                }
                for k, v in split_stats.items()
            },
            'n_shots': self.n_shots,
            'elapsed_time': elapsed_time,
            'samples_per_second': len(results) / elapsed_time,
            'max_workers': 1,
            'model_name': self.model_name,
            'temperature': self.temperature
        }

        return stats


def main():
    parser = argparse.ArgumentParser(description='改进版 CoT 测试脚本')

    # API 配置
    parser.add_argument('--api_url', required=True, help='API 端点')
    parser.add_argument('--api_key', default=None, help='API 密钥')
    parser.add_argument('--model_name', required=True, help='模型名称')

    # 测试配置
    parser.add_argument('--data_dir', default='data/data_jsonl', help='数据目录')
    parser.add_argument('--sample_ratio', type=float, default=0.01, help='采样比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # Few-shot 配置
    parser.add_argument('--n_shots', type=int, default=0, help='Few-shot 示例数量')
    parser.add_argument('--annotation_file', default=None,
                       help='标注文件路径（用于 few-shot）')

    # 模型参数
    parser.add_argument('--max_tokens', type=int, default=1024, help='最大生成 tokens')
    parser.add_argument('--temperature', type=float, default=0.1, help='温度参数')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p 采样')

    # 并行配置
    parser.add_argument('--max_workers', type=int, default=1, help='并行线程数')

    # 输出配置
    parser.add_argument('--output', required=True, help='结果输出路径')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 初始化评估器
    evaluator = ImprovedCOTEvaluator(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        n_shots=args.n_shots
    )

    # 加载测试数据
    test_data = evaluator.load_test_data(
        args.data_dir,
        sample_ratio=args.sample_ratio,
        stratified=True
    )

    # 如果使用 few-shot，加载示例池
    if args.n_shots > 0:
        if not args.annotation_file:
            default_annotation = "/cpfs01/sihan.yang/PertReasonQA/data/annotations/train_reasoning_49_seed_42.jsonl"
            print(f"⚠️  未指定 --annotation_file，使用默认: {default_annotation}")
            args.annotation_file = default_annotation

        evaluator.load_example_pool(args.annotation_file)

    # 运行评估
    stats = evaluator.evaluate(test_data, max_workers=args.max_workers)

    # 保存详细结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in evaluator.results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 保存统计信息
    stats_path = output_path.with_suffix('.json').with_name(
        output_path.stem + '_stats.json'
    )
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印结果
    print("\n" + "="*80)
    print("📊 评估结果")
    print("="*80)
    print(f"总体准确率: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total_samples']})")
    print(f"\n标签分布:")
    for label, count in stats['label_distribution'].items():
        print(f"  {label:10s}: {count}")

    print(f"\n混淆矩阵:")
    for true_label in ['up', 'down', 'unchanged']:
        if true_label in stats['confusion_matrix']:
            pred_counts = stats['confusion_matrix'][true_label]
            print(f"  真实={true_label:10s}: ", end='')
            for pred_label in ['up', 'down', 'unchanged']:
                count = pred_counts.get(pred_label, 0)
                print(f"{pred_label}={count:3d}  ", end='')
            print()

    print(f"\n各测试集准确率:")
    for split_name, split_stat in sorted(stats['split_statistics'].items()):
        print(f"  {split_name:60s}: {split_stat['accuracy']:.4f} ({split_stat['correct']}/{split_stat['total']})")

    print(f"\n⏱️  总耗时: {stats['elapsed_time']:.2f}s")
    print(f"📈 吞吐量: {stats['samples_per_second']:.2f} samples/s")
    print(f"\n结果已保存到: {output_path}")
    print(f"统计已保存到: {stats_path}")


if __name__ == '__main__':
    main()
