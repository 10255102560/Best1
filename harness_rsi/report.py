"""Readable Chinese experiment reports and dependency-free static SVG charts."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any


_LABELS = {
    "initial_reason": "初始策略：始终分步推理",
    "fixed_direct": "固定：直接回答",
    "fixed_reason": "固定：分步推理",
    "fixed_review": "固定：检查修订",
    "best_fixed": "验证集选出的最佳固定策略",
    "learned": "梯度学习调度器",
    "uniform": "均匀随机策略（期望指标）",
    "oracle": "逐题最优上界（使用测试标签）",
}
_ENGLISH_LABELS = {
    "fixed_direct": "Fixed direct",
    "fixed_reason": "Fixed reason",
    "fixed_review": "Fixed review",
    "best_fixed": "Best fixed (validation)",
    "learned": "Learned router",
    "uniform": "Uniform (expected)",
    "oracle": "Oracle (test labels)",
}


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, dict):
        value = value.get("utility")
        if value is None:
            return "未记录"
    return f"{float(value):.{digits}f}"


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _comparison_line(label: str, result: dict[str, Any], metric: str, scale: float = 1.0) -> str:
    item = result.get(metric)
    if not item:
        return f"- {label}：未记录。"
    low, high = item["ci95"]
    return (
        f"- {label}：{item['point'] * scale:+.4f}，"
        f"95% 配对 bootstrap 区间 [{low * scale:+.4f}, {high * scale:+.4f}]。"
    )


def _svg(metrics: dict[str, dict[str, Any]], run_name: str) -> str:
    keys = [key for key in _ENGLISH_LABELS if key in metrics]
    height = 150 + 49 * len(keys)
    max_tokens = max((float(metrics[key]["mean_tokens"]) for key in keys), default=1)
    max_tokens = max(max_tokens, 1.0)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img">',
        '<title>Held-out accuracy and token cost by harness strategy</title>',
        '<desc>Two bar charts compare answer accuracy and average token usage. Oracle uses test labels and is not deployable.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172b4d;font-size:13px}.title{font-size:20px;font-weight:bold}.muted{fill:#53657b;font-size:12px}</style>',
        f'<text class="title" x="24" y="32">{escape(run_name)}</text>',
        '<text x="220" y="65">Accuracy (higher is better)</text>',
        '<text x="720" y="65">Mean tokens (lower is cheaper)</text>',
    ]
    for index, key in enumerate(keys):
        row = metrics[key]
        y = 88 + index * 49
        accuracy = float(row["accuracy"])
        tokens = float(row["mean_tokens"])
        color = "#138468" if key == "learned" else "#426da9"
        if key == "oracle":
            color = "#8a94a2"
        chunks.extend(
            [
                f'<text x="24" y="{y + 17}">{escape(_ENGLISH_LABELS[key])}</text>',
                f'<rect x="220" y="{y}" width="340" height="24" rx="3" fill="#edf1f7"/>',
                f'<rect x="220" y="{y}" width="{340 * max(0, min(1, accuracy)):.2f}" height="24" rx="3" fill="{color}"/>',
                f'<text x="570" y="{y + 17}">{100 * accuracy:.1f}%</text>',
                f'<rect x="720" y="{y}" width="290" height="24" rx="3" fill="#edf1f7"/>',
                f'<rect x="720" y="{y}" width="{290 * tokens / max_tokens:.2f}" height="24" rx="3" fill="{color}"/>',
                f'<text x="1020" y="{y + 17}">{tokens:.1f}</text>',
            ]
        )
    chunks.extend(
        [
            f'<text class="muted" x="24" y="{height - 35}">Green: deployed router. Gray: oracle uses test labels and is not a deployed strategy.</text>',
            f'<text class="muted" x="24" y="{height - 16}">Accuracy and token cost are separate metrics; tokens are not a currency price.</text>',
            '</svg>',
        ]
    )
    return "\n".join(chunks) + "\n"


def write_report(summary: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write REPORT.md and summary.svg from the runner's recorded results."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config = summary.get("config", {})
    metrics = summary.get("metrics", {})
    comparisons = summary.get("comparisons", {})
    ledger = summary.get("ledger", {})
    learned = metrics.get("learned", {})
    initial = metrics.get("initial_reason", {})
    run_name = str(summary.get("run_name", "Harness RSI benchmark"))
    cost_weight = float(config.get("cost_weight", 0.05))
    token_unit = float(config.get("token_unit", 1000))
    lines = [
        f"# {_markdown(run_name)}：Benchmark 测评报告",
        "",
        "本实验固定基础模型，通过 CPU 上的梯度更新优化题目到执行策略的调度。"
        "循环包括收集训练反馈、更新候选调度器、验证集选择和保留或回退。"
        "这是参数化 harness 的受限自我改进，不代表任意代码递归修改或通用能力持续增长。",
        "",
        "## 实验设置",
        "",
        f"- 数据集：`{_markdown(config.get('dataset', '未记录'))}`；模型：`{_markdown(config.get('model', '未记录'))}`。",
        f"- 训练 / 验证 / 测试题数：{config.get('train_size', '未记录')} / {config.get('val_size', '未记录')} / {config.get('test_size', '未记录')}。",
        f"- 随机种子：{config.get('seed', '未记录')}；请求上限：{config.get('max_requests', '未记录')}。",
        f"- 效用定义：`accuracy - {cost_weight:g} × mean_tokens / {token_unit:g}`。"
        "该权重表达预先设定的质量与成本偏好，改变权重会改变结论。",
        f"- 最佳固定策略由验证集选为 `{_markdown(summary.get('best_fixed_strategy', '未记录'))}`；"
        f"最终保留的调度器版本为 `{_markdown(summary.get('selected_version', '未记录'))}`。",
        "- 调度器使用完整动作反馈的精确期望奖励梯度；上线与验证执行确定性 argmax。均匀随机基线报告期望指标。",
        "",
        "## 独立测试集结果",
        "",
        "除 oracle 外，策略选择均应在查看测试结果前冻结。Oracle 使用测试标签逐题选择最优动作，"
        "只表示已采集动作的效用上界，不能部署，也不能作为已取得的收益。",
        "",
        "| 策略 | 正确率 | 平均 token | 平均调用次数 | 平均延迟（秒） | 效用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in _LABELS.items():
        if key not in metrics:
            continue
        item = metrics[key]
        lines.append(
            f"| {label} | {100 * float(item['accuracy']):.2f}% | {_number(item.get('mean_tokens'), 1)} | "
            f"{_number(item.get('mean_calls'), 2)} | {_number(item.get('mean_latency_s'), 3)} | {_number(item.get('utility'), 4)} |"
        )
    lines += ["", "![正确率与 token 成本比较](summary.svg)", "", "## 收益证据", ""]
    best_comparison = comparisons.get("vs_best_fixed", {})
    best_utility = best_comparison.get("utility_difference")
    if best_utility:
        lower, upper = best_utility["ci95"]
        if lower > 0:
            verdict = "相对验证集选出的最佳固定策略，效用差的 95% 配对 bootstrap 区间下界大于 0，支持在本测试分布和成本权重下取得效用提升。"
        elif upper < 0:
            verdict = "相对验证集选出的最佳固定策略，效用差的 95% 配对 bootstrap 区间上界小于 0；当前调度器表现更差，不能宣称自我改进成功。"
        else:
            verdict = "相对验证集选出的最佳固定策略，效用差的 95% 配对 bootstrap 区间包含 0，尚不足以证明调度器取得稳定收益。"
    else:
        verdict = "未记录与最佳固定策略的配对比较，暂不能判断梯度调度是否超过简单固定选择。"
    lines += [verdict, "", "差值均为“学习调度器 − 对照”。准确率差以百分点表示；token 差为负表示节省。", ""]
    for comparison_key, label in (("vs_initial", "相对初始分步推理"), ("vs_best_fixed", "相对最佳固定策略")):
        result = comparisons.get(comparison_key, {})
        lines += [
            _comparison_line(label + "的准确率差（百分点）", result, "accuracy_difference", 100),
            _comparison_line(label + "的平均 token 差", result, "token_difference"),
            _comparison_line(label + "的效用差", result, "utility_difference"),
        ]
    lines += [
        "",
        "区间按题配对重采样，反映该批测试题的采样不确定性；不包含模型采样方差、数据污染风险或重复挑选测试结果的偏差。"
        "小样本区间包含 0 时，应报告证据不足，不能把点估计提升写成确定结论。效用提升也不等于准确率提升。",
        "",
        "## 自我改进循环",
        "",
        "| 轮次 | 累计训练题 | 接受候选 | incumbent 验证效用 | candidate 验证效用 | 原因 |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for item in summary.get("rounds", []):
        lines.append(
            f"| {_markdown(item.get('round', ''))} | {_markdown(item.get('train_tasks', ''))} | "
            f"{'是' if item.get('accepted') else '否'} | {_number(item.get('val_incumbent'), 4)} | "
            f"{_number(item.get('val_candidate'), 4)} | {_markdown(item.get('reason', ''))} |"
        )
    selected_actions = summary.get("selected_actions", {})
    review_effect = summary.get("review_effect", {})
    lines += [
        "",
        f"最终测试路由分布：direct={selected_actions.get('direct', '未记录')}，"
        f"reason={selected_actions.get('reason', '未记录')}，review={selected_actions.get('review', '未记录')}。",
        "",
        f"检查修订相对其初稿：纠正 {review_effect.get('fixes', '未记录')} 题，"
        f"将原本正确答案改错 {review_effect.get('breaks', '未记录')} 题。"
        "检查修订并不保证改善，因此应同时记录纠错与改错。",
        "",
        "验证集上的候选比较是选择依据，不能作为最终泛化收益。若所有候选均未被接受，保留初始版本也是有效实验结果。",
        "",
        "## 收集成本与部署摊销",
        "",
        f"请求尝试 {ledger.get('attempts', '未记录')} 次，成功 {ledger.get('successes', '未记录')} 次，失败 {ledger.get('failures', '未记录')} 次；"
        f"已记录输入 token={ledger.get('input_tokens', '未记录')}，输出 token={ledger.get('output_tokens', '未记录')}，"
        f"合计 token={ledger.get('total_tokens', '未记录')}。",
        "",
        "token 不等同于人民币或美元。不同模型输入、输出、缓存 token 计价可能不同；未给定单价时不报告货币收益。"
        "失败请求可能仍产生供应商费用，缺失 usage 的费用也无法从该日志完整还原。",
        "",
    ]
    if learned and initial and ledger.get("total_tokens") is not None:
        saved = float(initial["mean_tokens"]) - float(learned["mean_tokens"])
        accuracy_delta = float(learned["accuracy"]) - float(initial["accuracy"])
        accuracy_ci = comparisons.get("vs_initial", {}).get("accuracy_difference", {}).get("ci95")
        if saved <= 0:
            lines.append("相对初始策略没有平均 token 节省，因此不存在基于 token 节省的收集成本摊销点。")
        elif accuracy_delta < 0:
            lines.append("当前准确率点估计低于初始策略，质量保持条件不满足；不将 token 减少解释为等质量节约，也不报告等质量摊销门槛。")
        else:
            threshold = math.ceil(float(ledger["total_tokens"]) / saved)
            if accuracy_ci is not None and accuracy_ci[0] >= 0:
                qualifier = "本测试的准确率差区间未显示下降。在未来质量保持、成本与题目分布稳定的条件下"
            else:
                qualifier = "准确率点估计未下降，但区间尚不能排除下降。仅假设未来质量保持、成本与题目分布稳定时"
            lines.append(
                f"{qualifier}，每题平均节省 {saved:.2f} token，"
                f"条件摊销门槛为约 **{threshold:,} 道后续题目**："
                f"`ceil({ledger['total_tokens']} / {saved:.6f})`。"
                "这里把整个收集过程（含评测）的已记录 token 都计入前期成本，属于保守的探索成本核算；"
                "该数字是有条件估算，不是已经获得的净收益。"
            )
    else:
        lines.append("缺少初始策略、最终策略或收集 token 数据，无法估算部署摊销门槛。")
    lines += [
        "",
        "## 解释边界",
        "",
        "- 本实验只更新小型调度器，不训练基础语言模型，也不需要 GPU。",
        "- 完整动作反馈会产生额外探索开销；缓存和重复梯度更新不等于免费的新模型反馈。",
        "- 测试标签只用于最终评测及标注为 oracle 的上界，不得用于选择轮次、特征或成本权重。",
        "- 同一验证集多轮选择可能过拟合；报告的是最终冻结版本在独立测试集上的表现。",
        "- 延迟受网络、并发及缓存影响，单次观测不能证明稳定的生产延迟收益。",
        "- 若数据集标识为 synthetic/demo/fixture，相应结果只证明工程链路可运行，不能替代真实模型 benchmark。",
        "",
    ]
    report_path = directory / "REPORT.md"
    chart_path = directory / "summary.svg"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    chart_path.write_text(_svg(metrics, run_name), encoding="utf-8")
    return {"report": str(report_path), "chart": str(chart_path)}
