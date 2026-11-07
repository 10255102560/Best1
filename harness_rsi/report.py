"""Readable Chinese experiment reports and dependency-free static SVG charts.

``write_report`` turns a run's ``summary.json`` into ``REPORT.md`` and
``summary.svg``. Both files are written with LF line endings, so an offline
replay reproduces them byte for byte on Windows, macOS and Linux. Missing
fields degrade to "未记录" instead of raising, because the same code also
reports partial or synthetic runs.

This module only formats recorded results. It is deliberately excluded from
the implementation hash in ``experiment.py``: changing the presentation never
changes an experiment.
"""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any, Callable

MISSING = "未记录"

_LABELS = {
    "learned": "梯度学习路由器（GFPG）",
    "best_fixed": "验证集选出的最佳固定策略",
    "initial_reason": "初始策略（始终 reason）",
    "fixed_direct": "固定 direct（只给答案）",
    "fixed_reason": "固定 reason（简短分步）",
    "fixed_review": "固定 review（分步 + 复核）",
    "uniform": "均匀随机路由（期望指标）",
    "oracle": "逐题最优 oracle（使用测试标签，不可部署）",
}

_ROUND_REASONS = {
    "validation gate rejected candidate": "验证门控拒绝候选，保留现任版本",
    "validation utility improved and accuracy guard passed": "验证效用提升且准确率守卫通过，接纳候选",
}

# Palette shared with the course-report figures: blue = learned router,
# orange = best fixed strategy, gray = other baselines. Text stays in ink colours.
_BLUE, _ORANGE, _GRAY, _ORACLE = "#2a78d6", "#eb6834", "#a8a69e", "#e4e2db"
_INK, _INK2, _MUTED, _GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
_FONT = ("'PingFang SC','Hiragino Sans GB','Microsoft YaHei','Noto Sans CJK SC',"
         "'Source Han Sans SC','WenQuanYi Micro Hei',Arial,sans-serif")


# --------------------------------------------------------------------------- formatting helpers

def _number(value: Any, digits: int = 3) -> str:
    if isinstance(value, dict):
        value = value.get("utility")
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return MISSING


def _percent(value: Any, digits: int = 2) -> str:
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return MISSING


def _signed(value: float, digits: int) -> str:
    text = f"{value:+.{digits}f}"
    return f"{0:.{digits}f}" if float(text) == 0 else text


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _count(item: dict[str, Any], n: Any) -> str:
    accuracy = item.get("accuracy")
    if accuracy is None:
        return MISSING
    if isinstance(n, int) and n > 0:
        correct = float(accuracy) * n
        text = f"{correct:.0f}" if math.isclose(correct, round(correct), abs_tol=1e-9) else f"{correct:.2f}"
        return f"{text}/{n}"
    return _percent(accuracy)


def _difference(result: dict[str, Any], metric: str, scale: float, digits: int) -> str:
    item = result.get(metric)
    if not item:
        return MISSING
    low, high = item["ci95"]
    return (f"{_signed(item['point'] * scale, digits)} "
            f"[{_signed(low * scale, digits)}, {_signed(high * scale, digits)}]")


def _utility_verdict(result: dict[str, Any]) -> str:
    item = result.get("utility_difference")
    if not item:
        return MISSING
    low, high = item["ci95"]
    if low > 0:
        return "显著更优"
    if high < 0:
        return "显著更差"
    return "无法区分"


def _table(header: list[str], align: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


# --------------------------------------------------------------------------- report sections

def _key_metrics(summary: dict[str, Any]) -> list[str]:
    metrics = summary.get("metrics", {})
    n = summary.get("config", {}).get("test_size")
    best_name = _markdown(summary.get("best_fixed_strategy", MISSING))
    columns = [("learned", "梯度学习路由器"), ("best_fixed", f"最佳固定策略（{best_name}）"),
               ("initial_reason", "初始策略（reason）")]
    columns = [(key, label) for key, label in columns if key in metrics]
    if not columns:
        return []
    initial_tokens = metrics.get("initial_reason", {}).get("mean_tokens")

    def token_change(item: dict[str, Any]) -> str:
        try:
            return f"{_signed(100 * (float(item['mean_tokens']) / float(initial_tokens) - 1), 2)}%"
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return MISSING

    fields: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("正确题数", lambda item: _count(item, n)),
        ("正确率", lambda item: _percent(item.get("accuracy"))),
        ("平均 token / 题", lambda item: _number(item.get("mean_tokens"), 2)),
        ("token 相对初始策略", token_change),
        ("平均调用次数", lambda item: _number(item.get("mean_calls"), 2)),
        ("成本感知效用", lambda item: _number(item.get("utility"), 4)),
    ]
    rows = [[name] + [fn(metrics[key]) for key, _ in columns] for name, fn in fields]
    return ["## 关键数字", "",
            *_table(["测试集指标"] + [label for _, label in columns], ["---"] + ["---:"] * len(columns), rows), ""]


def _main_findings(summary: dict[str, Any]) -> list[str]:
    metrics = summary.get("metrics", {})
    comparisons = summary.get("comparisons", {})
    initial = metrics.get("initial_reason")
    learned = metrics.get("learned")
    best = metrics.get("best_fixed")
    if not initial or not learned or not best or not initial.get("mean_tokens"):
        return []
    n = summary.get("config", {}).get("test_size", 0)
    token_change = 100 * (learned["mean_tokens"] / initial["mean_tokens"] - 1)
    best_change = 100 * (best["mean_tokens"] / initial["mean_tokens"] - 1)
    best_name = _markdown(summary.get("best_fixed_strategy", MISSING))
    lines = [
        "## 主要结论",
        "",
        f"1. **相对初始 harness**：学习路由器与初始策略分别答对 **{_count(learned, n)}** 与 **{_count(initial, n)}**；"
        f"平均 token 从 {initial['mean_tokens']:.2f} 变为 {learned['mean_tokens']:.2f}（**{token_change:+.2f}%**），"
        f"效用配对比较结论为“{_utility_verdict(comparisons.get('vs_initial', {}))}”。",
        f"2. **相对最佳固定策略**：验证集事先选中的固定 `{best_name}` 答对 **{_count(best, n)}**，"
        f"平均仅用 {best['mean_tokens']:.2f} token（相对初始策略 **{best_change:+.2f}%**），"
        f"效用配对比较结论为“{_utility_verdict(comparisons.get('vs_best_fixed', {}))}”。"
        "梯度方法的增益必须同时与这一简单基线比较，不能只比较初始策略。",
    ]
    accuracy_difference = comparisons.get("vs_best_fixed", {}).get("accuracy_difference")
    if accuracy_difference and accuracy_difference["ci95"][0] <= 0 <= accuracy_difference["ci95"][1]:
        lines.append(
            "3. **准确率**：学习路由器与最佳固定策略的准确率差区间包含 0；"
            "即使某一方多答对了题目，也不能据此声称准确率显著不同。准确率证据与 token / 效用证据应分别解释。")
    oracle = metrics.get("oracle")
    if oracle and best["mean_tokens"] > 0 and math.isclose(oracle["accuracy"], best["accuracy"]):
        extra_tokens = best["mean_tokens"] - oracle["mean_tokens"]
        if extra_tokens >= 0:
            lines.append(
                f"{len(lines) - 1}. **路由空间诊断**：使用测试标签的 oracle 与最佳固定策略正确率相同，"
                f"仅再节省 **{extra_tokens:.4f} token/题（{100 * extra_tokens / best['mean_tokens']:.2f}%）**。"
                "这说明在本批题目、已采集的三条策略及当前效用权重下，精细路由超过固定选择的可用空间很小；"
                "oracle 不能部署，这一诊断也不能外推到其他模型或任务。")
    return lines + [""]


def _setup(summary: dict[str, Any]) -> list[str]:
    config = summary.get("config", {})
    cost_weight = float(config.get("cost_weight", 0.05))
    token_unit = float(config.get("token_unit", 1000))
    return [
        "## 实验设置",
        "",
        f"- 数据集：`{_markdown(config.get('dataset', MISSING))}`；模型：`{_markdown(config.get('model', MISSING))}`；"
        f"温度：{_markdown(config.get('temperature', MISSING))}。",
        f"- 训练 / 验证 / 测试题数：{config.get('train_size', MISSING)} / {config.get('val_size', MISSING)} / "
        f"{config.get('test_size', MISSING)}；随机种子：{config.get('seed', MISSING)}；请求尝试上限（含重试）：{config.get('max_requests', MISSING)}。",
        f"- 优化：{config.get('rounds', MISSING)} 轮 × 每轮 {config.get('steps', MISSING)} 步全批量梯度上升，"
        f"学习率 {config.get('lr', MISSING)}，熵系数 {config.get('entropy', MISSING)}，L2 系数 {config.get('l2', MISSING)}；"
        f"门控准确率容差 {config.get('gate_accuracy_tolerance', MISSING)}。",
        f"- 效用定义：`accuracy - {cost_weight:g} × mean_tokens / {token_unit:g}`。"
        "该权重表达预先设定的质量与成本偏好，改变权重会改变结论。",
        f"- 最佳固定策略由验证集选为 `{_markdown(summary.get('best_fixed_strategy', MISSING))}`；"
        f"最终冻结的路由器版本为 `{_markdown(summary.get('selected_version', MISSING))}`（0 表示初始版本）。",
        "- 路由器使用完整动作反馈的精确期望效用梯度；部署与验证执行确定性 argmax。均匀随机基线报告期望指标。",
        "",
    ]


def _results(summary: dict[str, Any]) -> list[str]:
    metrics = summary.get("metrics", {})
    n = summary.get("config", {}).get("test_size")
    rows = []
    for key, label in _LABELS.items():
        if key not in metrics:
            continue
        item = metrics[key]
        if key == "best_fixed":
            label = f"{label}（{_markdown(summary.get('best_fixed_strategy', MISSING))}）"
        if key == "learned":
            label = f"**{label}**"
        rows.append([label, _count(item, n), _percent(item.get("accuracy")), _number(item.get("mean_tokens"), 1),
                     _number(item.get("mean_calls"), 2), _number(item.get("mean_latency_s"), 3),
                     _number(item.get("utility"), 4)])
    return [
        "## 独立测试集结果",
        "",
        "除 oracle 外，所有策略都在查看测试结果前冻结。Oracle 使用测试标签逐题选择最优动作，"
        "只表示已采集动作的效用上界，不能部署，也不能作为已取得的收益。",
        "",
        *_table(["策略", "答对", "正确率", "平均 token", "平均调用次数", "平均延迟（秒）", "效用"],
                ["---", "---:", "---:", "---:", "---:", "---:", "---:"], rows),
        "",
        "![独立测试集上的 token 成本、正确题数与效用](summary.svg)",
        "",
    ]


def _evidence(summary: dict[str, Any]) -> list[str]:
    comparisons = summary.get("comparisons", {})
    best_utility = comparisons.get("vs_best_fixed", {}).get("utility_difference")
    if best_utility:
        lower, upper = best_utility["ci95"]
        if lower > 0:
            verdict = ("相对验证集选出的最佳固定策略，效用差的 95% 配对 bootstrap 区间下界大于 0，"
                       "支持在本测试分布和成本权重下取得效用提升。")
        elif upper < 0:
            verdict = ("相对验证集选出的最佳固定策略，效用差的 95% 配对 bootstrap 区间上界小于 0；"
                       "当前路由器劣于最佳固定策略，不能宣称梯度路由优于简单固定选择。")
        else:
            verdict = ("相对验证集选出的最佳固定策略，效用差的 95% 配对 bootstrap 区间包含 0，"
                       "尚不足以证明路由器取得稳定收益。")
    else:
        verdict = "未记录与最佳固定策略的配对比较，暂不能判断梯度路由是否超过简单固定选择。"
    rows = []
    for key, label in (("vs_initial", "相对初始策略"), ("vs_best_fixed", "相对最佳固定策略")):
        result = comparisons.get(key, {})
        rows.append([label, _difference(result, "accuracy_difference", 100, 2),
                     _difference(result, "token_difference", 1, 2),
                     _difference(result, "utility_difference", 1, 5), _utility_verdict(result)])
    return [
        "## 收益证据（逐题配对 bootstrap）",
        "",
        verdict,
        "",
        "差值均为“学习路由器 − 对照”，格式为“点估计 [95% 区间]”。准确率差以百分点表示；token 差为负表示节省。",
        "",
        *_table(["对照", "Δ 正确率（百分点）", "Δ 平均 token / 题", "Δ 效用", "效用结论"],
                ["---", "---:", "---:", "---:", "---"], rows),
        "",
        "区间按题配对重采样，反映该批测试题的采样不确定性；不包含模型采样方差、数据污染风险或重复挑选测试结果的偏差。"
        "小样本区间包含 0 时，应报告证据不足，不能把点估计提升写成确定结论。效用提升也不等于准确率提升。",
        "",
    ]


def _rounds(summary: dict[str, Any]) -> list[str]:
    rows = []
    for item in summary.get("rounds", []):
        reason = str(item.get("reason", ""))
        rows.append([
            _markdown(item.get("round", "")), _markdown(item.get("train_tasks", "")),
            f"{_number(item.get('objective_start'), 5)} → {_number(item.get('objective_end'), 5)}",
            _number(item.get("val_incumbent"), 5), _number(item.get("val_candidate"), 5),
            "**接纳**" if item.get("accepted") else "拒绝",
            _markdown(_ROUND_REASONS.get(reason, reason) or MISSING),
        ])
    selected_actions = summary.get("selected_actions", {})
    review_effect = summary.get("review_effect", {})
    format_failures = summary.get("format_failures", {})
    lines = [
        "## 自我改进循环",
        "",
        "每轮：新一批训练题 → 采集全部动作 → 在累计奖励矩阵上做精确梯度上升（暖启动）→ 验证门控 → 接纳或保留现任版本。",
        "",
    ]
    if rows:
        lines += _table(["轮次", "累计训练题", "训练目标 J（起点 → 终点）", "现任验证效用", "候选验证效用", "门控", "说明"],
                        ["---:", "---:", "---", "---:", "---:", "---", "---"], rows)
    else:
        lines.append("未记录轮次信息。")
    lines += [
        "",
        f"最终测试路由分布：direct={selected_actions.get('direct', MISSING)}，"
        f"reason={selected_actions.get('reason', MISSING)}，review={selected_actions.get('review', MISSING)}。",
        "",
        f"复核（review）相对其初稿：纠正 {review_effect.get('fixes', MISSING)} 题，"
        f"将原本正确的答案改错 {review_effect.get('breaks', MISSING)} 题。复核并不保证改善，因此同时记录纠错与改错。",
        "",
    ]
    if format_failures:
        lines += [
            "测试集上未给出有效数值答案的题数："
            + "，".join(f"{_markdown(key)}={value}" for key, value in format_failures.items())
            + "。这些题按答错计分，并保留其 token 成本。",
            "",
        ]
    lines += [
        "验证集上的候选比较是选择依据，不能作为最终泛化收益。若所有候选均未被接受，保留初始版本也是有效实验结果。",
        "",
    ]
    return lines


def _cost(summary: dict[str, Any]) -> list[str]:
    metrics = summary.get("metrics", {})
    comparisons = summary.get("comparisons", {})
    ledger = summary.get("ledger", {})
    learned = metrics.get("learned", {})
    initial = metrics.get("initial_reason", {})
    lines = [
        "## 收集成本与部署摊销",
        "",
        *_table(["请求尝试（含失败重试）", "成功", "失败", "输入 token", "输出 token", "合计 token"],
                ["---:"] * 6,
                [[str(ledger.get(key, MISSING)) for key in
                  ("attempts", "successes", "failures", "input_tokens", "output_tokens", "total_tokens")]]),
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
                "该数字是有条件估算，不是已经获得的净收益。")
    else:
        lines.append("缺少初始策略、最终策略或收集 token 数据，无法估算部署摊销门槛。")
    return lines + [""]


_BOUNDARIES = [
    "## 解释边界",
    "",
    "- 本实验只更新小型路由器，不训练基础语言模型，也不需要 GPU。",
    "- 完整动作反馈会产生额外探索开销；缓存和重复梯度更新不等于免费的新模型反馈。",
    "- 测试标签只用于最终评测及标注为 oracle 的上界，不得用于选择轮次、特征或成本权重。",
    "- 同一验证集多轮选择可能过拟合；报告的是最终冻结版本在独立测试集上的表现。",
    "- 延迟受网络、并发及缓存影响，单次观测不能证明稳定的生产延迟收益。",
    "- 若数据集标识为 synthetic/demo/fixture，相应结果只证明工程链路可运行，不能替代真实模型 benchmark。",
    "",
]


def build_markdown(summary: dict[str, Any]) -> str:
    """Return the REPORT.md text for a run summary."""
    run_name = str(summary.get("run_name", "Harness RSI benchmark"))
    lines = [
        f"# {_markdown(run_name)}：实验测评报告",
        "",
        "> 本文件由 `harness_rsi/report.py` 根据同目录的 `summary.json` 自动生成；"
        "运行 `python scripts/replay_report.py --run <运行目录>` 可在不调用模型、不读取密钥的情况下离线复核全部数值并重建本文件。",
        "",
        "本实验固定基础模型与提示词，只在 CPU 上用梯度更新一个小型路由器，为每道题选择执行路线（direct / reason / review）。"
        "循环包括收集训练反馈、更新候选路由器、验证门控和接纳或回退。"
        "这是参数化 harness 的受限自我改进，不代表任意代码递归修改或通用能力持续增长。",
        "",
        *_key_metrics(summary),
        *_main_findings(summary),
        *_setup(summary),
        *_results(summary),
        *_evidence(summary),
        *_rounds(summary),
        *_cost(summary),
        *_BOUNDARIES,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- SVG chart

def _chart_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    metrics = summary.get("metrics", {})
    best_name = str(summary.get("best_fixed_strategy", ""))
    rows = [("learned", "梯度学习路由器（GFPG）"), ("best_fixed", f"最佳固定策略：{best_name or MISSING}")]
    if best_name != "reason":
        rows.append(("initial_reason", "初始策略：reason"))
    for strategy, label in (("direct", "固定策略：direct"), ("reason", "固定策略：reason"),
                            ("review", "固定策略：review")):
        if strategy not in (best_name, "reason"):
            rows.append((f"fixed_{strategy}", label))
    rows += [("uniform", "均匀随机（期望）"), ("oracle", "Oracle（不可部署）")]
    return [(key, label) for key, label in rows if key in metrics]


def _svg(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    config = summary.get("config", {})
    n = config.get("test_size")
    rows = _chart_rows(summary)
    run_name = str(summary.get("run_name", "Harness RSI benchmark"))
    cost_weight = float(config.get("cost_weight", 0.05))
    token_unit = float(config.get("token_unit", 1000))
    width, top, row_height = 960, 104, 40
    height = top + row_height * max(len(rows), 1) + 70
    bar_x, bar_width, count_x, utility_x = 250, 330, 760, 920
    tokens = [float(metrics[key].get("mean_tokens") or 0) for key, _ in rows]
    max_tokens = max(tokens + [1.0])
    test_text = f"（{n} 题）" if n else ""
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(run_name)}：独立测试集上的 token 成本、正确题数与效用</title>',
        '<desc id="desc">每行一种策略：横条为平均 token（越短越省），右侧为正确题数与成本感知效用。'
        '蓝色为梯度学习路由器，橙色为验证集选出的最佳固定策略，灰色为其他基线；oracle 使用测试标签，不可部署。</desc>',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<style>text{{font-family:{_FONT};fill:{_INK2};font-size:13px}}'
        f'.title{{fill:{_INK};font-size:18px;font-weight:600}}.head{{fill:{_INK};font-size:13px;font-weight:600}}'
        f'.strong{{fill:{_INK};font-weight:600}}.note{{fill:{_MUTED};font-size:12px}}</style>',
        f'<text class="title" x="24" y="36">{escape(run_name)}：独立测试集{escape(test_text)}部署指标</text>',
        f'<text class="head" x="{bar_x}" y="{top - 22}">平均 token / 题（越短越省）</text>',
        f'<text class="head" x="{count_x}" y="{top - 22}" text-anchor="end">正确题数</text>',
        f'<text class="head" x="{utility_x}" y="{top - 22}" text-anchor="end">效用 ↑</text>',
        f'<line x1="24" y1="{top - 10}" x2="{width - 24}" y2="{top - 10}" stroke="{_GRID}" stroke-width="1"/>',
    ]
    for index, (key, label) in enumerate(rows):
        item = metrics[key]
        y = top + index * row_height
        value = float(item.get("mean_tokens") or 0)
        length = max(2.0, bar_width * value / max_tokens)
        if key == "learned":
            fill, stroke = _BLUE, "none"
        elif key == "best_fixed":
            fill, stroke = _ORANGE, "none"
        elif key == "oracle":
            fill, stroke = _ORACLE, _MUTED
        else:
            fill, stroke = _GRAY, "none"
        emphasis = ' class="strong"' if key in ("learned", "best_fixed") else ""
        dash = ' stroke-dasharray="3 2"' if key == "oracle" else ""
        chunks.extend([
            f'<text{emphasis} x="24" y="{y + 16}">{escape(label)}</text>',
            f'<rect x="{bar_x}" y="{y + 3}" width="{length:.1f}" height="18" rx="4" fill="{fill}" '
            f'stroke="{stroke}"{dash}/>',
            f'<text x="{bar_x + length + 8:.1f}" y="{y + 16}">{value:.1f}</text>',
            f'<text{emphasis} x="{count_x}" y="{y + 16}" text-anchor="end">{escape(_count(item, n))}</text>',
            f'<text{emphasis} x="{utility_x}" y="{y + 16}" text-anchor="end">{escape(_number(item.get("utility"), 4))}</text>',
        ])
    bottom = top + row_height * max(len(rows), 1)
    chunks.extend([
        f'<line x1="24" y1="{bottom}" x2="{width - 24}" y2="{bottom}" stroke="{_GRID}" stroke-width="1"/>',
        f'<text class="note" x="24" y="{bottom + 26}">蓝色：梯度学习路由器；橙色：验证集选出的最佳固定策略；灰色：其他基线；'
        f'虚线框：oracle 使用测试标签，仅作上界、不可部署。</text>',
        f'<text class="note" x="24" y="{bottom + 46}">效用 = 正确率 − {cost_weight:g} × 平均 token / {token_unit:g}。'
        f'正确率与 token 成本分开报告；token 不等于货币价格。</text>',
        '</svg>',
    ])
    return "\n".join(chunks) + "\n"


def write_report(summary: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write REPORT.md and summary.svg (UTF-8, LF line endings) from the recorded results."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "REPORT.md"
    chart_path = directory / "summary.svg"
    report_path.write_text(build_markdown(summary), encoding="utf-8", newline="\n")
    chart_path.write_text(_svg(summary), encoding="utf-8", newline="\n")
    return {"report": str(report_path), "chart": str(chart_path)}
