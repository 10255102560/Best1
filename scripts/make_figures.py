"""Render the course-report figures and LaTeX tables from the saved v1 run.

Usage (from any directory):
    python scripts/make_figures.py

Outputs:
    report/figures/*.pdf   vector figures included by report/main.tex
    report/figures/*.png   raster copies used by README.md
    report/generated/*.tex numbers and tables, so no result is hand-transcribed

Everything here is offline: no credentials are read and no model is called.
Only recorded artifacts in artifacts/gsm8k_flash_v1 are visualised; nothing
changes the experiment's results. Two panels are explicitly labelled offline
illustrations of the gradient method itself (a finite-difference check and a
REINFORCE-vs-exact-gradient comparison on the recorded reward matrix) and one
uses the synthetic routing problem from tests/test_policy.py.

Requires matplotlib (pip install -r requirements-report.txt).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from harness_rsi.policy import FEATURE_NAMES, LinearPolicy, extract_features, objective_and_gradient
from harness_rsi.strategies import STRATEGIES

RUN = REPO_ROOT / "artifacts" / "gsm8k_flash_v1"
FIG_DIR = REPO_ROOT / "report" / "figures"
GEN_DIR = REPO_ROOT / "report" / "generated"

# Validated palette (blue/orange pass CVD + contrast checks on white); text never
# wears a series colour. Status colours are used only for request outcomes.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_WASH = "#e3eefb"
GRAY, GRAY_LIGHT = "#a8a69e", "#efeee9"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
TEXT_WIDTH = 6.3  # inches; matches \textwidth of report/main.tex


# --------------------------------------------------------------------------- style

def setup_style() -> None:
    preferred = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = [name for name in preferred if name in available][:1] + ["DejaVu Sans"]
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": family, "font.size": 8.5,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": 8.2, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 4, "ytick.major.pad": 5,
        "legend.fontsize": 7.8, "legend.frameon": False, "legend.handlelength": 1.4,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        # Type 3 embedding is portable for the CFF-based CJK system fonts.
        "pdf.fonttype": 3, "svg.fonttype": "none", "axes.unicode_minus": True,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03, dpi=300,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.06,
                metadata={"Software": None})
    plt.close(fig)


def hgrid(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _units_per_inch(ax: plt.Axes) -> tuple[float, float]:
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pos = ax.get_position()
    return (x1 - x0) / (pos.width * fig.get_figwidth()), abs(y1 - y0) / (pos.height * fig.get_figheight())


def rounded_box(ax: plt.Axes, x: float, y: float, width: float, height: float, color: str,
                radius_in: float = 0.03, zorder: int = 2) -> None:
    """A box with physically round corners regardless of the axes' data aspect."""
    units_x, units_y = _units_per_inch(ax)
    radius = min(radius_in * units_x, width / 2, (height / 2) * units_x / units_y)
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
                                mutation_aspect=units_y / units_x, facecolor=color, edgecolor="none",
                                zorder=zorder))


def rounded_hbar(ax: plt.Axes, y: float, width: float, height: float, color: str,
                 radius_in: float = 0.03) -> None:
    """Horizontal bar: square at the baseline, rounded at the data end."""
    if width <= 0:
        return
    units_x, _ = _units_per_inch(ax)
    rounded_box(ax, 0, y - height / 2, width, height, color, radius_in)
    ax.add_patch(Rectangle((0, y - height / 2), max(width - radius_in * units_x * 1.2, 0), height,
                           facecolor=color, edgecolor="none", zorder=2))


def rounded_vbar(ax: plt.Axes, x: float, height: float, width: float, color: str,
                 radius_in: float = 0.03) -> None:
    """Vertical bar: square at the baseline, rounded at the data end."""
    if height <= 0:
        return
    _, units_y = _units_per_inch(ax)
    rounded_box(ax, x - width / 2, 0, width, height, color, radius_in)
    ax.add_patch(Rectangle((x - width / 2, 0), width, max(height - radius_in * units_y * 1.2, 0),
                           facecolor=color, edgecolor="none", zorder=2))


def dot(ax: plt.Axes, x: float, y: float, color: str, hollow: bool = False, size: float = 6.5) -> None:
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor="white" if hollow else color,
            markeredgecolor=color if hollow else "white", markeredgewidth=1.4 if hollow else 1.2,
            linestyle="none", zorder=4)


# --------------------------------------------------------------------------- data

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_run() -> dict:
    summary = load_json(RUN / "summary.json")
    manifest = load_json(RUN / "split_manifest.json")
    rows = {split: [load_json(RUN / "observations" / split / f"{task['id']}.json") for task in manifest[split]]
            for split in ("train", "val", "test")}
    events = [json.loads(line) for line in (RUN / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"summary": summary, "rows": rows, "events": events,
            "probe": load_json(RUN / "probe_accounting.json"), "rounds": load_json(RUN / "rounds.json"),
            "frozen": load_json(RUN / "frozen_policy.json")}


def field_matrix(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray([[row["outcomes"][action][field] for action in STRATEGIES] for row in rows], dtype=float)


def features(rows: list[dict]) -> np.ndarray:
    return np.asarray([extract_features(row["task"]["question"]) for row in rows])


def initial_policy(seed: int) -> LinearPolicy:
    policy = LinearPolicy(len(FEATURE_NAMES), len(STRATEGIES), seed=seed)
    policy.weights[:] = 0
    policy.bias[1] = 0.5
    return policy


# --------------------------------------------------------------------------- figures

ROW_KEYS = ["learned", "best_fixed", "initial_reason", "fixed_review", "uniform", "oracle"]
ROW_LABELS = {
    "learned": "梯度学习路由器（GFPG）",
    "best_fixed": "最佳固定策略：direct",
    "initial_reason": "初始策略：reason",
    "fixed_review": "固定策略：review",
    "uniform": "均匀随机（期望）",
    "oracle": "Oracle 上界（不可部署）",
}
ROW_COLORS = {"learned": BLUE, "best_fixed": ORANGE}


def fig_main_results(data: dict) -> None:
    summary = data["summary"]
    metrics = summary["metrics"]
    n = summary["config"]["test_size"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.45), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})
    fig.subplots_adjust(left=0.215, right=0.985, top=0.80, bottom=0.17, wspace=0.12)
    ys = np.arange(len(ROW_KEYS))[::-1].astype(float)
    panels = [
        ("答对题数（/%d）↑" % n, lambda m: m["accuracy"] * n, (121.4, 125.0), [122, 123, 124, 125], "{:.0f}"),
        ("平均 token / 题 ↓", lambda m: m["mean_tokens"], (0, 880), [0, 200, 400, 600, 800], "{:.0f}"),
        ("成本感知效用 ↑", lambda m: m["utility"], (0.915, 0.962), [0.92, 0.93, 0.94, 0.95, 0.96], "{:.4f}"),
    ]
    for panel, (ax, (title, value, xlim, ticks, fmt)) in enumerate(zip(axes, panels)):
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(ROW_KEYS) - 0.4)
        ax.set_xticks(ticks)
        ax.set_title(title, fontsize=8.3)
        hgrid(ax)
        ax.spines["left"].set_visible(panel == 0)
        for key, y in zip(ROW_KEYS, ys):
            v = float(value(metrics[key]))
            color = ROW_COLORS.get(key, GRAY)
            if panel == 1:
                rounded_hbar(ax, y, v, 0.46, color if key != "oracle" else "#dddbd4")
                if key in ROW_COLORS:
                    ax.text(v + 14, y, fmt.format(v), va="center", ha="left", fontsize=7.6, color=INK2)
            else:
                ax.plot([xlim[0], xlim[1]], [y, y], color=GRID, linewidth=0.7, zorder=1)
                dot(ax, v, y, color, hollow=(key == "oracle"))
                if key in ROW_COLORS:
                    label = fmt.format(v)
                    if panel == 0:
                        label = f"{v:.0f}/{n}"
                    ax.text(v, y + 0.36, label, va="bottom", ha="center", fontsize=7.4, color=INK2)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([ROW_LABELS[key] for key in ROW_KEYS])
    for tick, key in zip(axes[0].get_yticklabels(), ROW_KEYS):
        if key in ROW_COLORS:
            tick.set_color(INK)
            tick.set_fontweight("bold")
    fig.text(0.215, 0.955, "独立测试集（GSM8K 官方 test 抽样 128 题）上的部署指标", fontsize=9.2,
             fontweight="bold", color=INK, ha="left")
    save(fig, "fig_main_results")


def fig_paired_ci(data: dict) -> None:
    comparisons = data["summary"]["comparisons"]
    rows = [("vs_initial", "相对初始 reason"), ("vs_best_fixed", "相对最佳固定 direct")]
    panels = [
        ("Δ 正确率（百分点）", "accuracy_difference", 100.0, "{:+.2f}"),
        ("Δ 平均 token / 题", "token_difference", 1.0, "{:+.2f}"),
        ("Δ 效用（×10⁻³）", "utility_difference", 1000.0, "{:+.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.85), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.70, bottom=0.2, wspace=0.16)
    ys = [1.0, 0.0]
    for panel, (ax, (title, metric, scale, fmt)) in enumerate(zip(axes, panels)):
        values = [(comparisons[key][metric]["point"] * scale, *[v * scale for v in comparisons[key][metric]["ci95"]])
                  for key, _ in rows]
        low = min(v[1] for v in values + [(0, 0, 0)])
        high = max(v[2] for v in values + [(0, 0, 0)])
        span = high - low or 1.0
        ax.set_xlim(low - 0.18 * span, high + 0.18 * span)
        ax.set_ylim(-0.6, 1.85)
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
        hgrid(ax)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=8.3)
        label_transform = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for (point, lo, hi), y in zip(values, ys):
            ax.plot([lo, hi], [y, y], color=BLUE, linewidth=2.2, zorder=3)
            dot(ax, point, y, BLUE)
            ax.text(0.02, y + 0.3, f"{fmt.format(point)}  [{fmt.format(lo)}, {fmt.format(hi)}]",
                    transform=label_transform, ha="left", va="bottom", fontsize=6.9, color=INK2)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _, label in rows])
    fig.text(0.17, 0.95, "学习路由器 − 对照：逐题配对 bootstrap 95% 区间（5000 次重采样）", fontsize=9.2,
             fontweight="bold", color=INK, ha="left")
    save(fig, "fig_paired_ci")


def round_decision_changes(data: dict) -> list[dict]:
    """Per round: how many validation decisions the candidate changes vs the incumbent."""
    seed = data["summary"]["config"]["seed"]
    xv = features(data["rows"]["val"])
    xt_all = features(data["rows"]["train"])
    incumbent = initial_policy(seed)
    result = []
    for entry in data["rounds"]:
        candidate = LinearPolicy.from_dict(load_json(RUN / "versions" / f"candidate_{entry['round']}.json"))
        train_x = xt_all[:entry["train_tasks"]]
        result.append({
            "round": entry["round"],
            "val_changed": int(np.sum(candidate.predict(xv) != incumbent.predict(xv))),
            "train_changed": int(np.sum(candidate.predict(train_x) != incumbent.predict(train_x))),
            "val_direct": int(np.sum(candidate.predict(xv) == 0)),
            "accepted": bool(entry["accepted"]),
        })
        if entry["accepted"]:
            incumbent = candidate
    return result


def fig_rounds(data: dict, changes: list[dict]) -> None:
    rounds = data["rounds"]
    summary = data["summary"]
    val_rows = data["rows"]["val"]
    val_correct = field_matrix(val_rows, "correct")
    val_tokens = field_matrix(val_rows, "total_tokens")
    cw, unit = summary["config"]["cost_weight"], summary["config"]["token_unit"]
    direct_utility = float((val_correct - cw * val_tokens / unit).mean(axis=0)[0])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), gridspec_kw={"width_ratios": [1.6, 1.0]})
    fig.subplots_adjust(left=0.09, right=0.985, top=0.79, bottom=0.2, wspace=0.27)
    incumbent = [r["val_incumbent"]["utility"] for r in rounds]
    candidate = [r["val_candidate"]["utility"] for r in rounds]
    accepted = [r["accepted"] for r in rounds]
    after = [c if a else i for i, c, a in zip(incumbent, candidate, accepted)]
    xs = np.arange(0, len(rounds) + 1)
    ax.set_xlim(-0.35, len(rounds) + 0.45)
    ax.set_ylim(0.9620, 0.9651)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axhline(direct_utility, color=ORANGE, linewidth=1.0, zorder=1)
    ax.text(-0.3, direct_utility + 0.00006, f"参照：验证集上固定 direct = {direct_utility:.4f}", ha="left",
            va="bottom", fontsize=7.2, color=INK2)
    ax.plot(xs, [incumbent[0]] + after, color=BLUE, linewidth=2, marker="o", markersize=4.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    for xi, inc, cand, acc in zip(xs[1:], incumbent, candidate, accepted):
        ax.plot([xi + 0.2], [cand], marker="o", markersize=6.5, linestyle="none", zorder=5,
                markerfacecolor=BLUE if acc else "white", markeredgecolor=BLUE if acc else GRAY,
                markeredgewidth=1.3)
        if acc:
            ax.annotate(f"接纳（效用 +{cand - inc:.5f}）", xy=(xi + 0.2, cand), xytext=(xi - 1.25, cand + 0.00095),
                        fontsize=7.3, color=INK2, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_xticks(xs)
    ax.set_xticklabels(["初始"] + [f"第{r['round']}轮\n累计{r['train_tasks']}题" for r in rounds], fontsize=7.3)
    ax.set_ylabel("验证集效用（48 题）")
    ax.set_title("(a) 现任版本（折线）与本轮候选（圆点）", fontsize=8.5)
    ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=2, label="现任版本"),
                       Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE,
                              markersize=6, label="候选：接纳"),
                       Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=GRAY,
                              markersize=6, label="候选：拒绝")],
              loc="upper left", bbox_to_anchor=(0.0, 0.82), ncol=3, fontsize=7.2, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.3)
    values = [c["val_changed"] for c in changes]
    x = np.arange(1, len(rounds) + 1)
    bx.set_xlim(0.4, len(rounds) + 0.6)
    bx.set_ylim(0, max(values + [1]) * 1.3 + 0.5)
    bx.grid(axis="y", color=GRID, linewidth=0.7)
    bx.set_axisbelow(True)
    for xi, v in zip(x, values):
        rounded_vbar(bx, xi, v, 0.34, BLUE)
        bx.text(xi, v + bx.get_ylim()[1] * 0.025, str(v), ha="center", va="bottom", fontsize=7.6, color=INK2)
    bx.set_xticks(x)
    bx.set_xticklabels([f"第{r['round']}轮" for r in rounds], fontsize=7.4)
    bx.set_ylabel("验证集题数")
    bx.set_title("(b) 候选改变的部署决策数", fontsize=8.5)
    fig.text(0.09, 0.955, "四轮增量自我改进：未改变任何部署决策的候选必然被门控拒绝", fontsize=9.2,
             fontweight="bold", color=INK, ha="left")
    save(fig, "fig_rounds")


def fig_tokens(data: dict) -> None:
    rows = data["rows"]["test"]
    tokens = field_matrix(rows, "total_tokens")
    labels = {"direct": "direct：只给答案（1 次调用）", "reason": "reason：简短分步（1 次调用）",
              "review": "review：分步 + 复核（2 次调用）"}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.9))
    fig.subplots_adjust(left=0.27, right=0.80, top=0.80, bottom=0.22)
    rng = np.random.default_rng(0)
    ys = np.arange(len(STRATEGIES))[::-1].astype(float)
    hgrid(ax)
    for index, (action, y) in enumerate(zip(STRATEGIES, ys)):
        values = tokens[:, index]
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        lo = values[values >= q1 - 1.5 * iqr].min()
        hi = values[values <= q3 + 1.5 * iqr].max()
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(values, y + jitter, s=5, color=GRAY, alpha=0.45, linewidths=0, zorder=2)
        ax.plot([lo, q1], [y, y], color=INK2, linewidth=1.0, zorder=3)
        ax.plot([q3, hi], [y, y], color=INK2, linewidth=1.0, zorder=3)
        ax.add_patch(Rectangle((q1, y - 0.2), q3 - q1, 0.4, facecolor=BLUE_WASH, edgecolor=INK2,
                               linewidth=1.0, zorder=3))
        ax.plot([med, med], [y - 0.2, y + 0.2], color=INK, linewidth=1.6, zorder=4)
        mean = values.mean()
        ax.plot([mean], [y], marker="D", markersize=4.5, color=BLUE, markeredgecolor="white",
                markeredgewidth=0.8, zorder=5)
        ax.text(7600, y, f"均值 {mean:.1f}｜中位数 {med:.0f}", va="center", ha="left", fontsize=7.4, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[a] for a in STRATEGIES])
    ax.set_ylim(-0.6, len(STRATEGIES) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(60, 7000)
    ax.set_xticks([100, 200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["100", "200", "500", "1000", "2000", "5000"])
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("单题总 token（对数轴；输入 + 输出，review 计两步）")
    ax.legend(handles=[Patch(facecolor=BLUE_WASH, edgecolor=INK2, label="四分位区间与中位数"),
                       Line2D([0], [0], marker="D", color="none", markerfacecolor=BLUE, markeredgecolor="white",
                              markersize=5, label="均值"),
                       Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="none",
                              markersize=4, label="单题观测")],
              loc="lower right", ncol=3, fontsize=7.2, bbox_to_anchor=(1.25, 1.0), handletextpad=0.4)
    fig.text(0.27, 0.955, "测试集 128 题的单题 token 分布", fontsize=9.2, fontweight="bold", color=INK, ha="left")
    save(fig, "fig_tokens")


def request_breakdown(data: dict) -> dict:
    events = data["events"]
    failures = [e for e in events if e["event"] == "failure"]
    successes = [e for e in events if e["event"] == "success"]
    by_status = {"HTTP 429 限流": 0, "HTTP 503 不可用": 0, "无效/缺失内容": 0}
    for event in failures:
        status = event.get("http_status")
        if status == 429:
            by_status["HTTP 429 限流"] += 1
        elif status is not None and 500 <= status < 600:
            by_status["HTTP 503 不可用"] += 1
        else:
            by_status["无效/缺失内容"] += 1
    by_split = {"val": 0, "train": 0, "test": 0}
    for event in successes:
        by_split[event["tag"].split(":", 1)[0]] += 1
    return {"attempts": sum(e["event"] == "attempt" for e in events), "successes": len(successes),
            "failures": len(failures), "by_status": by_status, "by_split": by_split,
            "cap": data["summary"]["config"]["max_requests"]}


def fig_budget(data: dict, breakdown: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 1.9))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.76, bottom=0.04, hspace=1.35)
    cap = breakdown["cap"]
    rows = [
        ("请求尝试（上限 %d）" % cap,
         [("成功", breakdown["successes"], GOOD, "#ffffff")]
         + [(name, count, color, INK) for (name, count), color in zip(breakdown["by_status"].items(),
                                                                     (WARNING, SERIOUS, CRITICAL))]
         + [("未使用预算", cap - breakdown["attempts"], GRAY_LIGHT, INK2)]),
        ("成功请求按阶段",
         [("验证集采集", breakdown["by_split"]["val"], BLUE, "#ffffff"),
          ("训练集采集（四轮）", breakdown["by_split"]["train"], ORANGE, "#ffffff"),
          ("冻结后测试", breakdown["by_split"]["test"], AQUA, INK)]),
    ]
    for ax, (title, segments) in zip(axes, rows):
        total = cap
        ax.set_xlim(0, total)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(-total * 0.012, 0.5, title, ha="right", va="center", fontsize=7.9, color=INK2)
        units_x, _ = _units_per_inch(ax)
        gap = 2 / 72 * units_x  # a 2pt surface gap between blocks
        left = 0.0
        handles = []
        for name, count, color, text_color in segments:
            if count <= 0:
                continue
            rounded_box(ax, left, 0.0, max(count - gap, count * 0.4), 1.0, color, radius_in=0.035)
            if count / total > 0.06:
                ax.text(left + 0.06 / 1 * units_x, 0.5, f"{count}", ha="left", va="center", fontsize=7.6,
                        color=text_color, fontweight="bold")
            handles.append(Patch(facecolor=color, edgecolor="none", label=f"{name}（{count}）"))
            left += count
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.1), ncol=len(handles),
                  fontsize=7.0, handlelength=1.0, handletextpad=0.4, columnspacing=0.9, borderaxespad=0)
    fig.text(0.2, 0.985, "预算账本：所有尝试（含失败重试）都计入上限", fontsize=9.2, fontweight="bold",
             color=INK, ha="left")
    save(fig, "fig_budget")


def gradient_check(data: dict) -> dict:
    """Central finite differences on the real normalised training features."""
    rows = data["rows"]["train"]
    config = data["summary"]["config"]
    x = features(rows)
    rewards = field_matrix(rows, "correct") - config["cost_weight"] * field_matrix(rows, "total_tokens") / config["token_unit"]
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (x - x.mean(axis=0)) / scale
    rng = np.random.default_rng(2026)
    weights = rng.normal(scale=0.3, size=(len(FEATURE_NAMES), len(STRATEGIES)))
    bias = rng.normal(scale=0.3, size=len(STRATEGIES))
    options = {"entropy": config["entropy"], "l2": config["l2"]}
    _, grad_w, grad_b = objective_and_gradient(weights, bias, normalized, rewards, **options)
    eps = 1e-6
    numeric = []
    analytic = []
    for parameter, gradient in ((weights, grad_w), (bias, grad_b)):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + eps
            upper = objective_and_gradient(weights, bias, normalized, rewards, **options)[0]
            parameter[index] = original - eps
            lower = objective_and_gradient(weights, bias, normalized, rewards, **options)[0]
            parameter[index] = original
            numeric.append((upper - lower) / (2 * eps))
            analytic.append(gradient[index])
    analytic_arr, numeric_arr = np.asarray(analytic), np.asarray(numeric)
    error = np.abs(analytic_arr - numeric_arr)
    return {"analytic": analytic_arr, "numeric": numeric_arr, "max_abs_error": float(error.max()),
            "max_rel_error": float((error / np.maximum(np.abs(numeric_arr), 1e-12)).max()),
            "n_params": len(analytic_arr)}


def reinforce_vs_exact(data: dict, repetitions: int = 400) -> dict:
    """Offline: REINFORCE direction quality vs actual API calls on the recorded rewards.

    At temperature 0 a cached (task, action) outcome is deterministic, so repeated
    samples cost no extra calls; review reuses a sampled reason draft when present.
    """
    rows = data["rows"]["train"]
    config = data["summary"]["config"]
    x = features(rows)
    rewards = field_matrix(rows, "correct") - config["cost_weight"] * field_matrix(rows, "total_tokens") / config["token_unit"]
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x_tilde = np.hstack([(x - x.mean(axis=0)) / scale, np.ones((len(x), 1))])
    logits = np.zeros((len(x), len(STRATEGIES)))
    logits[:, 1] = 0.5
    pi = np.exp(logits - logits.max(axis=1, keepdims=True))
    pi /= pi.sum(axis=1, keepdims=True)
    expected = np.sum(pi * rewards, axis=1, keepdims=True)
    exact = (x_tilde[:, :, None] * (pi * (rewards - expected))[:, None, :]).mean(axis=0).ravel()
    rng = np.random.default_rng(7)
    n = len(x)
    if not np.allclose(pi, pi[0]):
        raise AssertionError("the initial policy is expected to be context-independent")
    rows_index = np.arange(n)[:, None]
    results = []
    for k in (1, 2, 3, 4, 6, 8, 12, 16):
        cosines, calls = [], []
        for _ in range(repetitions):
            actions = rng.choice(len(STRATEGIES), size=(n, k), p=pi[0])
            sampled = rewards[rows_index, actions]
            baseline = sampled.mean()
            one_hot = np.eye(len(STRATEGIES))[actions]  # n, k, A
            score = (one_hot - pi[:, None, :]) * (sampled - baseline)[:, :, None]
            estimate = (x_tilde[:, :, None] * score.mean(axis=1)[:, None, :]).mean(axis=0).ravel()
            cosines.append(float(estimate @ exact / (np.linalg.norm(estimate) * np.linalg.norm(exact) + 1e-300)))
            seen = np.zeros((n, len(STRATEGIES)), dtype=bool)
            seen[rows_index, actions] = True
            # direct: 1 call; reason draft: 1 call (shared with review); review check: +1 call.
            calls.append(int(seen[:, 0].sum() + (seen[:, 1] | seen[:, 2]).sum() + seen[:, 2].sum()))
        results.append({"k": k, "calls": float(np.mean(calls)), "cos_mean": float(np.mean(cosines)),
                        "cos_lo": float(np.percentile(cosines, 10)), "cos_hi": float(np.percentile(cosines, 90))})
    return {"exact_calls": 3 * n, "curve": results, "n": n}


def fig_gradient_method(check: dict, reinforce: dict) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.45), gridspec_kw={"width_ratios": [1.0, 1.35]})
    fig.subplots_adjust(left=0.085, right=0.975, top=0.79, bottom=0.2, wspace=0.32)
    a, num = check["analytic"] * 100, check["numeric"] * 100
    lim = max(np.abs(a).max(), np.abs(num).max()) * 1.12
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, linewidth=1.0, zorder=1)
    ax.scatter(a, num, s=16, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("解析梯度分量（×10⁻²）")
    ax.set_ylabel("中心差分估计（×10⁻²）")
    ax.set_title(f"(a) 解析梯度 vs 有限差分（{check['n_params']} 个参数）", fontsize=8.5)
    ax.text(0.04, 0.96, f"最大绝对误差 {check['max_abs_error']:.1e}", transform=ax.transAxes, ha="left",
            va="top", fontsize=7.4, color=INK2)
    curve = reinforce["curve"]
    calls = np.array([c["calls"] for c in curve])
    bx.fill_between(calls, [c["cos_lo"] for c in curve], [c["cos_hi"] for c in curve], color=GRAY, alpha=0.18,
                    linewidth=0, zorder=1)
    bx.plot(calls, [c["cos_mean"] for c in curve], color=GRAY, linewidth=2, marker="o", markersize=4,
            markerfacecolor=GRAY, markeredgecolor="white", zorder=3)
    bx.plot([reinforce["exact_calls"]], [1.0], marker="o", markersize=7.5, color=BLUE, markeredgecolor="white",
            markeredgewidth=1.2, linestyle="none", zorder=5)
    bx.annotate(f"全信息精确梯度\n{reinforce['exact_calls']} 次调用，cos = 1", xy=(reinforce["exact_calls"], 1.0),
                xytext=(reinforce["exact_calls"] * 1.08, 0.80), fontsize=7.3, color=INK2,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    first = curve[0]
    bx.annotate(f"REINFORCE（每题 1 个采样）\n{first['calls']:.0f} 次调用，cos ≈ {first['cos_mean']:.2f}",
                xy=(first["calls"], first["cos_mean"]), xytext=(12, 0.62),
                fontsize=7.3, color=INK2, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    bx.set_xlim(0, max(calls.max(), reinforce["exact_calls"]) * 1.12)
    bx.set_ylim(min(-0.05, min(c["cos_lo"] for c in curve) - 0.05), 1.08)
    bx.grid(color=GRID, linewidth=0.7)
    bx.set_axisbelow(True)
    bx.set_xlabel("真实 API 调用次数（缓存命中与共享初稿不重复计费）")
    bx.set_ylabel("与精确梯度的余弦相似度")
    bx.set_title("(b) 同等调用预算下的梯度方向质量", fontsize=8.5)
    fig.text(0.085, 0.955, "方法验证：解析梯度精确，且全信息反馈避免采样式策略梯度的方差", fontsize=9.2,
             fontweight="bold", color=INK, ha="left")
    save(fig, "fig_gradient_method")


def synthetic_routing() -> dict:
    """The synthetic routing check from tests/test_policy.py, traced step by step."""
    rng = np.random.default_rng(3)
    train_x = rng.uniform(-2, 2, size=(180, 2))
    train_rewards = np.column_stack([train_x[:, 0] < 0, train_x[:, 0] >= 0,
                                     np.full(len(train_x), 0.2)]).astype(float)
    test_x = rng.uniform(-2, 2, size=(300, 2))  # drawn after the training points, as in the test
    correct = (test_x[:, 0] >= 0).astype(int)
    policy = LinearPolicy(2, 3, seed=7)
    objective, accuracy, confidence = [], [], []
    for _ in range(350):
        objective.extend(policy.fit(train_x, train_rewards, steps=1, lr=0.2))
        accuracy.append(float(np.mean(policy.predict(test_x) == correct)))
        confidence.append(float(np.mean(policy.probs(test_x)[np.arange(len(test_x)), correct])))
    return {"objective": objective, "accuracy": accuracy, "confidence": confidence, "policy": policy,
            "test_x": test_x, "correct": correct, "train_x": train_x}


def fig_synthetic(result: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.15), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.21, wspace=0.42)
    steps = np.arange(1, len(result["objective"]) + 1)
    ax = axes[0]
    ax.plot(steps, result["objective"], color=BLUE, linewidth=2)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("梯度上升步数")
    ax.set_title("(a) 训练目标 J(θ)", fontsize=8.5)
    ax.set_xlim(0, steps[-1])
    bx = axes[1]
    bx.plot(steps, result["confidence"], color=BLUE, linewidth=2)
    bx.grid(axis="y", color=GRID, linewidth=0.7)
    bx.set_axisbelow(True)
    bx.set_xlabel("梯度上升步数")
    bx.set_title("(b) 最优动作概率均值（未见输入）", fontsize=8.5)
    bx.set_xlim(0, steps[-1])
    bx.set_ylim(0.3, 1.0)
    final = result["confidence"][-1]
    bx.text(steps[-1], final - 0.04, f"{final:.2f}", ha="right", va="top", fontsize=7.6, color=INK2)
    cx = axes[2]
    grid = np.linspace(-2, 2, 161)
    gx, gy = np.meshgrid(grid, grid)
    choice = result["policy"].predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    tint = matplotlib.colors.ListedColormap(["#d4e4f7", "#fbdccf", "#d2f0e4"])
    cx.pcolormesh(gx, gy, choice, cmap=tint, vmin=-0.5, vmax=2.5, shading="auto", rasterized=True)
    pts, correct = result["test_x"], result["correct"]
    cx.scatter(pts[correct == 0, 0], pts[correct == 0, 1], s=5, color=BLUE, linewidths=0, label="最优动作 A")
    cx.scatter(pts[correct == 1, 0], pts[correct == 1, 1], s=5, color=ORANGE, linewidths=0, label="最优动作 B")
    cx.axvline(0, color=INK, linewidth=1.0)
    cx.set_xlim(-2, 2)
    cx.set_ylim(-2, 2)
    cx.set_xticks([-2, 0, 2])
    cx.set_yticks([-2, 0, 2])
    cx.set_title(f"(c) 决策区域（未见输入 {100 * result['accuracy'][-1]:.0f}% 正确）", fontsize=8.5)
    cx.set_xlabel("特征 x₁（真实边界 x₁ = 0）")
    legend = cx.legend(loc="upper left", ncol=1, fontsize=6.6, handletextpad=0.1, markerscale=1.8,
                       frameon=True, borderpad=0.3, borderaxespad=0.25)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.9)
    fig.text(0.075, 0.955, "合成路由任务（与单元测试相同的设定）：精确梯度学会依赖上下文的路由", fontsize=9.2,
             fontweight="bold", color=INK, ha="left")
    save(fig, "fig_synthetic")


# --------------------------------------------------------------------------- LaTeX

def tex_escape(text: str) -> str:
    return (str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$"))


def signed(value: float, digits: int) -> str:
    text = f"{value:+.{digits}f}"
    if float(text) == 0:
        text = f"{0:.{digits}f}"
    return f"${text}$"


def signed_text(value: float, digits: int) -> str:
    """Signed number for text (sans-serif) contexts, with a typographic minus."""
    text = f"{value:+.{digits}f}"
    if float(text) == 0:
        text = f"{0:.{digits}f}"
    return text.replace("-", "\u2212")


def sci_text(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}\\,\u00d7\\,10\\textsuperscript{{{int(exponent)}}}".replace("-", "\u2212")


def write_generated(data: dict, changes: list[dict], breakdown: dict, check: dict, reinforce: dict,
                    synthetic: dict) -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    summary = data["summary"]
    config = summary["config"]
    metrics = summary["metrics"]
    comps = summary["comparisons"]
    ledger = summary["ledger"]
    probe = data["probe"]
    n = config["test_size"]
    initial_tokens = metrics["initial_reason"]["mean_tokens"]
    oracle_extra = metrics["best_fixed"]["mean_tokens"] - metrics["oracle"]["mean_tokens"]
    saved = initial_tokens - metrics["learned"]["mean_tokens"]
    macros = {
        "numTestN": n, "numTrainN": config["train_size"], "numValN": config["val_size"],
        "numTotalTasks": config["train_size"] + config["val_size"] + config["test_size"],
        "numRounds": config["rounds"], "numSteps": config["steps"], "numSeed": config["seed"],
        "numMaxRequests": config["max_requests"], "numMaxTokens": config["max_tokens"],
        "numCostWeight": f"{config['cost_weight']:g}", "numLR": f"{config['lr']:g}",
        "numEntropy": f"{config['entropy']:g}", "numLtwo": f"{config['l2']:g}",
        "numGateTol": f"{config['gate_accuracy_tolerance']:g}",
        "numParams": len(FEATURE_NAMES) * len(STRATEGIES) + len(STRATEGIES), "numFeatures": len(FEATURE_NAMES),
        "numAttempts": ledger["attempts"], "numSuccesses": ledger["successes"], "numFailures": ledger["failures"],
        "numInputTokens": f"{ledger['input_tokens']:,}", "numOutputTokens": f"{ledger['output_tokens']:,}",
        "numTotalTokens": f"{ledger['total_tokens']:,}", "numProbeTokens": probe["total_tokens"],
        "numCombinedAttempts": probe["combined_attempts"],
        "numSelectedVersion": summary["selected_version"], "numBestFixed": summary["best_fixed_strategy"],
        "numPickDirect": summary["selected_actions"]["direct"], "numPickReason": summary["selected_actions"]["reason"],
        "numPickReview": summary["selected_actions"]["review"],
        "numReviewFixes": summary["review_effect"]["fixes"], "numReviewBreaks": summary["review_effect"]["breaks"],
        "numOracleExtraTokens": f"{oracle_extra:.2f}",
        "numOracleExtraPct": f"{100 * oracle_extra / metrics['best_fixed']['mean_tokens']:.2f}",
        "numAmortize": f"{math.ceil(ledger['total_tokens'] / saved):,}" if saved > 0 else "--",
        "numSavedPerTask": f"{saved:.2f}",
        "numValBestUtil": f"{summary['validation_best_fixed_metrics']['utility']:.4f}",
        "numValBestTokens": f"{summary['validation_best_fixed_metrics']['mean_tokens']:.2f}",
        "numGradMaxErr": f"{check['max_abs_error']:.1e}".replace("e-", r"\times10^{-") + "}",
        "numGradMaxErrTxt": sci_text(check["max_abs_error"]),
        "numGradParams": check["n_params"],
        "numReinforceCos": f"{reinforce['curve'][0]['cos_mean']:.2f}",
        "numReinforceCalls": f"{reinforce['curve'][0]['calls']:.0f}",
        "numExactCalls": reinforce["exact_calls"],
        "numSyntheticAcc": f"{100 * synthetic['accuracy'][-1]:.1f}",
        "numRejectedNoChange": sum(1 for c in changes if not c["accepted"] and c["val_changed"] == 0),
        "numAcceptedValChanged": next((c["val_changed"] for c in changes if c["accepted"]), 0),
        "numWallSeconds": f"{summary['wall_seconds']:.0f}",
        "numFormatFailReason": summary["format_failures"]["reason"],
        "numFormatFailReview": summary["format_failures"]["review"],
        "numFormatFailDirect": summary["format_failures"]["direct"],
    }
    for key, label in (("learned", "Learned"), ("best_fixed", "Best"), ("initial_reason", "Init"),
                       ("fixed_review", "Review"), ("uniform", "Uniform"), ("oracle", "Oracle")):
        m = metrics[key]
        macros[f"num{label}Acc"] = f"{100 * m['accuracy']:.2f}"
        macros[f"num{label}Correct"] = f"{m['accuracy'] * n:.0f}" if key != "uniform" else f"{m['accuracy'] * n:.2f}"
        macros[f"num{label}Tokens"] = f"{m['mean_tokens']:.2f}"
        macros[f"num{label}Util"] = f"{m['utility']:.4f}"
        macros[f"num{label}TokenChange"] = signed(100 * (m["mean_tokens"] / initial_tokens - 1), 2)
        macros[f"num{label}TokenChangeTxt"] = signed_text(100 * (m["mean_tokens"] / initial_tokens - 1), 2)
    for comp_key, label in (("vs_initial", "VsInit"), ("vs_best_fixed", "VsBest")):
        for metric, short, scale, digits in (("accuracy_difference", "Acc", 100, 2), ("token_difference", "Tok", 1, 2),
                                             ("utility_difference", "Util", 1, 5)):
            item = comps[comp_key][metric]
            macros[f"num{label}{short}"] = signed(item["point"] * scale, digits)
            macros[f"num{label}{short}CI"] = f"[{signed(item['ci95'][0] * scale, digits)}, {signed(item['ci95'][1] * scale, digits)}]"
    # Descriptive statistics of the recorded reward matrices (used in the discussion).
    for split, label in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        correct = field_matrix(data["rows"][split], "correct")
        tokens = field_matrix(data["rows"][split], "total_tokens")
        utility = (correct - config["cost_weight"] * tokens / config["token_unit"]).mean(axis=0)
        macros[f"num{label}DirectUtil"] = f"{utility[0]:.5f}"
        macros[f"num{label}ReasonUtil"] = f"{utility[1]:.5f}"
        macros[f"num{label}FlipToReason"] = int(np.sum((correct[:, 0] == 0) & (correct[:, 1] == 1)))
        macros[f"num{label}FlipToDirect"] = int(np.sum((correct[:, 1] == 0) & (correct[:, 0] == 1)))
        macros[f"num{label}AllWrong"] = int(np.sum(correct.max(axis=1) == 0))
        macros[f"num{label}DirectCheaperPct"] = f"{100 * np.mean(tokens[:, 0] < tokens[:, 1]):.1f}"
    convergence = load_json(REPO_ROOT / "artifacts" / "convergence_check" / "frozen_policy.json")
    macros["numConvUtil"] = f"{convergence['selected_validation']['utility']:.4f}"
    macros["numConvTokens"] = f"{convergence['selected_validation']['mean_tokens']:.2f}"
    macros["numConvDirect"] = convergence["selected_actions"]["direct"]
    macros["numConvSteps"] = convergence["steps_per_round"]
    # Unit-test inventory, counted from the test sources so the report cannot drift from the suite.
    test_counts = {
        path.stem: sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                       if line.lstrip().startswith("def test_"))
        for path in sorted((REPO_ROOT / "tests").glob("test_*.py"))
    }
    macros["numUnitTests"] = sum(test_counts.values())
    for stem, label in (("test_policy", "Policy"), ("test_client", "Client"), ("test_evaluation", "Evaluation"),
                        ("test_tasks", "Tasks"), ("test_experiment", "Experiment"), ("test_report", "Report")):
        macros[f"numTests{label}"] = test_counts.get(stem, 0)
    lines = ["% Auto-generated by scripts/make_figures.py from artifacts/gsm8k_flash_v1. Do not edit."]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (GEN_DIR / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # Main results table.
    order = [("learned", "梯度学习路由器（GFPG，冻结版本 %d）" % summary["selected_version"]),
             ("best_fixed", "验证集选出的最佳固定策略（direct）"),
             ("initial_reason", "初始策略 = 固定 reason"),
             ("fixed_review", "固定 review（分步 + 复核）"),
             ("uniform", "均匀随机（期望指标）"),
             ("oracle", r"逐题最优 Oracle$^{\dagger}$")]
    body = []
    for key, label in order:
        m = metrics[key]
        change = 100 * (m["mean_tokens"] / initial_tokens - 1)
        correct = f"{m['accuracy'] * n:.2f}" if key == "uniform" else f"{m['accuracy'] * n:.0f}"
        row = (f"{label} & {correct}/{n} & {100 * m['accuracy']:.2f}\\% & {m['mean_tokens']:.1f} & "
               f"{signed(change, 2) if key != 'initial_reason' else '--'} & {m['mean_calls']:.2f} & "
               f"{m['mean_latency_s']:.2f} & {m['utility']:.4f} \\\\")
        if key == "learned":
            row = r"\rowcolor{rowhighlight}" + row
        body.append(row)
        if key == "best_fixed":
            body.append(r"\midrule")
        if key == "uniform":
            body.append(r"\midrule")
    (GEN_DIR / "table_main.tex").write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")

    # Paired comparisons table.
    body = []
    for comp_key, label in (("vs_initial", "相对初始 reason"), ("vs_best_fixed", "相对最佳固定 direct")):
        item = comps[comp_key]
        cells = []
        for metric, scale, digits in (("accuracy_difference", 100, 2), ("token_difference", 1, 2), ("utility_difference", 1, 5)):
            point = item[metric]["point"] * scale
            lo, hi = (v * scale for v in item[metric]["ci95"])
            cells.append(f"{signed(point, digits)} & [{signed(lo, digits)}, {signed(hi, digits)}]")
        verdict_lo, verdict_hi = item["utility_difference"]["ci95"]
        verdict = "显著更优" if verdict_lo > 0 else ("显著更差" if verdict_hi < 0 else "无法区分")
        body.append(f"{label} & " + " & ".join(cells) + f" & {verdict} \\\\")
    (GEN_DIR / "table_paired.tex").write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")

    # Self-improvement rounds table.
    body = []
    for entry, change in zip(data["rounds"], changes):
        decision = r"\textbf{接纳}" if entry["accepted"] else "拒绝"
        body.append(
            f"{entry['round']} & {entry['train_tasks']} & {entry['objective_start']:.5f} & {entry['objective_end']:.5f} & "
            f"{entry['val_incumbent']['utility']:.5f} & {entry['val_candidate']['utility']:.5f} & "
            f"{change['val_changed']} & {decision} \\\\")
    (GEN_DIR / "table_rounds.tex").write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")

    # Request accounting table.
    split_names = {"val": "验证集采集", "train": "训练集采集（四轮）", "test": "冻结后测试"}
    body = [f"{split_names[k]} & {v} \\\\" for k, v in breakdown["by_split"].items()]
    body.append(r"\midrule")
    body += [f"失败：{tex_escape(k)} & {v} \\\\" for k, v in breakdown["by_status"].items()]
    body.append(r"\midrule")
    body.append(f"主实验尝试合计（上限 {breakdown['cap']}） & {breakdown['attempts']} \\\\")
    body.append(f"另：早期连通性探测（独立账本） & {probe['attempts']} \\\\")
    (GEN_DIR / "table_requests.tex").write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    setup_style()
    data = load_run()
    changes = round_decision_changes(data)
    breakdown = request_breakdown(data)
    check = gradient_check(data)
    reinforce = reinforce_vs_exact(data)
    synthetic = synthetic_routing()
    fig_main_results(data)
    fig_paired_ci(data)
    fig_rounds(data, changes)
    fig_tokens(data)
    fig_budget(data, breakdown)
    fig_gradient_method(check, reinforce)
    fig_synthetic(synthetic)
    write_generated(data, changes, breakdown, check, reinforce, synthetic)
    print(json.dumps({
        "figures": sorted(p.name for p in FIG_DIR.glob("*.pdf")),
        "round_decision_changes": changes,
        "gradient_check_max_abs_error": check["max_abs_error"],
        "reinforce_curve": reinforce["curve"],
        "synthetic_final_routing_accuracy": synthetic["accuracy"][-1],
        "requests": breakdown,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
