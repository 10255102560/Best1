import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from harness_rsi.report import MISSING, build_markdown, write_report

RUN = Path(__file__).resolve().parents[1] / "artifacts" / "gsm8k_flash_v1"


def _summary() -> dict:
    return json.loads((RUN / "summary.json").read_text(encoding="utf-8"))


class ReportTests(unittest.TestCase):
    def test_report_states_recorded_numbers_and_both_comparisons(self):
        text = build_markdown(_summary())
        for expected in ("123/128", "124/128", "322.00", "245.20", "-1.87%", "-25.28%", "0.9448", "0.9565",
                         "-6.14 [-11.47, -1.83]", "-0.01165 [-0.02879, -0.00284]", "显著更优", "显著更差",
                         "51,431", "direct=7，reason=121，review=0"):
            self.assertIn(expected, text)
        # The learned router must always be compared with the validation-selected fixed strategy.
        self.assertIn("当前路由器劣于最佳固定策略", text)
        self.assertIn("验证效用提升且准确率守卫通过，接纳候选", text)

    def test_written_files_use_lf_line_endings_and_svg_is_well_formed(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = write_report(_summary(), folder)
            for path in paths.values():
                self.assertNotIn(b"\r", Path(path).read_bytes())
            root = ET.parse(paths["chart"]).getroot()
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            texts = " ".join(element.text or "" for element in root.iter("{http://www.w3.org/2000/svg}text"))
            for expected in ("梯度学习路由器（GFPG）", "最佳固定策略：direct", "0.9448", "0.9565", "763.0"):
                self.assertIn(expected, texts)
            # Rows are not duplicated when the best fixed strategy is also listed as a fixed baseline.
            self.assertEqual(texts.count("固定策略：direct"), 1)

    def test_utility_verdict_follows_the_paired_interval(self):
        cases = [((0.001, 0.004), "支持在本测试分布和成本权重下取得效用提升"),
                 ((-0.004, -0.001), "当前路由器劣于最佳固定策略"),
                 ((-0.002, 0.003), "尚不足以证明路由器取得稳定收益")]
        for interval, expected in cases:
            summary = _summary()
            summary["comparisons"]["vs_best_fixed"]["utility_difference"] = {
                "point": sum(interval) / 2, "ci95": list(interval)}
            self.assertIn(expected, build_markdown(summary))

    def test_partial_summary_degrades_without_raising(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = write_report({"run_name": "partial <run> & co"}, folder)
            text = Path(paths["report"]).read_text(encoding="utf-8")
            self.assertIn(MISSING, text)
            self.assertIn("无法估算部署摊销门槛", text)
            ET.parse(paths["chart"])  # escaped run name keeps the SVG well-formed

    def test_markdown_table_cells_are_escaped(self):
        summary = copy.deepcopy(_summary())
        summary["best_fixed_strategy"] = "a|b"
        text = build_markdown(summary)
        self.assertIn("a\\|b", text)
        self.assertNotIn("（a|b）", text)


if __name__ == "__main__":
    unittest.main()
