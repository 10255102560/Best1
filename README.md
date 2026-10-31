# Gradient Harness RSI

项目代码采用 [MIT 许可证](LICENSE)；GSM8K 数据保留其[原始许可证](data/GSM8K_LICENSE.txt)与[来源说明](data/SOURCE.md)。

无需 GPU、固定基础模型，用梯度更新一个 48 参数的执行流程调度器。项目解决的问题是：简单题是否值得详细推理？已经有一个答案时，再检查一次是否值得付出额外成本？

这是可运行的工程与小规模真实 API 实验。**它属于 harness 参数层面的受限自改进，不训练基础模型，也不声称实现了开放式递归智能提升。**

真实实验已完成：冻结梯度路由在 128 道独立测试题上保持 **123/128 正确**，平均 token 减少 **1.87%**；但验证集选出的固定直接回答达到 **124/128 正确、token 减少 25.28%**，优于复杂路由。完整的正负结果都保留在 [测评报告](artifacts/gsm8k_flash_v1/REPORT.md) 和 [结果讨论](docs/findings.md)。

## 交付物

- `docs/research_plan.md`：问题筛选、数学推导、参考工程、实验设计和几天内的实施计划。
- `artifacts/gsm8k_flash_v1/REPORT.md`：已完成的真实 GSM8K 子集测评报告。
- `artifacts/gsm8k_flash_v1/summary.json`：可核查的指标、置信区间和请求成本。
- `artifacts/gsm8k_flash_v1/observations/`：逐题、逐策略输出和评分。
- `artifacts/gsm8k_flash_v1/rounds.json`、`versions/`、`frozen_policy.json`：自改进过程与最终冻结版本。
- `scripts/replay_report.py`：不调用模型，离线复核已有实验数据并重新生成报告。

## 方法

固定模型，定义三个执行动作：

| 动作 | 行为 | 部署请求数 |
|---|---|---:|
| `direct` | 请求只给最终答案 | 1 |
| `reason` | 请求简短分步计算 | 1 |
| `review` | 先分步计算，再检查并修订 | 2 |

调度器只读取题面，提取长度、数字数量、运算关键词等 15 个特征。三个动作对应 45 个线性权重和 3 个偏置。模型内部可能仍产生隐藏推理 token；直接回答不等于禁用了模型内部推理。

\[
\pi_\theta(a\mid x)=\mathrm{softmax}(Wx+b)_a,\qquad
r_{ia}=\mathbf 1[\text{答案正确}]-0.05\,\frac{\text{总 token}_{ia}}{1000}.
\]

用各动作实际输出构造完整反馈矩阵，梯度上升最大化期望奖励，加上熵正则和 L2 正则。这里使用数值参数的**精确期望梯度**，不是把自然语言反馈称作梯度，也不是采样 REINFORCE。部署与验证统一使用 `argmax`。

每轮收集一批新的训练题反馈 → 在累计训练反馈上更新参数 → 在验证集检验候选 → 通过门禁则接纳，否则保留当前版本。共四轮；测试前冻结最终版本与最佳固定基线。三个动作会全部探索，因此这是全信息反馈下的增量自适应路由，训练探索费用会单独报告。

`review` 复用 `reason` 的同一个初稿，实际数据采集每题只需 3 次请求；但报告中 `review` 的部署成本始终累计两次调用，不因缓存而记成一次。

## 本地运行

需要 Python 3.10+、NumPy 和系统 `curl`。Windows 自带的 `curl.exe` 可用；所有网络请求保留 TLS 证书校验。

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m unittest discover -s tests -v
```

先在本机设置以下环境变量。不要将真实密钥写进 Git 文件：

```powershell
$env:HARNESS_BASE_URL = 'https://your-provider.example/v1'
$env:HARNESS_MODEL = 'your-model-name'
$env:HARNESS_API_KEY = '<your-local-key>'
python -m harness_rsi.experiment --output artifacts/my_run --max-requests 980 --workers 4
```

支持 `/v1/chat/completions` 格式的接口。本次实验使用用户指定的 `deepseek-v4.1-flash`。也可通过 `--credentials .secrets/api.json` 读取本机被 Git 忽略的配置。

默认 128 训练题、48 验证题、128 测试题，共 304 题；正常采集 912 次请求。980 是包含失败重试的硬上限，另有一次早期连通性探测。API 没有提供价格配置，因此项目报告实测 token，不把 token 等同于人民币或美元。

中断后重复同一命令即可复用缓存和逐题结果。请求账本持久化，重启不会重置预算；同一个输出目录只允许运行一个进程。参数、题库或实现不一致时会拒绝继续，避免悄悄混用实验版本。

小规模试验可改为：

```powershell
python -m harness_rsi.experiment --output artifacts/pilot --train-size 40 --val-size 20 --test-size 40 --max-requests 350 --workers 4
```

## 离线复核已有结果

```powershell
python scripts/replay_report.py --run artifacts/gsm8k_flash_v1
```

此命令不使用密钥，不产生模型调用；根据逐题数据、冻结策略和预定指标重新计算结果。原始题库可重新下载；抽中的题目与划分已保存在 `split_manifest.json`。

## 结果应如何解释

主要比较包括原始 `reason`、三个固定流程、验证集选择的最佳固定流程、学得的路由，以及均匀随机路由的期望值。`oracle` 使用测试标签选动作，只是不可部署的上界。

在独立测试题上报告正确率、总 token、调用数和成本惩罚效用，并按题配对 bootstrap 得到 95% 区间。如果相对最佳固定流程的效用差区间跨过零，应写“没有明确优势证据”；不能只挑对自己有利的基线。

重复使用同一个验证集可能过拟合；本项目固定四轮，不进行大规模超参数搜索。128 道测试题只能支持一次小规模实验结论。GSM8K 是公开题库，不能排除基础模型预训练见过题目，因此不宣称测量了全新的数学泛化能力。

数值梯度、评分、预算与缓存等关键行为有自动化检查；`tests` 中模拟响应只用于软件验证，不会混入真实 API 实验报告。

## 项目结构

```text
harness_rsi/
  client.py       # 缓存、限速、预算、真实 token 记账
  tasks.py        # GSM8K 划分和精确数字评分
  strategies.py   # 三个冻结执行流程
  policy.py       # 精确梯度与 48 参数调度器
  experiment.py   # 四轮自改进、验证门禁、冻结测试
  evaluation.py   # 指标与配对 bootstrap
  report.py       # 中文报告与静态图
scripts/          # 数据下载、离线复核
docs/             # 研究计划和来源核验
tests/            # 关键数学和工程检查
artifacts/        # 真实实验数据与报告
```

## 用冻结策略回答新题

实验生成 `frozen_policy.json` 后，可对新题应用已选中的路由器；此命令会调用真实 API，模型与接口地址必须匹配冻结配置。可以沿用 `HARNESS_*` 环境变量，或指定本地凭据文件：

```powershell
python scripts/solve.py --question "A box contains 12 pencils. How many pencils are in 7 boxes?" --run artifacts/gsm8k_flash_v1 --credentials .secrets/api.json
```

JSON 输出包含路线、回答及完整路线的 token 和调用数；`review` 始终累计两步成本。新推理使用 `artifacts/inference/gsm8k_flash_v1/` 下独立缓存和账本，默认最多累计尝试 2 次请求，重启不重置，缓存命中不消耗新请求。需要继续处理新题时，主动用 `--max-requests 10` 等参数提高该账本的累计上限；`new_request_attempts` 与 `new_successful_request_tokens` 表示本次新增请求及成功请求 token，不包含缓存已有成本。
