# HMAS：基于动态任务分解与自适应验证的分层多智能体协同框架

本项目是一个分层多智能体协同框架，重点研究动态任务分解、自适应验证与错误恢复机制。通过多智能体协作完成复杂任务，并评估系统的任务成功率、执行成本与可靠性。

论文实验与毕设系统共用的核心代码。当前是 W1 版本：工程骨架 + 单智能体基线 + 评测流水线。

## 目录

```
configs/models.yaml      模型端点（DeepSeek / 通义千问 / 本地 Ollama），只写环境变量名，不写密钥
hmas/llm.py              OpenAI 兼容客户端：响应缓存（重跑免费、可复现）、token 与调用次数统计、重试
hmas/sandbox.py          Docker 沙箱：断网、只读文件系统、CPU/内存/进程数限制、超时强杀
hmas/agents/single.py    单智能体基线 DirectAgent（一次调用，无规划、无验证）
hmas/bench/run.py        在 HumanEval+ / MBPP+ 上生成答案（可断点续跑）
hmas/bench/evaluate.py   在沙箱里用官方 EvalPlus 评测器打分，输出 pass@1、调用次数、token、成本
docker/                  沙箱镜像（python:3.11-slim + evalplus 0.3.1 + 一处上游修复的回移）
tests/                   单元测试与端到端管道测试
results/                 每次实验一个目录：records.jsonl / samples.jsonl / meta.json / summary.json
```

## 快速开始

```bash
pip install -e ".[dev]"
python scripts/download_data.py          # 下载 HumanEval+ v0.1.10、MBPP+ v0.2.0
scripts/build_sandbox.sh                 # 构建沙箱镜像 hmas-sandbox:0.2
cp .env.example .env                     # 填入 DEEPSEEK_API_KEY（.env 不进 git）
python -m pytest                         # 16 个测试

# 先在 20 题子集上试跑，确认无误再跑全量
python -m hmas.bench.run --dataset humaneval --model deepseek-flash --limit 20
python -m hmas.bench.evaluate results/humaneval_direct_deepseek-flash_n20

# 全量基线
python -m hmas.bench.run --dataset humaneval --model deepseek-flash
python -m hmas.bench.evaluate results/humaneval_direct_deepseek-flash
```

## 评测口径（写论文时可直接引用）

- 数据集：EvalPlus 的 HumanEval+（164 题，v0.1.10）与 MBPP+（378 题，v0.2.0）。
- 打分：官方 `evalplus.sanitize` + `evalplus.evaluate`，在断网 Docker 容器中执行。报告 base（原始测试）与 plus（加强测试）两个 pass@1。
- 单智能体基线的提示词与 EvalPlus 官方 OpenAI 接口提示词逐字一致，温度 0（贪心），分数可以和 EvalPlus 排行榜对照。
- 模型：DeepSeek-V4.1-Flash-1017（接口模型名 `deepseek-flash`，2026-10-19 测试；接口返回的 model 字段不含版本号，论文中按官方文档注明版本和测试日期）。该接口默认开启思考模式，所以每组实验都在 `configs/models.yaml` 里显式设置 `thinking.type`：`deepseek-flash` 为关闭，`deepseek-flash-thinking` 为开启。
- 每题记录 LLM 调用次数、输入/输出 token、耗时，用于论文的成本/效率表。
- 子集运行（`--limit`）只在所选题目上计分。

## 流水线自检（用标准答案，不是模型分数）

| 数据集 | 标准答案通过 | 说明 |
|---|---|---|
| HumanEval+ | 164/164（base 与 plus 均 100%） | 需要 `docker/patches/fix_find_zero.py`：evalplus 0.3.1 对 HumanEval/32 的特殊判题少记了通过，任何答案都会被判错；上游 master 已修，PyPI 还没发版，这里回移同样的三行 |
| MBPP+ | base 377/378，plus 376/378 | Mbpp/630：标准答案的辅助函数是生成器（没有 `return`），被 EvalPlus 的 sanitize 删掉了；Mbpp/255：plus 测试输入规模极大，在资源限制下判失败。两者都是 EvalPlus 自身行为，对所有方法一视同仁，保持官方口径不改 |

注意：sanitize 会删掉"没有 return 语句"的顶层函数（例如生成器辅助函数）。这是 EvalPlus 官方行为，多智能体方法生成代码时也受同样约束。

## 下一步（W2）

在 `hmas/agents/` 下实现 Planner / Coder / Verifier 三层框架，复用同一个 `LLMClient`（统计成本）和 `DockerSandbox`（Verifier 的 L2/L3 验证在沙箱中执行）。

## 许可证

GPL-3.0，见 [LICENSE](LICENSE)。
