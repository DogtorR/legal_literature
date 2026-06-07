<!-- BEAUTIFIED -->
<h1 align="center">Legal Literature Acquisition Agent</h1>
<p align="center">
  <strong>面向合法开放获取文献的 metadata 检索、OA 审计、下载前 QA 与受控全文采集。</strong>
  <br />
  <em>CRISPR 文献工作流 · MCP server · CLI shell · OpenAI-compatible LLM provider</em>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-2E8B57?style=for-the-badge" alt="Quick Start" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/MCP-Server-0A0A0A?style=flat" alt="MCP Server" />
  <img src="https://img.shields.io/badge/LangGraph-Agent-1C7ED6?style=flat" alt="LangGraph Agent" />
  <img src="https://img.shields.io/badge/OpenAlex%20%7C%20PubMed%20%7C%20Crossref-Search-4C6EF5?style=flat" alt="Search Sources" />
</p>

<p align="center">
  <a href="README.md">English</a> · 中文
</p>

当前仓库包含一个 Agent 包：`legal_literature_acquisition_agent_core_v1_latest`。

它会从公开学术数据源中检索文献、合并去重、执行范围过滤、审计开放获取证据、生成下载前审核文件，并且只在明确允许时采集合规开放全文。

它不会绕过付费墙、登录、机构账号、cookie、session、CAPTCHA、VPN 或 proxy trick，也不会使用 Sci-Hub、LibGen、Z-Library 或未授权镜像。无法确认合法 OA 证据的记录会留在 metadata-only、manual-review 或 failure 结果中，而不是被自动下载。

## Features

| 功能 | 说明 |
|---|---|
| 自然语言入口 | 通过 `run_literature_acquisition_agent`、交互式 shell 和 MCP tools 接收采集请求。 |
| 公开数据源检索 | 通过统一工具层检索 OpenAlex、PubMed、Crossref 和 Europe PMC。 |
| OA 审计与 QA | 在任何全文采集动作之前先生成批准、拒绝和人工复核文件。 |
| 受控下载门 | 只有同时满足 `allow_download=True` 和 `yes=True` 才会采集批准项全文。 |
| 运行产物 | 写出 manifest、search log、failure log 和 Markdown 报告，便于调试和审计。 |
| CRISPR 优先范围 | 当前稳定的 v1 规则 profile 主要面向 CRISPR 和 Cas 相关采集工作流。 |

## Quick Start

### Prerequisites

```powershell
python --version
```

### Install

```powershell
python -m pip install -e ".[dev]"
```

### Configure

```powershell
Copy-Item .env.example .env
```

### Run

```powershell
python -m lit_agent.acquisition_chat
```

## Usage

### Python API

```python
from lit_agent.mcp_server import run_literature_acquisition_agent

result = run_literature_acquisition_agent(
    request="抓取所有年份 CRISPR 相关合法开放全文，拿不到全文时保留 metadata。",
    allow_download=False,
    yes=False,
    output_dir="agent_runs/crispr_broad_all_years",
    max_rounds=1,
    max_batches_per_round=4,
    max_results_per_batch=20,
    max_additional_results_per_batch=20,
)

print(result)
```

### 交互式 Shell

```powershell
python -m lit_agent.acquisition_chat
```

### MCP Server

```powershell
lit-agent-mcp --transport stdio
```

### 批准项全文采集

```python
from lit_agent.mcp_server import collect_approved_legal_fulltexts

result = collect_approved_legal_fulltexts(
    output_dir="agent_runs/crispr_detection_2020_2024",
    approved_file="approved_for_download.jsonl",
    allow_download=True,
    yes=True,
    max_items=10,
    prefer_formats=[
        "pdf",
        "pmc_xml",
        "europe_pmc_xml",
        "publisher_html",
        "repository_file",
        "preprint_file",
    ],
)

print(result)
```

## Configuration

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI-compatible OpenAI provider 的 API key。 | - |
| `DEEPSEEK_API_KEY` | DeepSeek provider 的 API key。 | - |
| `DASHSCOPE_API_KEY` | DashScope 或 Qwen provider 的 API key。 | - |

### LLM Provider 设置

| 键 | 说明 | 默认值 |
|---|---|---|
| `active_provider` | `llm_config.yaml` 中默认启用的 provider。 | `deepseek` |
| `agent.language` | 默认报告语言。 | `zh` |
| `agent.default_year_from` | 默认任务参数中的起始年份。 | `2024` |
| `agent.default_year_to` | 默认任务参数中的结束年份。 | `2025` |
| `agent.max_results` | 默认 metadata 结果上限。 | `20` |
| `agent.max_downloads` | 默认全文采集上限。 | `20` |

### 运行产物

这些文件对开发、调试和审计很有用，但它们不是源码文件，所以默认被 `.gitignore` 忽略。

| 文件 | 用途 |
|---|---|
| `metadata_results.jsonl` | 旧 pipeline 路径写出的原始或合并 metadata。 |
| `candidates.jsonl` | 进入下一阶段的候选记录。 |
| `legality_audit.jsonl` | 每条记录的 OA 和合法性审计结果。 |
| `download_plan.jsonl` | 审计后生成的下载计划。 |
| `manifest.jsonl` | 最终或中间结果清单。 |
| `failures.jsonl` | 失败步骤及原因。 |
| `search_log.jsonl` | 检索阶段执行日志。 |
| `agent_runs/reports/report_zh.md` | 中文运行报告。 |
| `agent_runs/reports/report_en.md` | 英文运行报告。 |

## Project Structure

```text
lit_agent/
|- acquisition_chat.py      # 交互式自然语言 shell
|- acquisition_request.py   # 请求解析与 query profile 构建
|- agent_graph.py           # LangGraph 编排与高层流程
|- mcp_server.py            # MCP server 与 Python API 入口
|- tool_registry.py         # 搜索、QA、下载动作的统一工具层
|- sources/                 # 公开文献数据源适配器

examples/
|- search_config.yaml       # 示例搜索配置
|- queries.csv              # 示例查询列表
|- mock_metadata.jsonl      # 示例 metadata fixture

tests/
|- test_*.py                # 回归与安全测试
```

## Tech Stack

### 核心运行时

| 技术 | 用途 |
|---|---|
| Python 3.10+ | 主运行时 |
| setuptools | 打包与 editable install |
| PyYAML | YAML 配置读取 |

### Agent 与接口

| 技术 | 用途 |
|---|---|
| LangGraph | Agent 编排 |
| MCP Python SDK | MCP server transport 与 tool 暴露 |
| argparse | 命令行接口 |

### 检索与证据来源

| 技术 | 用途 |
|---|---|
| OpenAlex | 学术 metadata 检索 |
| PubMed | 生物医学 metadata 检索 |
| Crossref | DOI 与 metadata 检索 |
| Europe PMC | metadata 与 OA 证据检索 |
| Unpaywall | OA 证据查询 |

### 测试

| 技术 | 用途 |
|---|---|
| pytest | 测试套件 |

## Contributing

1. Fork 仓库。
2. 创建功能分支。
3. 用清晰的提交信息提交修改。
4. 推送分支并发起 pull request。

## License

No LICENSE file detected. Add a LICENSE to clarify project licensing.
