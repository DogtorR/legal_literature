<!-- BEAUTIFIED -->
<h1 align="center">Legal Literature Acquisition Agent</h1>
<p align="center">
  <strong>Legal open-access literature acquisition for metadata discovery, OA auditing, pre-download QA, and guarded full-text collection.</strong>
  <br />
  <em>CRISPR corpus workflows · MCP server · CLI shell · OpenAI-compatible LLM providers</em>
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
  English · <a href="README-zh.md">中文</a>
</p>

This repository contains one agent package: `legal_literature_acquisition_agent_core_v1_latest`.

It searches public scholarly sources, merges and deduplicates records, applies scope filters, audits open-access evidence, generates pre-download review files, and collects legal open full text only when explicitly allowed.

It does not bypass paywalls, login, institutional accounts, cookies, sessions, CAPTCHA, VPN or proxy tricks, Sci-Hub, LibGen, Z-Library, or unauthorized mirrors. If legal OA evidence cannot be confirmed, the record stays in metadata-only, manual-review, or failure outputs instead of being downloaded.

## Features

| Feature | Description |
|---|---|
| Natural-Language Entry | Accepts literature acquisition requests through `run_literature_acquisition_agent`, the interactive shell, and MCP tools. |
| Public Source Search | Searches OpenAlex, PubMed, Crossref, and Europe PMC through a common tool registry. |
| OA Audit and QA | Audits OA evidence and writes approval, rejection, and manual-review files before any full-text collection step. |
| Guarded Downloads | Requires both `allow_download=True` and `yes=True` before collecting approved full texts. |
| Runtime Artifacts | Writes manifests, search logs, failure logs, and Markdown reports that help with debugging and auditability. |
| CRISPR-First Scope | Ships with stable v1 rule-based support for CRISPR and Cas-related collection workflows. |

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
    request="Collect legal open-access CRISPR literature across all years. Keep metadata if full text cannot be obtained.",
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

### Interactive Shell

```powershell
python -m lit_agent.acquisition_chat
```

### MCP Server

```powershell
lit-agent-mcp --transport stdio
```

### Approved Full-Text Collection

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

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | API key for the OpenAI-compatible OpenAI provider. | - |
| `DEEPSEEK_API_KEY` | API key for the DeepSeek provider. | - |
| `DASHSCOPE_API_KEY` | API key for the DashScope or Qwen provider. | - |

### LLM Providers

| Key | Description | Default |
|---|---|---|
| `active_provider` | Provider selected by default in `llm_config.yaml`. | `deepseek` |
| `agent.language` | Default report language. | `zh` |
| `agent.default_year_from` | Lower bound used by default task settings. | `2024` |
| `agent.default_year_to` | Upper bound used by default task settings. | `2025` |
| `agent.max_results` | Default metadata result cap. | `20` |
| `agent.max_downloads` | Default full-text collection cap. | `20` |

### Runtime Artifacts

These files are useful during development, debugging, and audit review, but they are not source files and they are intentionally ignored by `.gitignore`.

| File | Purpose |
|---|---|
| `metadata_results.jsonl` | Raw or merged metadata collected by the older pipeline path. |
| `candidates.jsonl` | Candidate records selected for the next stage. |
| `legality_audit.jsonl` | OA and legality audit results per record. |
| `download_plan.jsonl` | Planned download actions created after audit. |
| `manifest.jsonl` | Final or intermediate record manifest. |
| `failures.jsonl` | Failed steps and recorded reasons. |
| `search_log.jsonl` | Search-stage execution log. |
| `agent_runs/reports/report_zh.md` | Chinese runtime report. |
| `agent_runs/reports/report_en.md` | English runtime report. |

## Project Structure

```text
lit_agent/
|- acquisition_chat.py      # Interactive natural-language shell
|- acquisition_request.py   # Request parsing and query profile construction
|- agent_graph.py           # LangGraph orchestration and high-level flows
|- mcp_server.py            # MCP server and Python API entry points
|- tool_registry.py         # Standardized tool layer for search, QA, and download actions
|- sources/                 # Source adapters for public literature services

examples/
|- search_config.yaml       # Example search configuration
|- queries.csv              # Example query list
|- mock_metadata.jsonl      # Example metadata fixture

tests/
|- test_*.py                # Regression and safety coverage
```

## Tech Stack

### Core Runtime

| Technology | Purpose |
|---|---|
| Python 3.10+ | Primary runtime |
| setuptools | Packaging and editable installs |
| PyYAML | YAML configuration loading |

### Agent and Interfaces

| Technology | Purpose |
|---|---|
| LangGraph | Agent orchestration |
| MCP Python SDK | MCP server transport and tool exposure |
| argparse | Command-line interfaces |

### Search and Evidence Sources

| Technology | Purpose |
|---|---|
| OpenAlex | Scholarly metadata search |
| PubMed | Biomedical metadata search |
| Crossref | DOI and metadata search |
| Europe PMC | Metadata and OA evidence search |
| Unpaywall | OA evidence lookup |

### Testing

| Technology | Purpose |
|---|---|
| pytest | Test suite |

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit the change with a clear message.
4. Push the branch and open a pull request.

## License

No LICENSE file detected. Add a LICENSE to clarify project licensing.
