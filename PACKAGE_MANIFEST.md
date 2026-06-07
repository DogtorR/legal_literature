# Package Manifest

This package contains the core source code and documentation for the Legal Literature Acquisition Agent.

Included:
- lit_agent/: Agent source code
- tests/: regression and safety tests
- examples/: small example configs/mock metadata
- README.md: English project README
- README-zh.md: Chinese project README
- llm_config.yaml: OpenAI-compatible provider configuration template; API keys are read from environment variables
- .env.example: environment variable example
- pyproject.toml / requirements.txt: Python packaging and dependency metadata

Excluded intentionally:
- .env and API keys
- agent_runs/ large runtime outputs
- downloads/ and fulltexts
- root JSONL runtime logs
- .pytest_cache, __pycache__, egg-info
- previous dist packages

Main entry points:
- Natural language shell: python -m lit_agent.acquisition_chat
- Python API: from lit_agent.mcp_server import run_literature_acquisition_agent
- Count summary: ask natural language count/statistics requests through run_literature_acquisition_agent

Safety boundary:
The agent only collects legal open metadata/fulltext and does not bypass paywalls, login, cookies, CAPTCHA, VPN, sessions, or use Sci-Hub/LibGen/Z-Library.
