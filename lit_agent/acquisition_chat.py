"""Interactive natural-language shell for the literature acquisition agent."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from .agent_graph import run_literature_acquisition_agent


HELP = """
Natural-language literature acquisition shell

Type a request directly, for example:
  帮我看下有多少 CRISPR 的文献，OpenAlex 上有多少，PubMed 上有多少，我本地有多少。
  抓取所有年份 CRISPR 相关合法开放全文，全文格式不限，拿不到全文保留 metadata。
  抓取 2020-2024 年 CRISPR detection 相关文献，合法全文能拿就拿。
  抓取所有年份关于二维材料传感器的合法开放全文。

Commands:
  :help                 show this help
  :exit                 quit
  :out PATH             set output_dir
  :rounds N             set max_rounds
  :batches N            set max_batches_per_round
  :results N            set max_results_per_batch
  :download on|off      allow fulltext acquisition gate 1
  :yes on|off           confirm fulltext acquisition gate 2
  :status               show current settings
  :open                 open the latest output_dir in Windows Explorer

Safety:
  Fulltext collection only runs when both :download on and :yes on are set.
  The agent never bypasses paywalls, login, cookies, CAPTCHA, VPN, Sci-Hub,
  LibGen, Z-Library, or other unauthorized access paths.
"""


def _bool_arg(value: str) -> bool:
    return value.strip().lower() in {"on", "true", "1", "yes", "y"}


def _print_summary(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))
    artifacts = result.get("artifacts") or {}
    if artifacts:
        print("\nArtifacts:")
        for key, value in artifacts.items():
            print(f"  {key}: {value}")
    parsed = result.get("parsed_request") or {}
    collection = result.get("collection") or {}
    local = result.get("local") or {}
    output_dir = (
        parsed.get("output_dir")
        or collection.get("artifacts", {}).get("batch_manifest", "")
        or local.get("output_dir")
        or ""
    )
    if output_dir and output_dir.endswith("batch_manifest.jsonl"):
        from pathlib import Path

        output_dir = str(Path(output_dir).parent)
    print("\nNext step:")
    if result.get("intent") == "count_summary":
        print("  这是数量统计结果，不需要打开全文目录。")
    elif result.get("status") == "needs_clarification":
        print("  这个主题太宽，先根据 clarification_question 选择更具体的方向。")
    elif output_dir:
        print(f"  请打开输出目录查看结果: {output_dir}")
        print("  建议先看:")
        print("    - corpus_manifest.jsonl")
        print("    - metadata_only_manifest.jsonl")
        print("    - approved_for_download.jsonl")
        print("    - dataset_card.md")
        print("    - full_collection_report.md")
        if result.get("actual_fulltexts"):
            print("    - collected_fulltexts_manifest.jsonl")
            print("    - fulltexts/")
    else:
        print("  没有检测到输出目录；请查看上面的 artifacts 字段。")


def main() -> int:
    settings: dict[str, Any] = {
        "allow_download": False,
        "yes": False,
        "output_dir": None,
        "max_rounds": 3,
        "max_batches_per_round": 6,
        "max_results_per_batch": 100,
        "max_additional_results_per_batch": 100,
        "time_budget_seconds": 1800,
        "graceful_stop_buffer_seconds": 120,
    }
    last_output_dir = ""
    print(HELP.strip())
    while True:
        try:
            text = input("\nAgent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not text:
            continue
        if text in {":exit", "exit", "quit", ":quit"}:
            print("bye")
            return 0
        if text == ":help":
            print(HELP.strip())
            continue
        if text == ":status":
            print(json.dumps(settings, ensure_ascii=False, indent=2))
            continue
        if text == ":open":
            target = last_output_dir or settings.get("output_dir")
            if not target:
                print("还没有可打开的输出目录。先运行一个请求，或用 :out PATH 设置输出目录。")
                continue
            path = Path(str(target)).resolve()
            if not path.exists():
                print(f"输出目录还不存在: {path}")
                continue
            subprocess.Popen(["explorer", str(path)])
            print(f"已打开: {path}")
            continue
        if text.startswith(":out "):
            settings["output_dir"] = text[5:].strip() or None
            print(f"output_dir = {settings['output_dir']}")
            continue
        if text.startswith(":rounds "):
            settings["max_rounds"] = int(text.split(maxsplit=1)[1])
            print(f"max_rounds = {settings['max_rounds']}")
            continue
        if text.startswith(":batches "):
            settings["max_batches_per_round"] = int(text.split(maxsplit=1)[1])
            print(f"max_batches_per_round = {settings['max_batches_per_round']}")
            continue
        if text.startswith(":results "):
            value = int(text.split(maxsplit=1)[1])
            settings["max_results_per_batch"] = value
            settings["max_additional_results_per_batch"] = value
            print(f"max_results_per_batch = {value}")
            continue
        if text.startswith(":download "):
            settings["allow_download"] = _bool_arg(text.split(maxsplit=1)[1])
            print(f"allow_download = {settings['allow_download']}")
            continue
        if text.startswith(":yes "):
            settings["yes"] = _bool_arg(text.split(maxsplit=1)[1])
            print(f"yes = {settings['yes']}")
            continue
        try:
            result = run_literature_acquisition_agent(request=text, **settings)
            _print_summary(result)
            parsed = result.get("parsed_request") or {}
            collection = result.get("collection") or {}
            artifact_dir = collection.get("artifacts", {}).get("batch_manifest", "")
            if artifact_dir:
                last_output_dir = str(Path(artifact_dir).parent)
            elif parsed.get("output_dir"):
                last_output_dir = str(parsed["output_dir"])
            elif (result.get("local") or {}).get("output_dir"):
                last_output_dir = str((result.get("local") or {})["output_dir"])
        except Exception as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
