from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from multi_slm import SLMTaskProfile, get_task_profile, resolve_slm_model_path


ROOT = Path(__file__).resolve().parent
DEFAULT_QWEN_GGUF = ROOT / "models" / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"


def find_llama_cli() -> Path:
    env = os.environ.get("LLAMA_CLI_PATH")
    if env and Path(env).is_file():
        return Path(env)
    winget = (
        Path.home()
        / "AppData"
        / "Local"
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "llama-cli.exe"
    )
    if winget.is_file():
        return winget
    raise FileNotFoundError("llama-cli.exe not found. Install llama.cpp or set LLAMA_CLI_PATH.")


@dataclass
class QwenCliGeneration:
    answer: str
    prompt: str
    elapsed_s: float
    model_path: str
    model_file_bytes: int
    backend: str
    memory: Dict[str, float]
    task_id: str = "academic_qa"


class QwenGgufCliGenerator:
    """Task-specialist GGUF generator using standalone llama.cpp CLI.

    Existing callers may still pass a concrete Qwen GGUF path. New callers
    should prefer ``for_task(...)`` so each workload can use its own SLM file.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        llama_cli_path: str | Path | None = None,
        ctx_size: int | None = None,
        max_tokens: int | None = None,
        threads: int | None = None,
        task_id: str = "academic_qa",
        task_profile: SLMTaskProfile | None = None,
    ) -> None:
        self.task_profile = task_profile or get_task_profile(task_id)
        self.task_id = self.task_profile.task_id
        self.model_path = resolve_slm_model_path(self.task_id, model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Qwen GGUF not found: {self.model_path}")
        self.llama_cli_path = Path(llama_cli_path) if llama_cli_path else find_llama_cli()
        self.ctx_size = int(ctx_size if ctx_size is not None else self.task_profile.ctx_size)
        self.max_tokens = int(max_tokens if max_tokens is not None else self.task_profile.max_tokens)
        self.threads = int(threads if threads is not None else self.task_profile.threads)

    @classmethod
    def for_task(
        cls,
        task_id: str,
        *,
        model_path: str | Path | None = None,
        llama_cli_path: str | Path | None = None,
        ctx_size: int | None = None,
        max_tokens: int | None = None,
        threads: int | None = None,
    ) -> "QwenGgufCliGenerator":
        profile = get_task_profile(task_id)
        return cls(
            model_path=resolve_slm_model_path(profile.task_id, model_path),
            llama_cli_path=llama_cli_path,
            ctx_size=ctx_size,
            max_tokens=max_tokens,
            threads=threads,
            task_id=profile.task_id,
            task_profile=profile,
        )

    def build_prompt(self, question: str, contexts: Sequence[str]) -> str:
        context = "\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
        return (
            "<|im_start|>system\n"
            f"{self.task_profile.system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"{self.task_profile.user_instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _parse_answer(output: str) -> str:
        before_perf = output.split("[ Prompt:", 1)[0]
        if "(truncated)" in before_perf:
            before_perf = before_perf.rsplit("(truncated)", 1)[-1]
        if "Answer with the shortest correct answer." in before_perf:
            before_perf = before_perf.rsplit("Answer with the shortest correct answer.", 1)[-1]
        if "<|im_start|>assistant" in before_perf:
            before_perf = before_perf.rsplit("<|im_start|>assistant", 1)[-1]
        lines = []
        for line in before_perf.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("Loading model", "build", "model", "modalities", "available commands", ">", "▄▄", "██")):
                continue
            if stripped in {"/exit or Ctrl+C     stop or exit", "/regen              regenerate the last response", "/clear              clear the chat history"}:
                continue
            lines.append(stripped)
        text = " ".join(lines).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _parse_memory(output: str) -> Dict[str, float]:
        memory: Dict[str, float] = {}
        host = re.search(r"- Host\s+\|\s+([0-9.]+)\s+=\s+([0-9.]+)\s+\+\s+([0-9.]+)\s+\+\s+([0-9.]+)", output)
        if host:
            memory["host_total_mib"] = float(host.group(1))
            memory["host_model_mib"] = float(host.group(2))
            memory["host_context_mib"] = float(host.group(3))
            memory["host_compute_mib"] = float(host.group(4))
        repack = re.search(r"- CPU_REPACK\s+\|\s+([0-9.]+)\s+=", output)
        if repack:
            memory["cpu_repack_mib"] = float(repack.group(1))
        return memory

    def generate(self, question: str, contexts: Sequence[str]) -> QwenCliGeneration:
        prompt = self.build_prompt(question, contexts)
        cmd = [
            str(self.llama_cli_path),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n",
            str(self.max_tokens),
            "--temp",
            str(self.task_profile.temperature),
            "--no-display-prompt",
            "--single-turn",
            "--simple-io",
            "-dev",
            "none",
            "-ngl",
            "0",
            "-c",
            str(self.ctx_size),
            "-t",
            str(self.threads),
            "--no-repack",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        elapsed = time.perf_counter() - t0
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode not in {0, 137}:
            raise RuntimeError(f"llama-cli failed with code {proc.returncode}:\n{output}")
        return QwenCliGeneration(
            answer=self._parse_answer(output),
            prompt=prompt,
            elapsed_s=elapsed,
            model_path=str(self.model_path),
            model_file_bytes=self.model_path.stat().st_size,
            backend="llama.cpp-cli-cpu-gguf",
            memory=self._parse_memory(output),
            task_id=self.task_id,
        )
