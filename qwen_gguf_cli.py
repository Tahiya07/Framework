from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence


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


class QwenGgufCliGenerator:
    """Qwen2.5 GGUF generator using standalone llama.cpp CLI."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_QWEN_GGUF,
        *,
        llama_cli_path: str | Path | None = None,
        ctx_size: int = 512,
        max_tokens: int = 48,
        threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Qwen GGUF not found: {self.model_path}")
        self.llama_cli_path = Path(llama_cli_path) if llama_cli_path else find_llama_cli()
        self.ctx_size = int(ctx_size)
        self.max_tokens = int(max_tokens)
        self.threads = int(threads)

    @staticmethod
    def build_prompt(question: str, contexts: Sequence[str]) -> str:
        context = "\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
        return (
            "<|im_start|>system\n"
            "You are a careful academic assistant. Answer using only the supplied context. "
            "If the answer is not in the context, say: I don't know based on the provided context. "
            "Keep the answer concise.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer with the shortest correct answer.\n"
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
            "0",
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
        )
