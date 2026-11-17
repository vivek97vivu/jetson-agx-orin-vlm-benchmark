#!/usr/bin/env python3
"""Stream one Cosmos3-Edge image question and report latency and RAM."""

import argparse
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
COSMOS_ROOT = Path(__file__).resolve().parent.parent


def parse_ram(line):
    match = RAM_RE.search(line or "")
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="media/example_reasoning_input.png")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--question",
        default="The task is to put the flower into the red bottle. Generate a plan consisting of subtasks to accomplish the task.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    image = str((image_path if image_path.is_absolute() else COSMOS_ROOT / image_path).resolve())
    payload = {
        "model": "reasoning",
        "messages": [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.question},
        ]}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    stop = threading.Event()
    ram = {"baseline_mb": None, "peak_mb": None, "total_mb": None}
    proc = None
    reader = None
    tegrastats = shutil.which("tegrastats")
    if tegrastats:
        proc = subprocess.Popen(
            [tegrastats, "--interval", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        first = proc.stdout.readline()
        ram["baseline_mb"] = parse_ram(first)
        if ram["baseline_mb"] is not None:
            found = RAM_RE.search(first)
            ram["total_mb"] = int(found.group(2))
            ram["peak_mb"] = ram["baseline_mb"]

        def sample_ram():
            for line in proc.stdout:
                if stop.is_set():
                    break
                used = parse_ram(line)
                if used is not None:
                    ram["peak_mb"] = max(ram["peak_mb"] or 0, used)

        reader = threading.Thread(target=sample_ram, daemon=True)
        reader.start()

    chunks = []
    first_token_s = None
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - started
                    chunks.append(text)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        if proc:
            proc.terminate()
            proc.wait(timeout=3)
        if reader:
            reader.join(timeout=1)

    answer = "".join(chunks)
    token_count = None
    tokenization_error = None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(
            COSMOS_ROOT / "work/Cosmos3-Edge/engines/reasoning/tokenizer.json"
        ))
        token_count = len(tokenizer.encode(answer).ids)
    except Exception as exc:
        tokenization_error = str(exc)

    generation_s = max(elapsed - (first_token_s or elapsed), 1e-9)
    print(json.dumps({
        "image": image,
        "answer": answer,
        "ttft_s": round(first_token_s, 3) if first_token_s is not None else None,
        "total_s": round(elapsed, 3),
        "completion_tokens": token_count,
        "tokenization_error": tokenization_error,
        "decode_tokens_per_s": round((token_count or 0) / generation_s, 2) if token_count else None,
        "ram_baseline_mb": ram["baseline_mb"],
        "ram_peak_mb": ram["peak_mb"],
        "ram_total_mb": ram["total_mb"],
        "ram_delta_mb": (ram["peak_mb"] - ram["baseline_mb"])
        if ram["peak_mb"] is not None and ram["baseline_mb"] is not None else None,
    }, indent=2))


if __name__ == "__main__":
    main()
