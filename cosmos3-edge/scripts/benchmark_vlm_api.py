#!/usr/bin/env python3
"""Repeatable streaming benchmark for local OpenAI-style and TensorRT VLM APIs."""

import argparse
import base64
import json
import mimetypes
import re
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def median(values):
    return round(statistics.median(values), 3) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default="media/example_reasoning_input.png")
    parser.add_argument("--question", default=(
        "The task is to put the flower into the red bottle. Generate a plan "
        "consisting of subtasks to accomplish the task."
    ))
    parser.add_argument("--format", choices=("edgellm", "openai"), required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    image_bytes = image_path.read_bytes()
    if args.format == "edgellm":
        image_content = {"type": "image", "image": str(image_path)}
    else:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
        image_content = {"type": "image_url", "image_url": {"url": data_url}}

    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))

    monitor_samples = []
    monitor_stop = threading.Event()
    monitor = None
    tegrastats = shutil.which("tegrastats")
    if tegrastats:
        monitor = subprocess.Popen(
            [tegrastats, "--interval", "250"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

        def sample_metrics():
            for line in monitor.stdout:
                if monitor_stop.is_set():
                    break
                row = {}
                ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
                gpu = re.search(r"GR3D_FREQ\s+(\d+)%", line)
                if ram:
                    row["ram_used_mib"] = int(ram.group(1))
                    row["ram_total_mib"] = int(ram.group(2))
                if gpu:
                    row["gpu_percent"] = int(gpu.group(1))
                if row:
                    monitor_samples.append(row)

        threading.Thread(target=sample_metrics, daemon=True).start()

    results = []
    try:
        for index in range(args.runs + 1):
            message_content = ([image_content, {"type": "text", "text": args.question}]
                               if args.format == "edgellm"
                               else [{"type": "text", "text": args.question}, image_content])
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": message_content}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
            }
            request = urllib.request.Request(
                args.api_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            first_token = None
            chunks = []
            with urllib.request.urlopen(request, timeout=240) as response:
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
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    text = delta.get("content") or ""
                    if text:
                        if first_token is None:
                            first_token = time.perf_counter()
                        chunks.append(text)
            elapsed = time.perf_counter() - started
            answer = "".join(chunks)
            token_count = len(tokenizer.encode(answer).ids) if tokenizer else None
            generation = max(elapsed - ((first_token - started) if first_token else elapsed), 1e-9)
            row = {
                "run": "warmup" if index == 0 else index,
                "ttft_s": round(first_token - started, 3) if first_token else None,
                "total_s": round(elapsed, 3),
                "completion_tokens": token_count,
                "decode_tokens_per_s": round(token_count / generation, 2) if token_count else None,
                "answer": answer,
            }
            if index:
                results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if monitor:
            monitor_stop.set()
            monitor.terminate()
            monitor.wait(timeout=3)

    summary = {
        "model": args.model,
        "api_url": args.api_url,
        "image": str(image_path),
        "format": args.format,
        "runs": len(results),
        "median_ttft_s": median([row["ttft_s"] for row in results if row["ttft_s"] is not None]),
        "median_total_s": median([row["total_s"] for row in results]),
        "median_decode_tokens_per_s": median(
            [row["decode_tokens_per_s"] for row in results if row["decode_tokens_per_s"] is not None]
        ),
        "gpu_peak_percent": max((row.get("gpu_percent", 0) for row in monitor_samples), default=None),
        "ram_peak_mib": max((row.get("ram_used_mib", 0) for row in monitor_samples), default=None),
        "ram_total_mib": next((row.get("ram_total_mib") for row in reversed(monitor_samples)
                                if row.get("ram_total_mib")), None),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
