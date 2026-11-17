# Benchmark Methodology

## 1. Purpose

This document describes the procedures recorded in the benchmark workbook for comparing vision-language model (VLM) inference performance on an NVIDIA Jetson AGX Orin Developer Kit with 64 GB memory.

The benchmark measures serving latency, time to first token (TTFT), decode throughput, API concurrency behavior, system resource utilization, and a separate live-webcam exercise. It does **not** establish comparative visual-answer accuracy.

## 2. Models and benchmark runs

The workbook contains two related groups of tests:

### Matched fixed-image comparison

| Model | Runtime | Benchmark date |
|---|---|---|
| Cosmos3-Edge reasoning | TensorRT-Edge-LLM 0.10.0 | 2026-11-03 |
| `google/gemma-4-E2B-it` | vLLM 0.19.0 | 2026-11-03 |
| `Qwen3-VL-2B-Instruct` | vLLM 0.19.0 | 2026-11-04 |

The workbook explicitly notes that the installed Gemma checkpoint is **Gemma 4 E2B-it**, not a Gemma 4B checkpoint. The shorthand “Gemma 4B” should therefore not be used as the exact model identity.

Cosmos and Gemma were measured in the earlier matched matrix. Qwen3-VL-2B was benchmarked on the following day using the same fixed image and observation prompt. The comparison is useful, but the runs were independent rather than simultaneous.

### Qwen size rerun

A separate 2026-11-04 rerun compared:

- `Qwen3-VL-2B-Instruct`
- `Qwen3-VL-4B-Instruct`

Both were served one at a time with vLLM 0.19.0, BF16, an 8192-token serving context, `max_num_seqs=1`, and the same fixed image and prompt.

## 3. Fixed image and prompt

The fixed input image recorded in the workbook was:

`example_reasoning_input.png` (230,656 bytes)

The image was paired with this prompt:

> Write at least 25 concise numbered observations about this image. Include visible objects, colors, their positions, background, lighting, and likely setting. Continue enumerating. Do not add an introduction or conclusion.

The benchmark request used `temperature=0`, streaming enabled, and API usage reporting enabled (`stream_options.include_usage=true` where supported).

## 4. Sequential token sweep

For the Qwen benchmark and Qwen 2B/4B rerun, the protocol was:

1. Start the model server and ensure the other model container is stopped.
2. Send a warm-up request.
3. Run three measured requests at each output limit: 1, 10, 20, ..., 100 tokens.
4. Use the same image and prompt for each request.
5. Retain the individual request records and summarize using the workbook's median values.

The Qwen configuration reports 33 successful sequential requests and zero failures. The Qwen 2B and 4B rerun configurations also report 33/33 successful requests each.

For the original Cosmos/Gemma comparison, the workbook describes three measured repetitions at 1 and 100 output tokens, plus concurrency cases and a warm-up. Do not assume that every model has the same full 1-to-100 token sweep unless the corresponding raw request sheet confirms it.

## 5. Concurrency tests

Concurrency tests submitted either two or four simultaneous API requests, each with `max_tokens=100`, to a **single running model server**. No duplicate model server was started.

Important serving configuration:

- Cosmos engine: `max_batch_size=1`.
- vLLM models: `max_num_seqs=1`.

As a result, the vLLM server accepted the incoming requests and queued them; acceptance does not mean that all requests were processed in parallel. Cosmos rejected requests above its configured batch capacity with HTTP 503.

## 6. Telemetry and resource metrics

The workbook records CPU, GPU, RAM, and power telemetry during benchmark cases. The main comparison includes:

- Mean CPU utilization.
- Peak single CPU-core utilization.
- Mean and peak GPU utilization.
- Peak shared system RAM.
- Mean GPU rail power.
- Mean CPU/CV rail power.
- Mean 5V system input power.

Jetson AGX Orin uses unified memory. The reported RAM values are **shared system RAM**, not a separate dedicated VRAM pool. System memory includes the operating system and background applications.

Telemetry measurements are case-window measurements. They should not be interpreted as isolated per-model hardware consumption without accounting for background system activity.

## 7. Live webcam exercise

The separate Qwen 2B/4B live test used a Logitech Brio 300 camera and a local capture proxy/API path. The prompt was:

> Describe what you see in this image in one sentence.

The WebUI request used a 96-token output limit and temperature 0.7. The capture interval was every 30th webcam frame; the camera FPS was not instrumented.

The workbook reports:

| Metric | Qwen3-VL-2B | Qwen3-VL-4B |
|---|---:|---:|
| Successful response records | 29 | 12 |
| Mean full-response API latency | 2.427 s | 4.501 s |
| Median full-response API latency | 2.447 s | 4.422 s |
| Failed API records | 0 | 0 |

These were separate captures, not paired tests on identical frames. The live API used non-streaming chat completions, so TTFT was not separately available. The workbook states that webcam frames were not retained; image hashes and byte sizes were recorded.

## 8. Metric definitions and interpretation

- **API latency:** elapsed request time reported by the benchmark.
- **TTFT:** time to first generated token, where available.
- **Decode throughput:** completion tokens divided by the decode interval after TTFT, as described in the workbook.
- **Concurrency acceptance:** number of successful API requests divided by requests submitted.
- **Peak shared RAM:** highest system RAM reading in the measured case window.
- **Mean GPU utilization:** average GPU utilization over the measured case window.
- **Power:** distinguish GPU rail power, CPU/CV rail power, and 5V system input power; they are different measurements.

For image-plus-text calls, Cosmos reported `prompt_tokens=0` as an unavailable-count fallback. This must **not** be interpreted as zero prompt/image input tokens. Prompt-token counts from different model tokenizers are also not directly comparable.

## 9. Limitations

1. The benchmark measures serving performance and resource behavior, not visual reasoning accuracy.
2. No labeled ground-truth dataset or scored answer-quality evaluation is included.
3. Cosmos, Gemma, and Qwen runs were independent runs; the workbook notes their different benchmark dates.
4. The Qwen 2B/4B live tests used different captured frames and different capture durations.
5. The live camera FPS was not instrumented.
6. The vLLM sequence limit was one; concurrent requests were queued rather than served as simultaneous active sequences.
7. Telemetry includes shared Jetson system activity and background processes.
8. API-reported token counts have model/server-specific limitations, especially Cosmos prompt-token fallback reporting.

## 10. Source of truth

All measurements and configuration details in this document are transcribed from the accompanying benchmark workbook. Consult its raw request, case-summary, telemetry, configuration, and live-stream sheets for per-request records.
