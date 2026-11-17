# Deployment Recommendation

## 1. Executive recommendation

For a latency- and resource-sensitive real-time VLM deployment on the tested NVIDIA Jetson AGX Orin 64GB in `MODE_50W`, the benchmark evidence supports **Qwen3-VL-2B-Instruct as the practical starting point** when choosing between the Qwen 2B and Qwen 4B variants.

This is a performance recommendation based on the recorded tests. It is not a claim that Qwen 2B has better visual reasoning accuracy. The workbook does not contain a ground-truth accuracy evaluation.

For the broader three-model fixed-image comparison, Cosmos3-Edge delivered the highest measured 100-token decode throughput and lowest total API latency. The best choice therefore depends on the runtime and deployment requirements, not one metric alone.

## 2. Recommendation by use case

| Use case | Evidence-based recommendation | Reason |
|---|---|---|
| Lowest 100-token latency in the main three-model matrix | Cosmos3-Edge | 3.312 s median API latency |
| Highest decode throughput in the main matrix | Cosmos3-Edge | 31.50 tok/s |
| Lowest 1-token latency in the main matrix | Qwen3-VL-2B | 0.098 s |
| Qwen model for current real-time testing | Qwen3-VL-2B | Faster 100-token inference and better live response timing than Qwen3-VL-4B in the rerun |
| Gemma-specific ecosystem or compatibility requirement | Gemma 4 E2B-it | Choose when the model family or compatibility is a requirement, not because it won this speed test |
| Need to evaluate visual answer quality | Run a separate accuracy test | Current benchmark did not score answers against ground truth |

## 3. Why Qwen3-VL-2B is preferred over Qwen3-VL-4B in the rerun

The 2026-11-04 Qwen rerun found:

- 100-token median API latency: 4.590 s for 2B vs 14.025 s for 4B.
- 100-token decode speed: 22.28 tok/s for 2B vs 9.88 tok/s for 4B.
- Model load memory: 4.24 GiB for 2B vs 8.59 GiB for 4B.
- Mean CPU utilization: 62.9% for 2B vs 84.2% for 4B.
- Mean GPU utilization: 92.2% for 2B vs 98.1% for 4B.
- Peak shared system RAM: 47,842 MiB for 2B vs 49,349 MiB for 4B.
- Live webcam mean response latency: 2.427 s for 2B vs 4.501 s for 4B.

Both models accepted all 2-request and 4-request concurrency submissions. However, both were configured with `max_num_seqs=1`, so vLLM queued requests rather than processing multiple active sequences concurrently.

The live response counts (29 for 2B and 12 for 4B) should not be treated as a controlled FPS comparison: the runs used separate captures, different capture durations, and unpaired frames.

## 4. Main comparison: Cosmos vs Gemma vs Qwen 2B

The main fixed-image comparison produced a different trade-off:

- **Cosmos3-Edge:** strongest measured long-output speed (31.50 tok/s) and lowest 100-token API latency (3.312 s). It also used the least mean CPU and peak shared RAM in the matrix. Its configured max batch size was one, and excess concurrent requests were rejected with HTTP 503.
- **Qwen3-VL-2B:** fastest 1-token response (0.098 s), 29.89 tok/s at 100 tokens, and accepted all submitted concurrency requests by queueing them. It is a balanced option where the Qwen model family and API behavior are desired.
- **Gemma 4 E2B-it:** 21.64 tok/s at 100 tokens and 4.748 s median API latency. It may still be appropriate where Gemma-specific compatibility is important, but it did not lead the measured performance matrix.

The exact Gemma checkpoint tested was `google/gemma-4-E2B-it`, not a 4B checkpoint.

## 5. Recommended next validation before production

The current workbook is sufficient for an initial performance decision, but production selection should include the following additional tests.

### A. Task-specific quality evaluation

Create a representative, labeled image set from the actual deployment environment. Define a scoring rubric before testing, such as object/attribute correctness, missed critical details, hallucinated details, and response completeness. Run all candidate models on identical frames and prompts, then score outputs blind against ground truth.

### B. Paired live-camera test

Capture and save the same frames for every model. Replay identical frames through each model with the same prompt, output limit, sampling settings, and serving configuration. Record frame capture time, request start/end, TTFT where streaming is enabled, full response latency, and failures.

### C. Sustained-load and concurrency test

Test the actual expected number of clients. Record queue wait, request completion latency (median and p95), throughput, rejection/error rate, and thermal behavior. Test more than one `max_num_seqs` setting where memory permits, while preserving the same settings for fair model comparisons.

### D. Long-duration stability

Run a sustained test representative of deployment duration. Monitor shared RAM, GPU/CPU utilization, power, temperature, throttling, server restarts, and request failures.

### E. End-to-end application test

Measure the complete application pipeline, not only the model API. Include camera capture, image preprocessing, request transport, inference, post-processing, and downstream action latency.

## 6. Deployment starting point

For a first real-time Qwen deployment based on the recorded results:

- Start with `Qwen3-VL-2B-Instruct`.
- Preserve the tested BF16 model configuration and vLLM 0.19.0 as the baseline.
- Record the serving context (`max_model_len=8192`), `max_num_seqs=1`, and GPU memory utilization target (`0.20`) when reproducing the benchmark.
- Keep the model isolated during performance measurements.
- Measure actual application latency and quality before setting production thresholds.

These are benchmark reproduction settings, not a universal optimized configuration. Tune them only through controlled tests on the target workload.

## 7. Decision statement

**Recommended performance baseline: Qwen3-VL-2B-Instruct for the current real-time Qwen deployment evaluation.**

**Alternative when maximum measured decode speed in the main matrix is the priority: Cosmos3-Edge reasoning**, subject to its single-batch request behavior and TensorRT-Edge-LLM deployment requirements.

Do not select Qwen3-VL-4B solely on the assumption that a larger model is more accurate. The workbook does not demonstrate an accuracy advantage. Promote it only if a paired, ground-truth quality evaluation shows a meaningful improvement that justifies its higher measured latency and resource use.
