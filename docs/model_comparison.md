# Model Performance Comparison

## 1. Scope

This report summarizes the performance measurements recorded in the benchmark workbook for NVIDIA Jetson AGX Orin 64GB in `MODE_50W`.

It contains two comparisons:

1. The main fixed-image 100-token comparison: Cosmos3-Edge reasoning, Gemma 4 E2B-it, and Qwen3-VL-2B-Instruct.
2. A separate Qwen3-VL-2B vs Qwen3-VL-4B rerun.

These groups must remain distinct. The Qwen 2B/4B rerun has different run conditions and should not be silently mixed into the main comparison.

## 2. Main 100-token comparison

| Metric | Cosmos3-Edge reasoning | Gemma 4 E2B-it | Qwen3-VL-2B-Instruct |
|---|---:|---:|---:|
| Runtime | TensorRT-Edge-LLM 0.10.0 | vLLM 0.19.0 | vLLM 0.19.0 |
| 1-token median latency | 0.145 s | 0.131 s | **0.098 s** |
| 100-token median API latency | **3.312 s** | 4.748 s | 3.441 s |
| 100-token median TTFT | 0.136 s | **0.128 s** | 0.097 s |
| 100-token median decode throughput | **31.50 tok/s** | 21.64 tok/s | 29.89 tok/s |
| Mean CPU utilization | **35.98%** | 52.62% | 52.61% |
| Mean GPU utilization | 82.48% | 88.89% | 84.42% |
| Peak shared system RAM | **39,855 MiB** | 50,025 MiB | 40,448 MiB |
| Mean GPU rail power | 12.68 W | 12.70 W | 12.94 W |
| Mean CPU/CV rail power | **2.57 W** | 3.33 W | 3.38 W |
| Mean 5V system input power | **9.71 W** | 9.84 W | 10.06 W |

### Main comparison observations

- Cosmos recorded the highest 100-token decode throughput at 31.50 tokens/s and the lowest 100-token API latency at 3.312 seconds in this matrix.
- Qwen3-VL-2B recorded the lowest 1-token median latency at 0.098 seconds.
- Qwen3-VL-2B's 100-token throughput was 29.89 tokens/s, close to Cosmos in this run.
- Cosmos recorded the lowest mean CPU utilization and lowest peak shared system RAM among these three measurements.
- Gemma 4 E2B-it recorded the lowest throughput and highest 100-token API latency in this comparison.
- Mean GPU utilization is an activity measure, not an efficiency score; higher is not automatically better.

## 3. Concurrency behavior in the main comparison

| Test | Cosmos3-Edge | Gemma 4 E2B-it | Qwen3-VL-2B |
|---|---:|---:|---:|
| 2 concurrent requests accepted | 1/2 | 2/2 | 2/2 |
| 2-request aggregate output throughput | 30.24 tok/s | 20.88 tok/s | 27.80 tok/s |
| 2-request wall time | 3.307 s | 9.577 s | 7.194 s |
| 4 concurrent requests accepted | 1/4 | 4/4 | 4/4 |
| 4-request aggregate output throughput | 30.08 tok/s | 20.96 tok/s | 29.16 tok/s |
| 4-request wall time | 3.325 s | 19.085 s | 13.719 s |

The Cosmos engine was configured with max batch size 1 and rejected requests above that capacity. The vLLM servers were configured with `max_num_seqs=1`; they accepted and queued requests. Therefore, request acceptance must not be interpreted as four simultaneous active model sequences.

## 4. Separate Qwen3-VL-2B vs Qwen3-VL-4B rerun

The following results come from the separate 2026-11-04 rerun, with both models served individually using vLLM 0.19.0, BF16, an 8192-token serving context, and `max_num_seqs=1`.

| Metric | Qwen3-VL-2B | Qwen3-VL-4B |
|---|---:|---:|
| Weight file size | 4.26 GB | 8.88 GB |
| vLLM model load memory | 4.24 GiB | 8.59 GiB |
| 1-token median API latency | **0.110 s** | 0.155 s |
| 100-token median API latency | **4.590 s** | 14.025 s |
| 100-token median TTFT | **0.102 s** | 3.913 s |
| 100-token median decode speed | **22.28 tok/s** | 9.88 tok/s |
| 100-token mean CPU utilization | **62.9%** | 84.2% |
| 100-token mean GPU utilization | **92.2%** | 98.1% |
| 100-token peak shared system RAM | **47,842 MiB** | 49,349 MiB |
| 100-token mean GPU rail power | **11.66 W** | 13.22 W |
| 100-token mean CPU/CV rail power | **3.94 W** | 4.97 W |
| 100-token mean 5V system input power | **9.41 W** | 10.20 W |

### Qwen live webcam results

| Metric | Qwen3-VL-2B | Qwen3-VL-4B |
|---|---:|---:|
| Successful captured responses | 29 | 12 |
| Failed API records | 0 | 0 |
| Mean full-response API latency | **2.427 s** | 4.501 s |
| Median full-response API latency | **2.447 s** | 4.422 s |
| Peak shared system RAM | 48,474 MiB | 50,183 MiB |
| Peak GPU utilization | 99% | 99% |

The live runs used a Logitech Brio 300 and every 30th webcam frame. They were separate captures, not paired tests on identical frames; camera FPS was not instrumented. The API was non-streaming, so TTFT was unavailable for this live test.

### Qwen rerun interpretation

- Qwen3-VL-2B was faster in the measured 1-token and 100-token requests.
- Qwen3-VL-2B used less model load memory and had lower measured CPU, GPU, RAM, and power figures in the 100-token case.
- Both models accepted all submitted 2-request and 4-request concurrency tests, but requests queued at `max_num_seqs=1`.
- The workbook does not establish that the 4B model is more accurate. It explicitly records that visual correctness was not ground-truth scored.

## 5. Important measurement caveats

1. **Gemma identity:** The tested model is `google/gemma-4-E2B-it`, not a Gemma 4B checkpoint.
2. **Prompt-token counts:** Cosmos returned zero as an unavailable prompt-token count fallback for image-plus-text requests. This is not zero actual input tokens.
3. **Tokenizer differences:** API-reported prompt-token counts are not directly comparable across model tokenizers.
4. **Unified memory:** RAM values are shared Jetson system RAM, not dedicated VRAM.
5. **Independent runs:** The main comparison was run across separate model sessions; Qwen was measured on 2026-11-04 after the Cosmos/Gemma matrix.
6. **No accuracy score:** The benchmark is a performance/resource comparison, not a semantic quality evaluation.
7. **Concurrency:** vLLM accepted and queued requests with `max_num_seqs=1`; this is not multi-sequence parallel execution.

## 6. Overall performance conclusion

For the main fixed-image 100-token matrix, Cosmos3-Edge had the highest measured decode throughput and lowest total API latency. Qwen3-VL-2B was close to Cosmos in throughput and had the lowest 1-token latency. Gemma 4 E2B-it was slower on the measured long-output case.

In the separate Qwen size rerun, Qwen3-VL-2B outperformed Qwen3-VL-4B on the measured latency, throughput, and resource metrics. These findings support a performance-based preference for Qwen3-VL-2B in the tested setup, but a deployment decision should also consider task-specific answer quality, which this workbook did not score.
