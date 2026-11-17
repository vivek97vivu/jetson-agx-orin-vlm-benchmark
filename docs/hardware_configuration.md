# Hardware and Software Configuration

## 1. Target platform

The benchmark was conducted on an NVIDIA Jetson AGX Orin Developer Kit with a 64 GB module.

| Component | Recorded configuration |
|---|---|
| Platform | NVIDIA Jetson AGX Orin Developer Kit |
| Memory | 64 GB unified system memory |
| Power mode | `MODE_50W` |
| CPU architecture | ARM aarch64 |
| CPU | 12-core NVIDIA Carmel/Arm Cortex-A78AE platform as reported by `jetson_release` |
| GPU | Integrated NVIDIA Jetson AGX Orin GPU |
| Operating system | Ubuntu 22.04.5 LTS (Jammy) |
| Kernel | 5.15.148-tegra |
| JetPack | 6.2.1 |
| L4T | 36.4.7 |

The workbook's Jetson release output identifies the module as NVIDIA Jetson AGX Orin (64GB RAM). The CPU details are reproduced from the captured system report; consult the original system output if a hardware inventory needs to be independently revalidated.

## 2. Jetson software stack

The original Cosmos configuration records:

| Component | Version |
|---|---|
| JetPack | 6.2.1 |
| L4T | 36.4.7 |
| CUDA | 12.6.68 |
| cuDNN | 9.3.0 |
| TensorRT | 10.3.0.30 |
| VPI | 3.2.4.0 |
| Vulkan | 1.3.204 |
| OpenCV | 4.10.0, CUDA enabled |
| jetson-stats / jtop | 7.2.0 |

The benchmark workbook records these platform versions from the system configuration captured for the Cosmos run. The Qwen rerun records the Jetson AGX Orin 64GB and MODE_50W, and vLLM 0.19.0.

## 3. Runtime and model configuration

### Cosmos3-Edge reasoning

| Setting | Value |
|---|---|
| Runtime | TensorRT-Edge-LLM 0.10.0 |
| API endpoint | `http://127.0.0.1:8000/v1/chat/completions` |
| Engine max batch size | 1 |
| Maximum input length | 2048 |
| Maximum KV cache capacity | 4096 |
| KV cache dtype | FP16 |
| Engine files | `llm.engine` 3.137 GiB; `visual.engine` 0.916 GiB |
| Request settings | Temperature 0; streaming; output limits 1 or 100 |

The workbook's captured launch configuration includes `--max-input-len 2048`, `--max-batch-size 1`, and `--max-kv-cache-capacity 4096`.

### Gemma 4 E2B Instruct

| Setting | Value |
|---|---|
| Exact model ID | `google/gemma-4-E2B-it` |
| Runtime | vLLM 0.19.0 |
| Container | `gemma4-jetson-orin` image; container name `gemma4-e2b` |
| API endpoint | `http://127.0.0.1:8000/v1/chat/completions` |
| Maximum model length | 8192 |
| Maximum sequences | 1 |
| GPU memory utilization target | 0.20 |
| vLLM-reported model load memory | 9.79 GiB |
| Power mode | `MODE_50W` |

**Model naming clarification:** The workbook says the requested shorthand was “Gemma 4B”, but the exact loaded model was `google/gemma-4-E2B-it`. Use the exact ID in technical documentation.

### Qwen3-VL-2B Instruct

| Setting | Value |
|---|---|
| Exact model ID | `Qwen3-VL-2B-Instruct` |
| Runtime | vLLM 0.19.0 |
| Checkpoint dtype | BF16 |
| API endpoint | `http://127.0.0.1:8000/v1/chat/completions` |
| Maximum model length | 8192 |
| Maximum sequences | 1 |
| GPU memory utilization target | 0.20 |
| vLLM-reported model load memory | 4.24 GiB |
| Checkpoint file size | 4,255,140,312 bytes |
| Power mode | `MODE_50W` |

### Qwen3-VL-4B Instruct

| Setting | Value |
|---|---|
| Exact model ID | `Qwen3-VL-4B-Instruct` |
| Runtime | vLLM 0.19.0 |
| Checkpoint dtype | BF16 |
| API endpoint | `http://127.0.0.1:8000/v1/chat/completions` |
| Maximum model length | 8192 |
| Maximum sequences | 1 |
| GPU memory utilization target | 0.20 |
| vLLM-reported model load memory | 8.59 GiB |
| Checkpoint file size | 8,875,719,344 bytes |
| Power mode | `MODE_50W` |

The Qwen 2B and 4B configurations were run one at a time. The workbook records that the other model container was stopped during each model's benchmark.

## 4. Memory and power interpretation

Jetson AGX Orin uses unified memory. The benchmark's “peak shared system RAM” is system RAM and must not be described as dedicated GPU VRAM.

The workbook reports three distinct power measures:

- GPU rail power.
- CPU/CV rail power.
- 5V system input power.

Do not combine or substitute these values. When presenting system power comparisons, label the 5V system input measurement explicitly.

## 5. Reproducibility notes

The recorded model server settings, model identifiers, software versions, and power mode should be preserved when reproducing the tests. A different vLLM version, model quantization, serving context, sequence limit, GPU memory utilization setting, or power mode can change results.

The workbook contains model-specific configuration sheets. Refer to those sheets for exact command lines and per-run metadata.
