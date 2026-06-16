# SIS-GPU-Inference: LLM Inference Sustainability on GPU Deployments

**Developed by Urooj Asgher** 
Technological University Dublin, Ireland 
ORCID: 0000-0001-9218-3307

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Urooj%20Asgher-blue)](https://orcid.org/0000-0001-9218-3307)
[![HPC Nexus Lab](https://img.shields.io/badge/HPC-Nexus%20Lab-purple)](https://github.com/HPC-Nexus-Lab)
[![Institution](https://img.shields.io/badge/TU%20Dublin-Ireland-green)](https://www.tudublin.ie)
[![Paper](https://img.shields.io/badge/Paper-SIS--LLM-orange)](https://your-paper-link-here)
[![CPU Version](https://img.shields.io/badge/Also%20See-CPU%20Version-lightgrey)](https://github.com/urooj88/SIS-CPU-Inference)

---

## Overview

**SIS-GPU-Inference** is the GPU version of the SIS-LLM framework for evaluating the sustainability of Large Language Model (LLM) inference. It combines performance, efficiency, and environmental metrics into a single interpretable score — the **Sustainability Index Score (SIS)**.

> This is the **GPU version**, evaluated on an HPC server using NVIDIA L40S GPUs with physical power metering via an **Adcewatt** power meter. All model layers are fully offloaded to GPU using `llama.cpp` with `-ngl 999`.

---

## Methodology

The SIS framework follows three main steps:

### Step 1 — Measurement
During inference, both model-level and system-level metrics are captured:
- **Model-level:** accuracy (GSM8K, MMLU, TruthfulQA), FLOPs, model size, token counts
- **System-level:** energy consumption (Joules), execution time, GPU VRAM usage, GPU count, GPU hours
- **Environmental:** carbon emissions estimated from energy and carbon intensity (gCO₂/kWh)

Energy is measured using a physical **Adcewatt power meter** connected via serial port (`/dev/ttyUSB0`), reading channels `#activepow8` and `#activepow9`. Dynamic energy subtracts idle baseline:

```
Dynamic Energy (J) = (Total Power − Baseline Power) × Execution Time
Carbon Emissions   = (Dynamic Energy ÷ 3,600,000) × Carbon Intensity
GPU Hours          = GPU Count × Execution Time (hours)
Hardware Efficiency = Accuracy ÷ GPU Hours
```

### Step 2 — Normalisation
All metrics normalised to [0, 1]:

- **Lower is better** (energy, CO₂, runtime, memory, FLOPs, model size):
```
Norm = (x − min) / (max − min)
```
- **Higher is better** (accuracy, throughput, token energy efficiency):
```
Norm = 1 − (x − min) / (max − min)
```

### Step 3 — SIS Score
```
SIS = 1 − Σ (wᵢ × Normᵢ)     where wᵢ = 1/9 (equal weights)
```
**Lower SIS = better sustainability.**

### SIS Classification

| SIS Score | Sustainability Level |
|---|---|
| 0.0 – 0.3 | 🟢 Low Impact |
| 0.3 – 0.7 | 🟡 Medium Impact |
| 0.7 – 1.0 | 🔴 High Impact |

### Metrics Used in SIS

| Metric | Unit | Goal |
|---|---|---|
| Energy Consumption | Joules/Query | Lower is better |
| Carbon Emissions | gCO₂eq/Query | Lower is better |
| Execution Time | Seconds/Query | Lower is better |
| Memory Usage | GB (VRAM) | Lower is better |
| FLOPs | Per inference | Lower is better |
| Model Size | MB | Lower is better |
| Accuracy | % | Higher is better |
| Throughput | Tokens/sec | Higher is better |
| Token Energy Efficiency | Tokens/J | Higher is better |

---

## CPU vs GPU Comparison

| Feature | CPU Version | GPU Version |
|---|---|---|
| Hardware | Intel Xeon Gold 6430 | NVIDIA L40S (48 GB) |
| GPU offload | Disabled (`CUDA_VISIBLE_DEVICES=""`) | Full (`-ngl 999`) |
| Hardware efficiency | Accuracy ÷ CPU-hours | Accuracy ÷ GPU-hours |
| Memory measured | Host RAM (GiB) | VRAM (GiB) |
| Repo | [SIS-CPU-Inference](https://github.com/urooj88/SIS-CPU-Inference) | This repo |

---

## Models Evaluated

| Model | Parameters | Quantisation | GPU Layers |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | GGUF Q4\_K\_M | 999 (full) |
| Mistral-7B-Instruct-v0.3 | 7B | GGUF Q4\_K\_M | 999 (full) |
| Meta-Llama-3.1-8B-Instruct | 8B | GGUF Q4\_K\_M | 999 (full) |
| Phi-3.5-mini-Instruct | 3.8B | GGUF Q4\_K\_M | 999 (full) |

All models are deployed using [`llama.cpp`](https://github.com/ggerganov/llama.cpp) with GGUF Q4\_K\_M quantisation and full GPU layer offloading.

---

## Benchmarks

| Dataset | Task | Samples |
|---|---|---|
| GSM8K | Mathematical Reasoning | 500 |
| MMLU | Multi-domain MCQ | 500 |
| TruthfulQA | Factual Truthfulness | 500 |

**Total: 1500 prompts per model** (fixed seed = 42)

---

## Repository Structure

```
SIS-GPU-Inference/
├── README.md
├── requirements.txt
├── main_sustainability_runner_LLM_GPU.py    ← Main entry point (GPU)
├── build_eval_dataset.py                    ← Builds evaluation dataset
├── run_omegawatt_log_both_models12_same.sh  ← Runs all 4 models with power logging
├── omegawatt_scripts/
│   ├── run_omegawatt_log_models12.py        ← Per-model power logger (GPU-aware, --gpu-count)
│   └── run_basepower_adcewatt_var_std.py    ← Baseline power measurement
└── test_models/
    ├── collect_inference_metrics.py         ← Computes accuracy and metrics
    ├── model1.sh                            ← Qwen2.5  (-ngl 999, CUDA_VISIBLE_DEVICES=0)
    ├── model2.sh                            ← Mistral  (-ngl 999, CUDA_VISIBLE_DEVICES=0)
    ├── model3.sh                            ← LLaMA    (-ngl 999, CUDA_VISIBLE_DEVICES=0)
    └── model4.sh                            ← Phi-mini (-ngl 999, CUDA_VISIBLE_DEVICES=0)
```

> `model_logs/` is auto-created at runtime. Do not commit it. 
> `test_models/prompts.txt` and `answers.csv` are auto-generated. Do not commit them.

---

## Installation

```bash
git clone https://github.com/urooj88/SIS-GPU-Inference.git
cd SIS-GPU-Inference
pip install -r requirements.txt
```

Install [`llama.cpp`](https://github.com/ggerganov/llama.cpp) **with CUDA support** and download GGUF Q4\_K\_M models:

| Model | HuggingFace |
|---|---|
| Qwen2.5-7B | [Link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |
| Mistral-7B | [Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3-GGUF) |
| LLaMA-3.1-8B | [Link](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| Phi-3.5-mini | [Link](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) |

Update `BASE_DIR`, `MODEL` and `LLAMA_CLI` paths in each `test_models/model*.sh` to point to your local files.

---

## Usage

### Step 1 — Build dataset (first time only)
```bash
python3 build_eval_dataset.py --reason 500 --mcq 500 --truth 500 --force-rebuild
```

### Step 2 — Run the full GPU pipeline
```bash
python3 main_sustainability_runner_LLM_GPU.py
```

### Step 3 — View results
```
model_logs/sustainability_detailed_report.txt
model_logs/updated_sustainability_metrics.xlsx
```

---

## Hardware Requirements

- NVIDIA GPU with sufficient VRAM (~6–8 GB per model)
- CUDA toolkit installed
- Adcewatt power meter via `/dev/ttyUSB0`
- Adcewatt binary (`wattmetre-readmv2new`)
- llama.cpp built with CUDA support (`-DLLAMA_CUDA=ON`)

> Tested on: NVIDIA L40S (48 GB VRAM), 2× Intel Xeon Gold 6430 (64 cores)

---

## Citation

```bibtex
@inproceedings{asgher2025sis,
  title  = {SIS: A Sustainability Index for Evaluating Energy-Efficient LLM
            Inference Across CPU and GPU Deployments},
  author = {Asgher, Urooj and Malik, Tania},
  year   = {2025},
  institution = {Technological University Dublin, Ireland}
}
```

---

## Acknowledgements

Experiments were carried out using the HPCNexus testbed at Technological University Dublin.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
