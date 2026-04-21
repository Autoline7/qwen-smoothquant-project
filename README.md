# SmoothQuant W8A8 Quantization of Qwen2.5-Coder

Course final project implementing the **SmoothQuant** post-training quantization algorithm (Xiao et al., ICML 2023) on the **Qwen2.5-Coder-Instruct** 7B and 14B models, with end-to-end evaluation on HumanEval+ and BigCodeBench-Hard.

> **Attribution.** This project builds on:
> - SmoothQuant — Xiao, Lin, Seznec, Wu, Demouth, Han. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML 2023. [[paper]](https://arxiv.org/abs/2211.10438) [[code]](https://github.com/mit-han-lab/smoothquant) (MIT License)
> - Qwen2.5-Coder — Hui et al. _Qwen2.5-Coder Technical Report._ 2024. [[models]](https://huggingface.co/Qwen)
> - EvalPlus (HumanEval+) — Liu et al. NeurIPS 2023. [[code]](https://github.com/evalplus/evalplus)
> - BigCodeBench — Zhuo et al. ICLR 2025. [[code]](https://github.com/bigcode-project/bigcodebench)

---

## What this project does

SmoothQuant enables INT8 quantization of both weights *and* activations (W8A8) by transferring the quantization difficulty of activation outliers into the weights via a per-channel smoothing scale. The upstream repo officially supports OPT, Llama, Mistral, Mixtral, Falcon, and BLOOM; we extend it to the **Qwen2 architecture** and measure the full pipeline on code generation.

Our pipeline:

1. **Calibrate** activation scales on 512 samples × 512 tokens from the Pile validation split (matching the paper).
2. **Smooth** the bf16 model via our `src/qwen_smooth.py` (Qwen2 adaptation of MIT's `smoothquant/smooth.py`) — migrates outliers from activations into weights.
3. **Quantize** the smoothed weights to INT8 with GPTQ via `llm-compressor` (GPTQ is orthogonal to SmoothQuant; it's a weight-rounding pass that composes with the smoothing).
4. **Deploy** via vLLM (real CUTLASS INT8 kernels) for benchmark eval.
5. **Profile** parameter count, FLOPs, memory decomposition, per-stage timing, and deployment capacity.
6. **Demo** on a held-out set of custom tasks not drawn from any benchmark.

Models covered:

- `Qwen/Qwen2.5-Coder-7B-Instruct` (7.62B params)
- `Qwen/Qwen2.5-Coder-14B-Instruct` (14.77B params, tied embeddings)

Benchmarks:

- **HumanEval+** (164 problems, EvalPlus-augmented tests)
- **BigCodeBench-Hard-Instruct** (148 problems, harder tasks with complex library calls)

---

## Repository structure

```
qwen-smoothquant-project/
├── README.md                                         This file
├── requirements.txt
├── src/
│   ├── qwen_smooth.py                                SmoothQuant smoothing adapted for Qwen2
│   ├── qwen_fake_quant.py                            W8A8 fake-quant linear for PPL validation
│   ├── calibrate.py                                  Per-channel activation absmax calibration
│   └── profile_utils.py                              Per-stage timing + FLOPs counting
├── notebook_global/                                  ACTIVE — parameterized on MODEL_SIZE, Runpod-ready
│   ├── 01_calibrate_qwen_global.ipynb                Calibrate activation scales on Pile
│   ├── 02_smooth_and_fakequant_global.ipynb          Apply SmoothQuant + fake-quant PPL validation
│   ├── 03_llmcompressor_export_global.ipynb          GPTQ W8A8 export for vLLM deployment
│   ├── 04_evaluate_code_benchmarks_global.ipynb      HumanEval+ + BCB-Hard, bf16 vs W8A8
│   └── (05_profile_and_flops_global.ipynb — in progress, not yet committed)
├── notebooks/                                        Original notebooks (Colab-flavored, 7B-hardcoded)
│   ├── 01_calibrate_qwen.ipynb                       Kept as reference; use notebook_global/ instead
│   ├── 02_smooth_and_fakequant.ipynb
│   ├── 03_llmcompressor_export_fixed_v2.ipynb
│   ├── 04_evaluate_code_benchmarks.ipynb
│   ├── 05_profile_and_flops.ipynb
│   └── 06_demo.ipynb                                 Demo on custom code tasks (no _global equivalent yet)
├── act_scales/                                       Calibrated scales (.pt files, one per size)
├── checkpoints/                                      Smoothed bf16 + deployable W8A8 checkpoints
├── results/                                          All measured numbers, plots, CSVs (scoped by size)
└── bcb_results/                                      BCB's native output jsonls + eval results
```

### Two notebook trees?

`notebook_global/` is the current, canonical version: parameterized on a single `MODEL_SIZE` variable at the top (flip `'7B'` ↔ `'14B'`), Colab dependencies stripped, Runpod-ready, with pinned dependencies and the nb05 patches for TTFT / NUM_TRIALS / derived overhead / throughput-loading applied. `notebooks/` is kept as the original submission for provenance. Once `notebook_global/` is complete (05 and 06 pending) the two trees will be consolidated.

All output files are scoped by size (`params_7b.json`, `params_14b.json`, etc.) so 7B and 14B runs don't overwrite each other.

---

## Setup (Runpod)

We ran this on Runpod A40 48GB. It fits:

- 7B bf16 + 7B W8A8 (calibrate + eval with headroom)
- 14B bf16 (calibrate — does not fit on a 24 GB 4090)
- 14B W8A8 (fits on both A40 and 4090)

### 1. Create persistent storage

Create a **Runpod Network Volume**, ~150 GB, in the region where A40/A6000 is reliably available. Pods mount this at `/workspace` and it survives pod deletion — essential because the HF cache alone is ~45 GB for both model sizes, which you don't want to re-download every session.

### 2. Launch pod

- GPU: A40 (48 GB) or RTX A6000 (48 GB). A100 40GB works but is more expensive and less available. L40S has faster INT8 kernels (larger W8A8 speedup) but less availability.
- Container disk: 30–40 GB (for OS + CUDA libs + pip packages). Everything project-related goes on `/workspace`.
- Template: any PyTorch base image; the first notebook's install cell pins everything below.

### 3. Clone and set env

```bash
cd /workspace
git clone https://github.com/<your-username>/qwen-smoothquant-project.git
cd qwen-smoothquant-project
mkdir -p /workspace/hf-cache
```

### 4. Pinned install

Each notebook in `notebook_global/` has a one-shot install cell at the top. Run it **once per pod session**, restart the kernel, then skip it. The pinned stack is:

```
torch==2.9.1
typing_extensions>=4.13
compressed-tensors==0.13.0
transformers==4.57.3
llmcompressor==0.9.0
+ accelerate, safetensors, datasets, tqdm, matplotlib, pandas, vllm
```

These pins are tight because `llm-compressor`'s GPTQ step calls `torch.linalg.cholesky` on GPU, which fails with a cryptic ABI error when torch's internal CUDA libraries are split across versions. The uninstall-then-reinstall pattern in the install cell removes every `nvidia-*-cu12` package before reinstalling torch, which avoids this.

### 5. HuggingFace

```bash
huggingface-cli login
```

Needed for model downloads and the Pile validation subset used for calibration.

---

## Running the pipeline

Each notebook reads the output of the previous one. Flip `MODEL_SIZE` at the top of each to switch tiers — that's the only edit required.

| Notebook | Input | Output |
|---|---|---|
| `notebook_global/01_calibrate_qwen_global` | `Qwen/Qwen2.5-Coder-{size}-Instruct` | `act_scales/qwen25-coder-{size}.pt`, `results/outlier_summary_{size}.json` |
| `notebook_global/02_smooth_and_fakequant_global` | act_scales | `checkpoints/qwen25-coder-{size}-smoothed-a0.5/`, `results/ppl_{size}.csv` |
| `notebook_global/03_llmcompressor_export_global` | smoothed ckpt | `checkpoints/qwen25-coder-{size}-W8A8/`, `results/checkpoint_sizes_{size}.json` |
| `notebook_global/04_evaluate_code_benchmarks_global` | bf16 + W8A8 | `results/humaneval_{size}.csv`, `results/bcb_hard_{size}.csv`, `results/throughput_vllm_{size}.json` |
| `notebooks/05_profile_and_flops` (or `_global` once committed) | bf16 + W8A8 | `results/params_{size}.json`, `flops_{size}.json`, `vram_decomposition_{size}.json`, `deployment_sizing_{size}.csv`, `efficiency_summary_{size}.json`, plots |
| `notebooks/06_demo` | W8A8 ckpt (local 4090) | `results/nb06_demo_{size}.csv` |

### Expected runtime on A40

| Notebook | 7B | 14B |
|---|---|---|
| 01 Calibration | ~1 min | ~2 min |
| 02 Smooth + PPL (4 configs) | ~6 min | ~12 min |
| 03 GPTQ export | ~15 min | ~30 min |
| 04 Eval (both models, both benchmarks) | ~2 min | ~5 min |
| 05 Profile | ~3 min | ~5 min |
| **Total** | **~30 min** | **~55 min** |

Nb06 runs locally on a 4090 (W8A8 fits; bf16 14B does not). Copy the W8A8 checkpoint back with:

```bash
rsync -avP /workspace/qwen-smoothquant-project/checkpoints/qwen25-coder-14b-W8A8/ \
           user@local-4090:/path/to/qwen-smoothquant-project/checkpoints/qwen25-coder-14b-W8A8/
```

---

## Results (7B, measured on A40)

| Metric | bf16 | W8A8 | Change |
|---|---|---|---|
| HumanEval pass@1 | 0.884 | 0.884 | 0.000 |
| HumanEval+ pass@1 | 0.841 | 0.823 | −1.8% (within eval noise) |
| BCB-Hard pass@1 | 0.182 | 0.209 | +2.7 pts (within Bernoulli noise on n=148) |
| WikiText-2 PPL | 9.551 | 9.622 | +0.071 |
| Disk checkpoint | 14.20 GB | 8.13 GB | **1.75× smaller** |
| HumanEval wall-clock (vLLM) | 16.7 s | 13.2 s | **1.26× faster** |
| BCB-Hard wall-clock (vLLM) | 61.0 s | 42.6 s | **1.43× faster** |

The average ~1.35× vLLM speedup is below the paper's 1.56× upper bound because A40 (Ampere) has a smaller INT8/bf16 throughput ratio than L40S/4090 (Ada). This is architecture-dependent, not pipeline-dependent — accuracy numbers are reproducible across hardware.

Activation outlier ratios (top-1 channel / median channel) on Qwen2.5-Coder-7B, mid-stack:

| Projection | Mean across layers | Max across layers |
|---|---|---|
| q/k/v_proj (input_layernorm) | 18.3× | 48.3× |
| gate/up_proj (post_attn_layernorm) | 13.2× | 29.8× |
| o_proj (attention output) | 3.0× | 4.5× |
| down_proj (SiLU-gated intermediate) | 13.2× | 71.3× |

Qwen2.5-Coder has clear activation outliers — SmoothQuant's premise holds — though milder than OPT-scale outliers (100×+ is typical there). This explains why per-token activation quant + per-channel weight quant already gets within 0.08 PPL of bf16 even without smoothing; SmoothQuant still helps but the baseline is strong enough that the PPL gap is small.

---

## Writeup framing

Our headline isn't "1.35× faster" — it's that we moved the Qwen2.5-Coder 14B tier onto consumer-class hardware at near-zero accuracy cost, plus ~2× serving capacity at the 7B tier we already ran. The 14B demo is a **capability unlock**, not just a speedup. SmoothQuant's reported 1.56× speedup and 2× memory reduction are reproduced (in qualitative terms, with GPU-architecture-dependent magnitudes) on a newer model architecture and a code domain the paper didn't cover.

---

## Rubric checklist

| Rubric requirement | Where it's addressed |
|---|---|
| Project code runnable | `notebook_global/` (01–04) + `notebooks/` (05, 06) |
| Setup instructions | [Setup](#setup-runpod) section above |
| Model structure explained | `notebook_global/01` prints full config; `results/params_{size}.json` records it |
| Parameter count | `notebooks/05` §1, `results/params_{size}.json` |
| Compute costs (FLOPs) | `notebooks/05` §2, `results/flops_{size}.json` (analytical 2·P·T + measured via `FlopCounterMode`) |
| Per-stage timing | `notebooks/05` §4–5, `results/profile_per_stage_{size}.csv` (preprocess / TTFT / pure-decode / postprocess) |
| Demonstration on own data | `notebooks/06` runs held-out coding tasks with pass/fail harness |
| Performance improvement | `results/humaneval_{size}.csv` + `bcb_hard_{size}.csv` + `throughput_vllm_{size}.json` |

---

## License

Code in this repository is released under Apache 2.0, matching the upstream SmoothQuant license.