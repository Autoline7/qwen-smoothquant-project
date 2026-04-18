# SmoothQuant W8A8 Quantization of Qwen2.5-Coder

Course final project implementing the **SmoothQuant** post-training quantization algorithm (Xiao et al., ICML 2023) on the **Qwen2.5-Coder-Instruct** 7B and 14B models, with end-to-end evaluation on HumanEval and BigCodeBench-Hard.

> **Attribution.** This project builds on:
> - SmoothQuant — Xiao, Lin, Seznec, Wu, Demouth, Han. _SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models._ ICML 2023. [[paper]](https://arxiv.org/abs/2211.10438) [[code]](https://github.com/mit-han-lab/smoothquant) (MIT License)
> - Qwen2.5-Coder — Hui et al. _Qwen2.5-Coder Technical Report._ 2024. [[models]](https://huggingface.co/Qwen)
> - EvalPlus (HumanEval+) — Liu et al. NeurIPS 2023. [[code]](https://github.com/evalplus/evalplus)
> - BigCodeBench — Zhuo et al. ICLR 2025. [[code]](https://github.com/bigcode-project/bigcodebench)

---

## What this project does

SmoothQuant enables INT8 quantization of both weights *and* activations (W8A8) in transformer LLMs by transferring the quantization difficulty of activation outliers into the weights via a per-channel smoothing scale. This gives ~2× memory reduction and up to ~1.5× inference speedup, with negligible accuracy loss.

We implement SmoothQuant end-to-end for the **Qwen2 architecture** (the original repo officially supports Llama/Mistral/Mixtral, not Qwen2) and measure the effect on code generation benchmarks for two model sizes:

- `Qwen/Qwen2.5-Coder-7B-Instruct` (7.62B params)
- `Qwen/Qwen2.5-Coder-14B-Instruct` (14.77B params)

We evaluate on two code benchmarks:

- **HumanEval+** (164 problems, EvalPlus-augmented tests)
- **BigCodeBench-Hard-Instruct** (148 problems, harder tasks with complex library calls)

---

## Repository structure

```
qwen-smoothquant-project/
├── README.md                           This file
├── REPORT.md                           Writeup: results, plots, discussion
├── requirements.txt
├── src/
│   ├── qwen_smooth.py                  SmoothQuant smoothing adapted for Qwen2
│   ├── qwen_fake_quant.py              W8A8 fake-quant linear for Qwen2
│   ├── calibrate.py                    Activation scale calibration
│   └── profile_utils.py                Per-stage timing + FLOPs counting
├── notebooks/
│   ├── 01_calibrate_qwen.ipynb         Generate activation scales on Pile
│   ├── 02_smooth_and_fakequant.ipynb   Apply SmoothQuant (path A: MIT repo)
│   ├── 03_llmcompressor_export.ipynb   Export W8A8 checkpoint (path B: vLLM)
│   ├── 04_evaluate_humaneval.ipynb     HumanEval+ before/after
│   ├── 05_evaluate_bcb.ipynb           BigCodeBench-Hard before/after
│   ├── 06_profile_and_flops.ipynb      Per-stage timing + FLOPs
│   ├── 07_alpha_ablation.ipynb         Sweep alpha on 7B
│   └── 08_demo.ipynb                   Personal-code generation demo
├── act_scales/                         Calibrated activation scales (.pt files)
└── results/                            All measured numbers, plots, CSVs
```

---

## Rubric checklist

| Rubric requirement | Where it's addressed |
|---|---|
| Project code runnable | This repo + notebooks 01–08 |
| GitHub fork of source | Each member forks `mit-han-lab/smoothquant`; this repo references both |
| Setup instructions step-by-step | [Setup](#setup) section below |
| Model structure explained | `REPORT.md` §2 + `notebooks/01` |
| Parameter count | Printed by `notebooks/01` and recorded in `results/param_counts.csv` |
| Compute costs (FLOPs) | `notebooks/06` + `results/flops_table.csv` |
| Per-stage timing | `notebooks/06` breaks out tokenization / prefill / decode / detokenization |
| Demonstration on own data | `notebooks/08` runs on a function from our own code |
| Performance improvement | `results/accuracy_table.csv` + `results/speed_table.csv` show bf16 vs W8A8 |

---

## Setup

Tested on Google Colab Pro with A100 40GB or L4 22GB. 14B bf16 baseline requires A100.

### 1. Clone this repo + the upstream SmoothQuant repo

```bash
git clone https://github.com/<your-username>/qwen-smoothquant-project.git
cd qwen-smoothquant-project
git clone https://github.com/mit-han-lab/smoothquant.git external/smoothquant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# also install the smoothquant package itself (editable)
pip install -e external/smoothquant
```

### 3. Log in to HuggingFace

```bash
huggingface-cli login
```

You'll need this for both model downloads and dataset downloads (the Pile subset for calibration).

### 4. Run notebooks in order

The notebooks are numbered and should be run sequentially. Each notebook:

- Lists its prerequisites at the top
- Writes its output to `act_scales/`, `results/`, or a checkpoint directory on Drive
- Can be re-run independently once its prerequisites are satisfied

If running on Colab, mount Drive first and put this project folder inside `/content/drive/MyDrive/` so intermediate results survive runtime disconnects.

### Expected runtime

| Notebook | 7B | 14B |
|---|---|---|
| 01 Calibration | ~8 min | ~15 min |
| 02 Smooth + fake-quant | ~2 min | ~3 min |
| 03 llm-compressor export | ~10 min | ~20 min |
| 04 HumanEval+ (both precisions) | ~25 min | ~40 min |
| 05 BCB-Hard (both precisions) | ~30 min | ~50 min |
| 06 Profile + FLOPs | ~10 min | ~10 min |
| 07 Alpha ablation (7B only) | ~60 min | — |
| 08 Demo | ~2 min | ~2 min |
| **Total** | **~2.5 hr** | **~2.5 hr** |

---

## Team member responsibilities

- **Member 1** — Quantization pipeline (notebooks 01, 02, 03, 07; `src/qwen_smooth.py`, `src/qwen_fake_quant.py`, `src/calibrate.py`)
- **Member 2** — Evaluation (notebooks 04, 05; result tables)
- **Member 3** — Profiling + demo (notebooks 06, 08; `src/profile_utils.py`; plots; presentation)

---

## License

Code in this repository is released under Apache 2.0, matching the upstream SmoothQuant license.
