## FT2: First-Token-Inspired Online Fault Tolerance on Critical Layers for Generative LLMs

> **Important:** Some experiments rely on a custom build of PyTorch contained in `package/pytorch`. Before running the scripts in this directory, build and install this local PyTorch version as you would build a standard PyTorch from source (no special compilation flags are required) and make sure your environment uses this build.

This directory contains code for FT2, a fault-injection characterization and online fault tolerance framework for generative Large Language Models (LLMs). FT2 selectively protects critical layers during inference by profiling neuron value bounds from the first token generation and applying range restriction to the following token generations. Experiments are implemented in PyTorch and Hugging Face Transformers, and cover seven LLMs (GPTJ-6B, OPT-6.7B, OPT-2.7B, Llama2-7B, Vicuna-7B, Qwen2-7B, Qwen2-1.5B) across three datasets (SQuAD 2.0, XTREME, GSM8K) under three fault models (1-bit flip, 2-bit flip, EXP bit flip).

- **`sigcode/modeling/`**: Customized model definitions with critical layer identification and FT2 protection hooks.
- **`sigcode/evaluation/`**: Fault-injection and performance evaluation scripts that measure SDC rate and runtime overhead under various fault models and protection configurations.

### `sigcode/modeling/`

This folder provides model implementations based on the official Hugging Face model classes, modified to support FT2's critical layer protection logic. Critical layers are those for which no scaling operation or activation layer is present before the next linear layer (e.g., V_PROJ, OUT_PROJ, FC2, UP_PROJ, DOWN_PROJ). The `_protected` variants instrument these layers with `torch.clamp`- and `torch.nan_to_num`-based range restriction, using bounds profiled during the first token generation (scaled by a factor of 2×) to protect subsequent token generations.

Main files:
- **`modeling_gptj.py` / `modeling_gptj_protected.py`**: GPTJ-6B causal language model and its FT2-protected variant, with critical layer hooks applied to V_PROJ, OUT_PROJ, and FC2.
- **`modeling_llama.py` / `modeling_llama_protected.py`**: Llama2-7B / Vicuna-7B model implementations, including attention, SiLU activation, rotary embeddings, KV caches, and a protected variant covering V_PROJ, OUT_PROJ, UP_PROJ, and DOWN_PROJ.
- **`modeling_opt.py` / `modeling_opt_protected.py`**: OPT-6.7B / OPT-2.7B model implementations and their protected variants, with critical layer coverage on V_PROJ, OUT_PROJ, and FC2.
- **`modeling_qwen2.py` / `modeling_qwen2_protected.py`**: Qwen2-7B / Qwen2-1.5B model implementations and protected variants, covering V_PROJ, OUT_PROJ, UP_PROJ, and DOWN_PROJ as critical layers.

These files define:
- Model architectures (attention, MLP, normalization, rotary embeddings, etc.).
- Forward passes for standard causal language modeling and related tasks.
- First-token bound profiling logic and per-layer range restriction hooks used by the FT2 protection methodology (especially in the `_protected` variants).

You typically do **not** run these files directly. They are imported by the evaluation scripts to construct the appropriate model class.

### `sigcode/evaluation/`

This folder contains scripts that run fault-injection campaigns and measure the SDC rate and runtime overhead of FT2 and baseline methods (Ranger, MaxiMals, Global Clipper) on the models defined in `sigcode/modeling/`. Fault injection is performed via PyTorch forward hooks that inject single- or multi-bit flips into randomly selected neurons of linear layers in decoder blocks, following the same statistical methodology as the paper (500 fault injection trials per input, 50 inputs per configuration).

Typical scripts:
- Files named **`<model>.py`** (e.g., `gptj.py`, `llama.py`, `qwen.py`, `opt.py`) run baseline (unprotected or Ranger/MaxiMals/Global Clipper) fault injection experiments and measure SDC rate and latency overhead.
- Corresponding **`*protect.py`** variants (e.g., `gptjprotect.py`, `llamaprotect.py`) enable FT2-protected execution: bounds are recorded during first token generation and applied as range restrictions on critical layers for all subsequent tokens.
- Additional `*xt.py` and `*gsm.py` variants run evaluations on the XTREME (cross-lingual QA) and GSM8K (math problem solving) datasets, respectively.

Common characteristics:
- Use Hugging Face `AutoTokenizer` / `AutoModelForCausalLM` or the custom classes from `sigcode/modeling`.
- Load evaluation datasets (SQuAD 2.0, XTREME, GSM8K) via `datasets`; 50 inputs are randomly selected from those answered correctly by all models in fault-free execution.
- Register forward hooks to inject single-bit flips, double-bit flips, or exponent-bit flips into randomly selected neurons of linear layers.
- Generate 60 output tokens for QA tasks and 180 tokens for the math task per inference; classify outcomes as Masked or SDC based on semantic correctness of the generated answer.
- Measure latency and SDC rate, writing outputs and logs to the `performance/output/` directory.

You normally run these scripts from the project root, for example:

```bash
cd performance/sigcode/evaluation
python gptj.py
```

Depending on the script, you may need to adjust:
- CUDA-visible devices (`CUDA_VISIBLE_DEVICES` environment variable or the setting inside the script).
- Paths to model checkpoints and datasets.
- Fault-injection configuration (fault model selection, bit positions, target layers, number of trials per input, etc.).

### Dependencies and Environment

The scripts assume:
- **Python** with **PyTorch** (custom build from `package/pytorch`), **Transformers**, and **Datasets** installed.
- Access to at least one **CUDA-capable GPU**; experiments in the paper were conducted on NVIDIA A100 (AMD EPYC 7742 CPU, Rocky Linux 8.10) and NVIDIA H100 via GH200 Grace Hopper Superchip (NVIDIA Grace 72-core CPU, Rocky Linux 9.3).

Before running evaluations, ensure the corresponding model checkpoints and datasets are downloaded and available, or modify the scripts to point to your local paths.
