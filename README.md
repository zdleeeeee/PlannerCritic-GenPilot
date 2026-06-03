# CriticPilot: GenPilot with Planner-Critic Mechanism

**Course**: Computer Graphics A (Spring 2026 Fudan University) – Project 3  
**Authors**: Zedong Li
**GitHub**: https://github.com/zdleeeeee/PlannerCritic-GenPilot

---

## Overview

CriticPilot is an extension of [GenPilot](https://github.com/27yw/GenPilot), a multi-agent system for test-time prompt optimization in text-to-image generation. 

**What we add**: A Planner-Critic mechanism that monitors the optimization pipeline, checks intermediate quality, and enables self-correction (retry → fallback → reset) when errors are detected.

**Motivation**: GenPilot’s linear feedforward pipeline lacks internal validation. As the original authors admit, VQA-based error detection can be unreliable in complex scenes. Our Critic addresses this gap.

---

## Key Features

- ✅ Inherits all GenPilot functionalities (error analysis, prompt refinement, memory)
- ✅ Adds 5 critical checkpoints throughout the pipeline (see `docs/checkpoints.md`)
- ✅ Implements graduated backtracking (L1局部重试 / L2模块回退 / L3全局重置)
- ✅ **No extra training** – works with the same API-based models as GenPilot

---

## Installation

```bash
# Clone this repository
git clone https://github.com/zdleeeeee/PlannerCritic-GenPilot.git
cd PlannerCritic-GenPilot

# Copy .env
cp .env.example .env

cp genpilot/error_analysis_pipline.sh.example genpilot/error_analysis_pipline.sh

# Install environment and dependencies
uv sync

# Download t2l model if needed
hf download black-forest-labs/FLUX.1-schnell --local-dir <your local download path>
```

> **Note**: We update the package `mkl-service` of genpilot's environment from 2.4.0 to 2.5.2 to fix the issue of `mkl-service` not working with Python 3.12.

## model

MLLM: Qwen3-VL-8B-Instruct

T2L: FLUX.1

## Run the Error Analysis

You need to modify `error_analysis_pipline.sh` to fill in your own config first.

```bash
chmod -x error_analysis_pipline.sh

./error_analysis_pipline.sh
```

## Run the Test-Time Prompt Optimization

### GenPilot(Baseline)

```bash
python run_baseline.py
```

### PlannerCritic-GenPilot

## Project Structure

```text
PlannerCritic-GenPilot/
├── PlannerCritic_GenPilot/
│   ├── __init__.py
│   ├── critic_checkpoints.py
│   └── backtracking.py
├── genpilot/              # Official GenPilot
├── run_baseline.py
├── data/                  # input folder
│   ├── ori_img/
│   └── original_prompts.txt
├── results/               # Outputs and logs
│   ├── baseline/
│   └── critic/
├── report/                # Course report & slides
└── README.md
```

## Experiments & Results
We evaluate CriticPilot on 20 prompts from DPG-bench and 5 custom complex prompts.

|Metric|GenPilot (baseline)|CriticPilot (ours)|Improvement|
|---|---|---|---|
|Final score (0-1)|?|?|?%|
|Convergence rounds|?|?|?%|
|Error detection accuracy|?%|?%|?%|

Detailed results and failure cases are in report/experiments.pdf.