# CriticPilot: GenPilot with Planner-Critic Mechanism

**Course**: Computer Graphics A (Spring 2026 Fudan University) – Project 3
**GitHub**: https://github.com/zdleeeeee/PlannerCritic-GenPilot

---

## Overview

CriticPilot is an extension of [GenPilot](https://github.com/27yw/GenPilot), a multi-agent system for test-time prompt optimization in text-to-image generation. 

**What we add**: A Planner-Critic mechanism that monitors the optimization pipeline, checks intermediate quality, and enables self-correction (retry → fallback → reset) when errors are detected.

**Motivation**: GenPilot’s linear feedforward pipeline lacks internal validation. As the original authors admit, VQA-based error detection can be unreliable in complex scenes. Our Critic addresses this gap.

---

## Key Features

- ✅ Inherits all GenPilot functionalities (error analysis, prompt refinement, memory)
- ✅ Implements error aggregator
- ✅ Implements graduated backtracking (L1 Local Retry / L3 Global Reset)
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

## Models

MLLM: Qwen3-VL-8B-Instruct

T2L: FLUX.1

## Run Error Analysis and Test-Time Prompt Optimization

```bash
python run_baseline.py
```

This scripts runs GenPilot with the baseline error analysis pipeline and CriticPilot with the new error analysis pipeline.

## Tests

We provide several test scripts to verify the correctness of the error analysis pipeline and the CriticPilot mechanism. You can run them using the following commands:

```bash
tests/test_error_taxonomy.py
tests/test_scorer_json.py
tests/test_compare_metrics.py
```

## Project Structure

```text
PlannerCritic-GenPilot
├── .env.example
├── .git
├── .gitignore
├── .gitmodules
├── .python-version
├── LICENSE
├── README.md
├── data
│   ├── original_prompts.txt
│   └── original_prompts_299.txt
├── genpilot/
├── optimize.md
├── pyproject.toml
├── run_baseline.py
├── tests
│   ├── test_compare_metrics.py
│   ├── test_error_taxonomy.py
│   └── test_scorer_json.py
└── uv.lock
```

## Experiments & Results
We evaluate baseline GenPilot and CriticPilot on 50 prompt generation tasks from DPG-bench.

| Batch | Baseline Avg | Critic Avg | Delta | L1 | L3 | Baseline scored rounds | Critic scored rounds | Scored rounds difference |
|-------|-------------|------------|-------|-----|-----|----------------------|---------------------|--------------------------|
| 1 | 13.5 | 13.605 | 0.105 | 13 | 4 | 28 | 38 | 10 |
| 2 | 13.87 | 14.316 | 0.446 | 3 | 1 | 23 | 19 | -4 |
| 3 | 13.37 | 13.871 | 0.501 | 15 | 4 | 35 | 31 | -4 |
| 4 | 12.288 | 13.13 | 0.842 | 8 | 13 | 81 | 59 | -22 |
| 5 | 12.69 | 13.39 | 0.700 | 4 | 4 | 24 | 24 | 0 |
| 6 | 13.85 | 13.817 | -0.033 | 9 | 5 | 60 | 60 | 0 |
| 7 | 13.211 | 13.628 | 0.417 | 17 | 3 | 38 | 43 | 5 |
| 8 | 13.588 | 14.045 | 0.457 | 11 | 3 | 51 | 44 | -7 |

Detailed results and failure cases are in report/experiments.pdf.