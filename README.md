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
git clone https://github.com/yourusername/CriticPilot.git
cd CriticPilot

# Clone GenPilot as a subdirectory (official code)
git clone https://github.com/27yw/GenPilot.git genpilot

# Install dependencies
pip install -r requirements.txt   # your own requirements
pip install -r genpilot/requirements.txt
```

## Quick Start

```python
from criticpilot import CriticPilot

pilot = CriticPilot(
    mllm_model="qwen2.5-vl-72b",   # or gpt-4o / gemini-2.0-pro
    t2i_model="flux-schnell",
    max_iterations=5
)

best_prompt, final_image = pilot.optimize(
    "A red apple on a wooden table, soft morning light, shallow depth of field"
)
```

See experiments/demo.ipynb for more examples.

## Project Structure

```text
CriticPilot/
├── criticpilot/           # Your code (Critic modules)
│   ├── __init__.py
│   ├── critic_checkpoints.py
│   └── backtracking.py
├── genpilot/              # Official GenPilot (submodule / subdirectory)
├── experiments/           # Testing scripts
│   ├── baseline_test.py
│   └── critic_ablation.py
├── results/               # Outputs and logs
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