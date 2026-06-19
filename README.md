# Distilling Knowledge from Large Language Models into Lightweight Reinforcement Learning Agents for Autonomous Cyber Operations

This repository contains the artifact code for **Distilling Knowledge from Large Language Models into Lightweight Reinforcement Learning Agents for Autonomous Cyber Operations**. The project studies whether a cybersecurity-focused LLM can act as a teacher policy in CybORG and whether that policy can be distilled into a much smaller reinforcement learning agent for autonomous cyber defense.

The manuscript describes three main components:

- Prompt engineering for an 8B-parameter cybersecurity LLM teacher.
- Online LLM-to-RL policy distillation into a lightweight PPO-based agent.
- Transferability evaluation across CybORG scenarios ranging from 4 to 12 hosts.

## Repository Layout

```text
|-- condaConfig.yaml
|-- customWrapper.py
|-- Distillation/
|-- LLMIntegration/
|-- Transferability/
|-- CybORGModified/
```

- `condaConfig.yaml` defines the Python environment used for the experiments.
- `customWrapper.py` contains the custom wrapper for the state space, action space and LLM interaction.
- `LLMIntegration/LLMAugmenter.py` contains prompt construction, LLM inference, action extraction, and action/host distribution logic.
- `Distillation/distillingLLMDistIntoAgent.py` is the main policy-distillation experiment for the 13-host scenario.
- `Distillation/LLMDistillations/` contains saved distilled policy checkpoints for host-count transferability experiments.
- `Transferability/evaluatingTransferability.py` evaluates distilled agents across 4- to 12-host CybORG scenarios.
- `CybORGModified/` local submodule containing the modified CybORG environment and scenarios.

## Setup

Run this command from the repository root:

```bash
conda env create -f condaConfig.yaml
conda activate distillationEnv
```

After creating and activating the conda environment, initialize the CybORG submodule:

```bash
git submodule update --init --recursive
```

The environment file installs the local CybORG dependency with:

```text
-e ./CybORGModified
```

The main distillation script uses:

```text
scenario_13hosts_3subnets.yaml
```

The transferability script uses:

```text
scenario_4hosts_3subnets.yaml
scenario_5hosts_3subnets.yaml
scenario_6hosts_3subnets.yaml
scenario_7hosts_3subnets.yaml
scenario_8hosts_3subnets.yaml
scenario_9hosts_3subnets.yaml
scenario_10hosts_3subnets.yaml
scenario_11hosts_3subnets.yaml
scenario_12hosts_3subnets.yaml
```

## Hardware and Model Requirements

`condaConfig.yaml` installs a CUDA 12.1 PyTorch build:

```text
torch==2.5.0+cu121
```

The LLM teacher is loaded through Hugging Face Transformers. The default model in `LLMIntegration/LLMAugmenter.py` is:

```text
Vanessasml/cyber-risk-llama-3-8b
```

The code calls `.half()` on the loaded model, so the LLM-guided runs are intended for a CUDA-capable GPU environment. First-time execution may require internet access to download model weights unless the model is already present in the local Hugging Face cache.

## Running Policy Distillation

```bash
cd Distilation; python distillingLLMDistIntoAgent.py
```

The default entry point trains a PPO-style student agent with LLM assistance enabled. By default, the script writes the distilled policy checkpoint to:

```text
Distillation/LLMDistillations/distilledLLMHost13.pth
```

The manuscript reports distillation of the LLM teacher policy into a lightweight RL agent and describes a transition after the teacher-guided distillation phase to independent student-agent action selection.

## Running Transferability Evaluation

Run from the repository root:

```bash
python Transferability/evaluatingTransferability.py
```

This script evaluates host-count-specific distilled checkpoints across 4- to 12-host CybORG scenarios. It loads checkpoints from:

```text
Distillation/LLMDistillations/distilledLLMHost4.pth
Distillation/LLMDistillations/distilledLLMHost5.pth
Distillation/LLMDistillations/distilledLLMHost6.pth
Distillation/LLMDistillations/distilledLLMHost7.pth
Distillation/LLMDistillations/distilledLLMHost8.pth
Distillation/LLMDistillations/distilledLLMHost9.pth
Distillation/LLMDistillations/distilledLLMHost10.pth
Distillation/LLMDistillations/distilledLLMHost11.pth
Distillation/LLMDistillations/distilledLLMHost12.pth
```

These experiments correspond to the manuscript's transferability evaluation across network topologies.

## Saved Artifacts

The repository includes saved distilled student policies:

```text
Distillation/LLMDistillations/*.pth
```

These are PyTorch state dictionaries for the actor policy. They are intended to support evaluation without rerunning the full LLM distillation process.

## Repository Scope

This repository contains the code used for the primary LLM-to-RL distillation experiments described in the paper.

The teacher-guided reinforcement learning stabilization experiments discussed in Section IV-D ("Improving Post-Learning Performance"), including modifications such as critic-loss augmentation, pretrained critics, dynamic learning-rate schedules, additional critic epochs, distribution-based guidance, critic-freezing, and multiplicative teacher decay, are not included in this repository.

These experiments were implemented as modifications to the teacher-guided reinforcement learning framework introduced in prior work (https://arxiv.org/abs/2509.05311) and are therefore outside the scope of the artifact released here.

## Reproducibility Notes

The scripts are research entry points rather than a packaged command-line interface. Main experiment settings are defined near the bottom of each script under `if __name__ == "__main__"`.

Furthermore, the current scripts use stochastic training operations such as NumPy shuffling and PyTorch sampling. If exact run-to-run reproducibility is required, add explicit seeding for Python, NumPy, PyTorch, and the CybORG environment before running experiments.
