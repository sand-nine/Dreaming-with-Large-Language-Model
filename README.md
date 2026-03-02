# Dreaming with Large Language Models (DLLM)

Code for the paper **"World Models with Hints from Large Language Models"**.

## Overview

Reinforcement learning struggles in the face of long-horizon tasks and sparse goals due to the difficulty in manual reward specification. While existing methods address this by adding intrinsic rewards, they may fail to provide meaningful guidance in long-horizon decision-making tasks with large state and action spaces, lacking purposeful exploration. Inspired by human cognition, we propose a new multi-modal model-based RL approach named **Dreaming with Large Language Models (DLLM)**. DLLM integrates the proposed hinting subgoals from the LLMs into the model rollouts to encourage goal discovery and reaching in challenging tasks. By assigning higher intrinsic rewards to samples that align with the hints outlined by the language model during model rollouts, DLLM guides the agent toward meaningful and efficient exploration. Extensive experiments demonstrate that the DLLM outperforms recent methods in various challenging, sparse-reward environments such as HomeGrid, Crafter, and Minecraft by 41.8%, 21.1%, and 9.9%, respectively.

## Project Structure

```
.
├── train.py              # Main training entry point
├── agent.py              # DLLM agent implementation
├── behaviors.py          # Agent behavior definitions
├── configs.yaml          # Configuration file for all environments
├── nets.py               # Neural network architectures
├── ninjax.py             # JAX-based neural network module system
├── jaxagent.py           # JAX agent wrapper
├── jaxutils.py           # JAX utility functions
├── expl.py               # Exploration module
├── Dockerfile            # Docker environment setup
├── embodied/             # Core embodied RL framework
│   ├── core/             # Core utilities (config, logging, replay, etc.)
│   ├── envs/             # Environment wrappers (Crafter, Atari, DMC, Minecraft, etc.)
│   ├── replay/           # Experience replay implementations
│   ├── run/              # Training and evaluation loops
│   ├── gpt_api.py        # LLM API integration for hint generation
│   ├── rnd.py            # Random Network Distillation
│   └── dicts.py          # Dictionary utilities
└── scripts/              # Training launch scripts
    ├── crafter.sh
    └── crafter_{0-3}.sh  # Multi-GPU training with different seeds
```

## Quick Start

### Training on Crafter

```bash
# Single GPU
bash scripts/crafter.sh

# Multi-GPU with different seeds
bash scripts/crafter_0.sh  # GPU 0, seed 0
bash scripts/crafter_1.sh  # GPU 1, seed 1
bash scripts/crafter_2.sh  # GPU 2, seed 2
bash scripts/crafter_3.sh  # GPU 3, seed 3
```

### Custom Training

```bash
python train.py \
  --run.script train_eval \
  --logdir ~/logdir/my_experiment \
  --configs crafter
```

## Tips

This codebase is forked from a co-author's renovated implementation: https://github.com/ibisbill/World_Models_w_Hints_from_LLMs
