# TESOLLO RL

![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![OS](https://img.shields.io/badge/OS-Ubuntu_22.04-E95420?style=flat-square&logo=ubuntu)
![Isaac Lab](https://img.shields.io/badge/NVIDIA-Isaac_Lab-76B900?style=flat-square&logo=nvidia)

**Learn in-hand manipulation with a TESOLLO delto hand in NVIDIA Isaac Lab.**

TESOLLO RL provides a reinforcement learning example for the **DG-5F-M**. In this example, a simulated hand learns to rotate a cylindrical object while keeping it near a reference point on the hand.

You can inspect the simulation, run a compatible trained policy, train a policy from scratch, and modify the task. **A physical robot hand is not required.** The commands in this repository run in simulation; deployment to physical hardware is outside the scope of this example.

The DG-5F-M has five fingers and 20 independently actuated joints. For hardware specifications and product documentation, see the [DG-5F-M product page](https://www.tesollo.com/products/grippers/dg-5f-m).

| Training (Simulation) | Deployment (Real World) |
| :---: | :---: |
| ![Training Demo](images/rl_train.gif) | ![Deployment Demo](images/rl_deploy.gif) |
| *Cylinder Z-rotation learning in Isaac Ecosystem* | *Real-world deployment on Delto Gripper 5F (DG-5F)* |


## Contents

- [Quick start](#quick-start)
- [What this example includes](#what-this-example-includes)
- [New to reinforcement learning?](#new-to-reinforcement-learning)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run a trained policy](#run-a-trained-policy)
- [Train your own policy](#train-your-own-policy)
- [Understand the task](#understand-the-task)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Support](#support)
- [License and acknowledgments](#license-and-acknowledgments)

## Quick start

Complete [Installation](#installation) once before using these commands. This section shows the sequence from opening the simulation to running a trained policy.

### 1. Prepare your terminal

In each new terminal, activate your Isaac Lab environment and enter the repository root. Replace the environment name and `/path/to/tesollo_rl` with your own values:

```bash
conda activate env_isaaclab
cd /path/to/tesollo_rl
```

For an installation using Isaac Lab's bundled Python, replace `python` below with `/path/to/IsaacLab/isaaclab.sh -p`.

### 2. Open the simulation

```bash
python scripts/random_agent.py \
  --task Isaac-Tesollo-Rotate-v0 \
  --num_envs 4
```

A window should show four simulated hands and cylinders. The fingers move randomly, so the cylinders may fall. This checks that the environment loads; it does not run a learned policy. Press `Ctrl+C` to stop this process before continuing.

### 3. Train a policy

If you already have a compatible training checkpoint, skip to step 4. Otherwise, start training:

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Tesollo-Rotate-v0 \
  --seed 42 \
  --logger tensorboard \
  --headless
```

Training progress appears in the terminal, and checkpoints are saved under `logs/rsl_rl/dg5fm_rotate/<run-directory>/`. No simulation window opens because `--headless` is enabled.

This command uses the configured defaults of 1,028 environments and 10,000 iterations. For a short installation check before a full training run, follow [Check the training pipeline](#1-check-the-training-pipeline). For memory settings and learning curves, see [Train your own policy](#train-your-own-policy).

### 4. Run the saved policy

Replace `RUN_DIRECTORY` and `ITERATION` with an existing run directory and checkpoint number:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Tesollo-Rotate-Play-v0 \
  --checkpoint logs/rsl_rl/dg5fm_rotate/RUN_DIRECTORY/model_ITERATION.pt \
  --num_envs 1
```

A window opens with one hand controlled by the saved policy. A successfully trained policy should rotate and retain the cylinder. If your checkpoint is stored elsewhere, pass its actual `.pt` path to `--checkpoint`.

Use the training task ID with `train.py` and the `-Play-v0` task ID with `play.py`. Press `Ctrl+C` to stop playback. To save a video, follow [Record a video](#record-a-video).

## What this example includes

- A DG-5F-M right-hand asset and a cylindrical object asset.
- An Isaac Lab manager-based, single-agent environment.
- A PPO training configuration for RSL-RL.
- Scripts for inspecting the environment, training, and policy playback.
- Separate environment configurations for training and playback.

| Task ID | Purpose |
| --- | --- |
| `Isaac-Tesollo-Rotate-v0` | Train a policy or inspect the training environment. |
| `Isaac-Tesollo-Rotate-Play-v0` | Run a compatible saved policy with the playback configuration. |

A pretrained checkpoint is not bundled with the source code. If a checkpoint is available for your release, obtain it from the repository's **Releases** tab. Otherwise, follow [Train your own policy](#train-your-own-policy).

## New to reinforcement learning?

Reinforcement learning trains a controller through repeated interaction with an environment. The controller tries actions, receives rewards, and updates its behavior to improve the task objective.

| Term | Meaning in this example |
| --- | --- |
| **Environment** | The simulated hand, cylinder, physics, and task rules. |
| **Observation** | Information given to the policy: normalized joint positions and the previous action. |
| **Action** | A command that is mapped to joint-position targets within the hand's joint limits. |
| **Reward** | A score encouraging rotation and object retention, while penalizing unwanted motion. |
| **Policy** | The neural network that converts observations into actions. |
| **Checkpoint** | A saved training model that can be loaded for playback or further training. |

The software stack has three main parts:

- **Isaac Sim** provides simulation and rendering.
- **Isaac Lab** defines the environment and coordinates simulation-based learning.
- **RSL-RL** trains the policy using **Proximal Policy Optimization (PPO)**.

The current policy uses joint information and its previous action. Camera images and tactile measurements are not part of the configured policy input.

## Requirements

The commands below use **Linux and Bash**. Start with a working **Isaac Lab v2.3.2** installation and its compatible Isaac Sim, Python, and RSL-RL dependencies.

Follow the [Isaac Lab v2.3.2 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html). Use the hardware and driver requirements for the Isaac Sim version you install.

You will need:

- An NVIDIA GPU and driver supported by that Isaac Sim version.
- A Python environment in which Isaac Lab and the RSL-RL training dependencies work.
- Git to clone this repository.
- A graphical desktop for interactive playback, or headless execution for training.
- Internet access for installation and any NVIDIA assets referenced by the USD files that are not already cached.

### Validated configuration

This example was tested with the following configuration. 
These versions describe the tested setup, not minimum system requirements.

| Component | Version or configuration |
| --- | --- |
| Isaac Lab | `2.3.2` |
| Isaac Sim | `5.1.0` |
| Operating system | `Ubuntu 22.04` |
| Python | `3.11` |
| NVIDIA driver | `580.178.04` |

## Installation

### 1. Activate your Isaac Lab environment

Use the environment created during the Isaac Lab installation. For example, if you named your Conda environment `env_isaaclab`:

```bash
conda activate env_isaaclab
```

Replace the environment name if yours is different. For a virtual environment or uv installation, use its corresponding activation procedure.

### 2. Clone and install TESOLLO RL

Copy this repository's clone URL from GitHub's **Code** button.
```bash
git clone https://github.com/tesollodelto/tesollo_rl.git
cd tesollo_rl
python -m pip install -e source/tesollo_rl
```

The editable installation makes the `tesollo_rl` Python package available while allowing you to edit its source files.

**Run all remaining commands from the repository root**, the directory containing this README and `scripts/`.

If your Isaac Lab installation uses its bundled Python launcher, replace `python` in the examples with `/path/to/IsaacLab/isaaclab.sh -p`.

### 3. Verify environment registration

```bash
python scripts/list_envs.py
```

The output should include:

```text
Isaac-Tesollo-Rotate-v0
Isaac-Tesollo-Rotate-Play-v0
```

### 4. Inspect the simulation

Run four environments with random actions:

```bash
python scripts/random_agent.py \
  --task Isaac-Tesollo-Rotate-v0 \
  --num_envs 4
```

**Expected result:** a simulation window opens with the hand and object assets. The fingers receive random commands. The object may fall; this script checks the environment and does not demonstrate a trained policy.

Stop the script with `Ctrl+C` before starting another simulation.

## Run a trained policy

Use a checkpoint trained for this task, hand model, and compatible code version. The observation order, action mapping, normalization, and network configuration must match the checkpoint.

If a pretrained checkpoint is provided in a release, follow its release notes and save the checkpoint as `checkpoints/rotate/model.pt`. Create the directory if needed:

```bash
mkdir -p checkpoints/rotate
```

After placing the checkpoint at that path, run:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Tesollo-Rotate-Play-v0 \
  --checkpoint checkpoints/rotate/model.pt \
  --num_envs 1
```

**Expected result:** the simulation opens with one hand controlled by the saved policy. A successfully trained rotation policy should keep the cylinder near the hand while rotating it. Playback does not perform additional learning.

The playback configuration disables observation noise and the episode time limit. The object-drop reset condition remains active.

The script also exports policy files to an `exported/` directory beside the checkpoint. Use the training checkpoint, rather than an exported inference file, with `--checkpoint`.

### Record a video

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Tesollo-Rotate-Play-v0 \
  --checkpoint checkpoints/rotate/model.pt \
  --num_envs 1 \
  --video \
  --video_length 200 \
  --headless
```

The script enables camera rendering when `--video` is set. Videos are saved under `videos/play/` beside the checkpoint. `--video_length` is measured in simulation control steps, not seconds.

## Train your own policy

### 1. Check the training pipeline

Start with a short run to check environment creation, rollout collection, and policy updates:

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Tesollo-Rotate-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --seed 42 \
  --logger tensorboard \
  --headless
```

This is a pipeline check. Ten iterations are not intended to produce a successful rotation policy.

### 2. Start a training run

The following command uses the environment count and training duration from the checked-in configuration:

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Tesollo-Rotate-v0 \
  --seed 42 \
  --logger tensorboard \
  --headless
```

The current defaults are **1,028 parallel environments** and **10,000 training iterations**. These are configuration defaults, not a claim that a particular performance level will be reached. Training time and memory use depend on the GPU, software versions, and configuration.

If GPU memory is limited, specify a smaller environment count, such as `--num_envs 128`. Changing the environment count also changes the amount of rollout data collected per iteration, so the learning behavior may change.

### 3. Monitor training

Training outputs are written to:

```text
logs/rsl_rl/dg5fm_rotate/<run-directory>/
```

Each run contains model checkpoints, logs, and the environment and agent configurations under `params/`.

To inspect TensorBoard logs, open another terminal in the same Python environment and repository directory:

```bash
tensorboard --logdir logs/rsl_rl/dg5fm_rotate
```

Open the URL printed by TensorBoard. Inspect the learning curves and play the resulting policy to evaluate its behavior; total reward alone does not fully describe rotation quality or object retention.

### 4. Play your checkpoint

Replace `RUN_DIRECTORY` and `ITERATION` with the names from your training output:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Tesollo-Rotate-Play-v0 \
  --checkpoint logs/rsl_rl/dg5fm_rotate/RUN_DIRECTORY/model_ITERATION.pt \
  --num_envs 1
```

For reproducible experiments, keep the checkpoint, `params/env.yaml`, `params/agent.yaml`, seed, code commit, asset version, and software versions together. Matching a seed alone does not guarantee identical training results across different systems.

### Common command-line options

| Option | Purpose |
| --- | --- |
| `--task` | Select the registered training or playback environment. |
| `--num_envs` | Set the number of parallel simulation environments. |
| `--max_iterations` | Set the number of training iterations; used with `train.py`. |
| `--seed` | Set the random seed. |
| `--headless` | Run without an interactive simulation window. |
| `--checkpoint` | Select a checkpoint for playback. |
| `--video` | Record rendered frames during execution. |

## Understand the task

The objective is continuous rotation of a cylinder around the world Z axis while keeping the object near a hand reference point.

| Setting | Current configuration |
| --- | --- |
| Robot | DG-5F-M right hand, `DG5FM_RIGHT_CFG` |
| Object | Cylinder |
| Workflow | Manager-based, single-agent RL |
| Algorithm | PPO through RSL-RL |
| Policy input | Joint-limit-normalized joint positions, then the previous action |
| Policy output | Joint-position commands mapped to joint limits, with exponential smoothing |

The reward combines positive rotation about world Z and proximity to the hand reference point with penalties for off-axis rotation, joint deviation, joint speed, object drops, and abrupt action changes.

Training adds noise to joint-position observations and randomizes the object's friction. The current reset position and velocity ranges are zero-width, so they do not provide a broad randomized reset distribution.


### Modify the example

| What you want to change | File to edit |
| --- | --- |
| Robot USD, initial pose, or actuator settings | [`assets/robots/tesollo.py`](source/tesollo_rl/tesollo_rl/assets/robots/tesollo.py) |
| Scene, observations, actions, reward weights, resets, or termination conditions | [`rotate_cylinder_env_cfg.py`](source/tesollo_rl/tesollo_rl/tasks/manager_based/rotate/rotate_cylinder_env_cfg.py) |
| Custom reward calculations | [`mdp/rewards.py`](source/tesollo_rl/tesollo_rl/tasks/manager_based/rotate/mdp/rewards.py) |
| Custom reset behavior | [`mdp/events.py`](source/tesollo_rl/tesollo_rl/tasks/manager_based/rotate/mdp/events.py) |
| Hand selection or playback overrides | [`tesollo_rotate_env_cfg.py`](source/tesollo_rl/tesollo_rl/tasks/manager_based/rotate/config/tesollo_hand/tesollo_rotate_env_cfg.py) |
| PPO hyperparameters or network dimensions | [`rsl_rl_ppo_cfg.py`](source/tesollo_rl/tesollo_rl/tasks/manager_based/rotate/config/tesollo_hand/agents/rsl_rl_ppo_cfg.py) |

Keep USD assets together with their referenced `configuration/` files. Asset paths are resolved relative to the installed package through `ASSET_DATA_DIR`.

Changing the robot, observation layout, action mapping, or network architecture can make existing checkpoints incompatible. Record these changes and train a matching policy when needed.

## Project structure

```text
tesollo_rl/
├── README.md
├── scripts/
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   └── rsl_rl/
│       ├── train.py
│       ├── play.py
│       └── cli_args.py
└── source/tesollo_rl/
    ├── setup.py
    ├── pyproject.toml
    ├── config/extension.toml
    └── tesollo_rl/
        ├── assets/
        │   ├── robots/tesollo.py
        │   └── data/
        │       ├── robots/dg5f_right/
        │       └── objects/cylinder/
        └── tasks/manager_based/rotate/
            ├── rotate_cylinder_env_cfg.py
            ├── mdp/
            └── config/tesollo_hand/
                ├── tesollo_rotate_env_cfg.py
                └── agents/rsl_rl_ppo_cfg.py
```


## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError: tesollo_rl` | Activate the Isaac Lab environment and run `python -m pip install -e source/tesollo_rl` from the repository root. |
| Missing `isaaclab`, `isaacsim`, or `rsl_rl` modules | Confirm that the selected Python environment contains the Isaac Lab installation and its RSL-RL dependencies. |
| Task not found | Run `python scripts/list_envs.py`, check the exact task ID, and use this repository's scripts, which import `tesollo_rl.tasks`. |
| GPU out of memory | Stop other simulation processes and reduce `--num_envs`. Rendering, physics buffers, and learning also consume memory. |
| A USD file cannot be loaded | Confirm that the asset and its referenced files are present. Check connectivity if a referenced NVIDIA asset must be downloaded. |
| The object falls in `random_agent.py` | Random actions are not a trained controller. Use a compatible checkpoint to evaluate learned behavior. |
| A checkpoint produces shape errors or unexpected behavior | Check the hand variant, code version, observation/action definitions, normalization, and agent configuration used during training. |
| No simulation window appears | Remove `--headless` and use a machine with a working graphical session. |
| Policy export fails during playback | Check the reported export error and the ONNX/export dependencies in the Isaac Lab environment. Playback exports policy files before entering its simulation loop. |


## License and acknowledgments

This example builds on:

- [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
