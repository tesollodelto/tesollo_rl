## Cylinder Z-Rotation

Experience the RL-based cylinder Z-rotation task in both simulation and real-world deployment.

| Training (Simulation) | Deployment (Real World) |
| :---: | :---: |
| ![Training Demo](images/rl_train.gif) | ![Deployment Demo](images/rl_deploy.gif) |
| *Cylinder Z-rotation learning in Isaac Ecosystem* | *Real-world deployment on Delto Gripper 5F (DG-5F)* |

### 📖 Task Overview
This task trains the Delto Gripper 5F (DG-5F-M) to perform dexterous in-hand manipulation, continuously rotating a cylinder along its Z-axis using the fingers. The reinforcement learning environment is built on **NVIDIA Isaac Lab**.

### 🚀 Quick Start
Use the following commands to train and evaluate the policy in the Isaac Lab environment.

**1. Train**
```bash
# Run Z-rotation training in Isaac Lab
python3 scripts/reinforcement_learning/rsl_rl/train.py --task=Tesollo-Cylinder-Z-Rotate-v0 --num_envs=4096 --headless
```

**2. Play**
```bash
# Load the trained policy checkpoint and evaluate
python3 scripts/reinforcement_learning/rsl_rl/play.py --task=Tesollo-Cylinder-Z-Rotate-v0 --checkpoint=/path/to/model.pth
```