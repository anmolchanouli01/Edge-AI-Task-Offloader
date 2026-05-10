# 🚀 Dynamic AI Task Offloading System for Edge Computing

An intelligent, hardware-aware task routing controller powered by Deep Reinforcement Learning (DQN). 

This project solves the edge-cloud latency dilemma by deploying an AI agent that actively monitors a host machine's physical hardware constraints (CPU, RAM, Battery) in real-time. Instead of relying on rigid, hardcoded routing rules, the AI learns the exact limits of the device and dynamically routes heavy computational tasks to the cloud while keeping lightweight tasks on the local edge processor to minimize total system latency.

## ✨ Key Features

* **Live Hardware Telemetry:** Uses `psutil` to read native OS-level metrics (macOS/Windows/Linux) rather than relying on pure software simulations.
* **Deep Q-Network (DQN):** Built from scratch using PyTorch, the agent uses the Bellman Equation and experience replay to map hardware strain to optimal routing decisions.
* **Continuous Hardware Adaptation:** Features an `adapt_mode` deployment script. When the pre-trained model is moved to a fundamentally different machine (e.g., from a MacBook to a Raspberry Pi), it maintains a 15% exploration rate to seamlessly rewire its neural weights to the new hardware limits.
* **Resource Preservation:** The custom mathematical reward function heavily penalizes actions that spike local RAM above 85% or drain a dying battery, forcing the AI to prioritize system survival over sheer speed.

## 🗂️ Project Architecture

The system is modularized into the classic Reinforcement Learning framework:

1. `universal_env.py` **(The Environment):** Acts as the physical sensory system. It monitors live hardware states and calculates the latency/hardware penalties (Rewards) after a routing decision is made.
2. `universal_agent.py` **(The Brain):** Contains the PyTorch neural network. It predicts the optimal Q-values for incoming tasks based on current system strain and handles the continuous learning/backpropagation.
3. `deploy_anywhere.py` **(The Router):** The main execution script. It loads the pre-trained model, simulates incoming tasks of varying complexities, and coordinates the interaction between the Environment and the Agent.
4. `universal_model.pth` **(Pre-trained Weights):** A fully trained PyTorch model initialized on macOS hardware, ready for instant inference or continuous adaptation fine-tuning on new machines.

## ⚙️ Installation and Setup

To test this offloading agent on your own machine, clone the repository and set up a local virtual environment.

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/Edge-AI-Task-Offloader.git](https://github.com/YOUR_USERNAME/Edge-AI-Task-Offloader.git)
cd Edge-AI-Task-Offloader

# 2. Create and activate a virtual environment
python -m venv venv
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install required dependencies
pip install torch numpy psutil
