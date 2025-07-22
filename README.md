# Reinforced Maze Navigation using Deep Q-Networks (DQN)

This project implements **Reinforced Pathfinding in a Maze** using a **Deep Q-Network (DQN)**.  
The agent learns to navigate a maze environment by maximizing rewards, avoiding obstacles, and reaching a target.  
It uses **PyTorch, OpenAI Gym (custom environment), and reinforcement learning principles**.

---

## Features

- **Custom Maze Environment** (`envs/maze_env.py`) built with OpenAI Gym interface.
- **Deep Q-Network Agent** (`agents/dqn_agent.py`) with experience replay and epsilon-greedy exploration.
- **Training & Evaluation** (`connected_maze_training.py`) for continuous learning.
- **Visualization** – Training metrics including rewards, epsilon decay, and temporal difference plots.
- **Pre-trained Model** (`dqn_model.pth`) and dataset (`training_data.xlsx`) included.
- **Configurable Hyperparameters** via `config.json`.

---

## Project Structure

neural networks/
└── mazegame_modified/
├── simulations/
│ ├── maze_simulation.py # Main simulation runner
│ ├── connected_maze_training.py # DQN agent training script
│ ├── config.json # Configurations
│ ├── maze_grid.npy # Maze layout
│ ├── source_destination.npy # Start & goal positions
│ ├── dqn_model.pth # Pre-trained weights
│ ├── training_data.xlsx # Collected episode data
│ ├── agents/
│ │ ├── dqn_agent.py # DQN agent class
│ │ └── models/dqn_model.pth # Model weights
│ ├── envs/
│ │ └── maze_env.py # Custom maze environment
│ ├── Utils/utils.py # Utility functions
│ └── plots/ # Training metrics & plots
└── __MACOSX/ (ignore)


## Installation

### Requirements
Make sure you have Python 3.9+ and install dependencies:

```bash
pip install torch numpy matplotlib gym pandas
Usage
1. Run Simulation with Pre-trained Model
bash
Copy
Edit
python simulations/maze_simulation.py
2. Train the DQN Agent
bash
Copy
Edit
python simulations/connected_maze_training.py
Training results (plots and logs) will be stored in the plots/ folder.

Outputs
The project generates:

Cumulative Reward over episodes.

Epsilon Decay (exploration vs exploitation).

Episode Lengths and Returns.

Final trained model stored in dqn_model.pth.

Example plots:



Configuration
All hyperparameters (learning rate, epsilon decay, replay buffer, etc.) are stored in config.json.
You can edit them to adjust agent performance.

Author
Developed by Varshini Myadam (SRM University).
For queries or collaboration, reach out via GitHub or LinkedIn.

License
This project is licensed under the MIT License – you can freely use and modify it.
