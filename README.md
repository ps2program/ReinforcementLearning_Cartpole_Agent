# 🎯 CartPole Balancing with Q-Learning

<div align="center">
  <img src="https://gymnasium.farama.org/_images/cart_pole.gif" alt="CartPole Demo" width="400"/>
  <br>
  <em>CartPole Environment Demo</em>
</div>

## 👨‍💻 Developer
**Prahlad** - AI/ML Learning Engineer

## 📝 Overview

This project implements a Q-learning agent to solve the classic CartPole balancing problem from OpenAI Gym. The agent learns to balance a pole on a moving cart by applying forces to move the cart left or right.

## 🎮 Problem Description

### Environment Components
- 🛒 A cart that can move left or right
- 📏 A pole attached to the cart that can swing freely
- 🎯 Goal: Keep the pole balanced upright by moving the cart

### State Space (4 continuous variables)
| Variable | Description | Range |
|----------|-------------|--------|
| x | Cart Position | [-4.8, 4.8] |
| x_dot | Cart Velocity | [-∞, ∞] |
| θ | Pole Angle | [-0.418, 0.418] rad |
| θ_dot | Pole Angular Velocity | [-∞, ∞] |

### Action Space (2 discrete actions)
| Action | Description |
|--------|-------------|
| 0 | Push cart to the left |
| 1 | Push cart to the right |

## 🧠 Q-Learning Theory

Q-learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state. The algorithm uses a Q-table to store the expected utility of taking a given action in a given state.

### Key Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| Q-Table | Q(s,a) | Stores state-action values |
| Learning Rate | α | Controls new information weight |
| Discount Factor | γ | Future reward importance |
| Epsilon | ε | Exploration rate |

### Q-Learning Update Rule

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?Q(s,a)%20=%20Q(s,a)%20+%20\alpha[r%20+%20\gamma%20\max(Q(s',a'))%20-%20Q(s,a)]" alt="Q-Learning Update Rule"/>
</div>

where:
- s: current state
- a: current action
- r: reward
- s': next state
- a': next action
- α: learning rate
- γ: discount factor

## 💻 Implementation Details

### State Discretization
```python
buckets = (1, 1, 6, 3)  # For each state dimension
state_bounds = [
    [-4.8, 4.8],    # Cart Position
    [-0.5, 0.5],    # Cart Velocity
    [-0.418, 0.418], # Pole Angle
    [-0.5, 0.5]     # Pole Angular Velocity
]
```

### Training Process

#### 1. Initialization
- 🎲 Q-table initialized with zeros
- 🔄 Epsilon starts at 1.0 (100% exploration)
- 📊 Learning rate = 0.1
- ⚖️ Discount factor = 0.95

#### 2. Episode Structure
- 🎯 Environment reset with challenging initial states
- 🎲 Actions selected using ε-greedy policy
- 📈 Q-values updated after each step
- 📉 Epsilon decayed over time

#### 3. Visualization
- 📊 Real-time training progress plots
- 🎮 Pole movement visualization
- 🔥 Q-value heatmap generation

### Key Features

#### 1. Challenging Initial States
- 🏃 Extreme positions
- 💨 High velocities
- 📐 Large angles
- 🎯 Combined challenging conditions

#### 2. Visualization Enhancements
- 📈 Real-time training progress
- 🔄 Exploration vs exploitation ratio
- 🔥 Q-value heatmaps
- 🎮 Smooth pole movement visualization

#### 3. Model Persistence
- 💾 Save/load trained Q-tables
- 🔄 Resume training from saved models

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cartpole-qlearning.git
cd cartpole-qlearning

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent
```bash
python cartpole_agent.py
```

The script will:
- 🎓 Train a new agent if no saved model exists
- 📂 Load and test an existing model if available
- 📊 Generate visualization plots
- 💾 Save the trained model

## 📁 Project Structure

```
cartpole-qlearning/
├── cartpole_agent.py  # Main implementation
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## 📦 Dependencies

- 🎮 gymnasium
- 🔢 numpy
- 📊 matplotlib
- 🎨 seaborn
- 💻 IPython

## 🔮 Future Improvements

### 1. Algorithm Enhancements
- [ ] Implement Double Q-learning
- [ ] Add experience replay
- [ ] Try Deep Q-learning (DQN)

### 2. Visualization
- [ ] Add 3D visualization
- [ ] Implement real-time Q-value updates
- [ ] Add more detailed performance metrics

### 3. Training
- [ ] Add parallel training
- [ ] Implement curriculum learning
- [ ] Add more challenging scenarios

## 📚 References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
2. [OpenAI Gym Documentation](https://gymnasium.farama.org/)
3. Watkins, C. J. C. H. (1989). Learning from delayed rewards

---

<div align="center">
  <sub>Built with ❤️ by Prahlad using Python and OpenAI Gym</sub>
  <br>
  <sub>© 2024 All Rights Reserved</sub>
</div> 