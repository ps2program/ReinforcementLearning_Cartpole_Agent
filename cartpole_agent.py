import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from IPython import display
import time
import pickle
import os

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.lr = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        
        # Initialize Q-table with zeros
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # For discretizing continuous state space
        self.buckets = (1, 1, 6, 3)  # Number of buckets for each state dimension
        self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        self.state_bounds[1] = [-0.5, 0.5]  # Cart velocity bounds
        self.state_bounds[3] = [-np.radians(50), np.radians(50)]  # Pole angular velocity bounds
        
        # For visualization
        self.episode_rewards = []
        self.epsilon_history = []
        self.exploration_ratio = []

    def save_model(self, filename='cartpole_model.pkl'):
        """Save the trained Q-table and parameters"""
        model_data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'buckets': self.buckets,
            'state_bounds': self.state_bounds,
            'epsilon': self.epsilon,
            'learning_rate': self.lr,
            'discount_factor': self.gamma
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename='cartpole_model.pkl'):
        """Load a trained Q-table and parameters"""
        if not os.path.exists(filename):
            print(f"No saved model found at {filename}")
            return False
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.q_table.update(model_data['q_table'])
        self.buckets = model_data['buckets']
        self.state_bounds = model_data['state_bounds']
        self.epsilon = model_data['epsilon']
        self.lr = model_data['learning_rate']
        self.gamma = model_data['discount_factor']
        print(f"Model loaded from {filename}")
        return True

    def discretize_state(self, state):
        """Convert continuous state to discrete state"""
        discretized = []
        for i, (s, (lower, upper)) in enumerate(zip(state, self.state_bounds)):
            scale = self.buckets[i] - 1
            bound_width = upper - lower
            offset = (scale * lower) / bound_width
            discretized.append(int(round((scale * s) / bound_width - offset)))
        return tuple(discretized)

    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def train(self, episodes=1000, render_interval=50):
        """Train the agent with visualization and challenging initial states"""
        scores = []
        exploration_count = 0
        total_actions = 0
        
        # Create a figure for real-time plotting
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Define challenging initial states for training
        training_states = [
            # Extreme positions with high velocity
            {'x': -2.4, 'x_dot': 2.0, 'theta': 0.0, 'theta_dot': 0.0},  # Far left, moving right fast
            {'x': 2.4, 'x_dot': -2.0, 'theta': 0.0, 'theta_dot': 0.0},  # Far right, moving left fast
            
            # Extreme angles with high angular velocity
            {'x': 0.0, 'x_dot': 0.0, 'theta': 0.4, 'theta_dot': 1.0},   # Tilted right, spinning right
            {'x': 0.0, 'x_dot': 0.0, 'theta': -0.4, 'theta_dot': -1.0}, # Tilted left, spinning left
            
            # Combined extreme conditions
            {'x': -2.0, 'x_dot': 1.5, 'theta': 0.3, 'theta_dot': 0.5},  # Far left, moving right, tilted right, spinning right
            {'x': 2.0, 'x_dot': -1.5, 'theta': -0.3, 'theta_dot': -0.5}, # Far right, moving left, tilted left, spinning left
            
            # Random extreme states
            {'x': np.random.uniform(-2.4, 2.4), 
             'x_dot': np.random.uniform(-2.0, 2.0),
             'theta': np.random.uniform(-0.4, 0.4),
             'theta_dot': np.random.uniform(-1.0, 1.0)},
            
            {'x': np.random.uniform(-2.4, 2.4), 
             'x_dot': np.random.uniform(-2.0, 2.0),
             'theta': np.random.uniform(-0.4, 0.4),
             'theta_dot': np.random.uniform(-1.0, 1.0)},
        ]
        
        for episode in range(episodes):
            # Reset the training environment
            state = self.env.reset()[0]
            
            # Create a new environment for rendering if needed
            if episode % render_interval == 0:
                render_env = gym.make('CartPole-v1', render_mode='human')
                render_state = render_env.reset()[0]
                
                # Use challenging initial states during training
                if episode < len(training_states) * render_interval:
                    state_idx = (episode // render_interval) % len(training_states)
                    # Set both training and render states
                    state[0] = render_state[0] = training_states[state_idx]['x']
                    state[1] = render_state[1] = training_states[state_idx]['x_dot']
                    state[2] = render_state[2] = training_states[state_idx]['theta']
                    state[3] = render_state[3] = training_states[state_idx]['theta_dot']
                    print(f"\nTraining Episode {episode} - Using challenging initial state:")
                    print(f"x={state[0]:.2f}, x_dot={state[1]:.2f}, theta={state[2]:.2f}, theta_dot={state[3]:.2f}")
            
            state = self.discretize_state(state)
            score = 0
            done = False
            render_done = False
            steps = 0
            falling = False
            fall_steps = 0
            
            while not done:
                action = self.get_action(state)
                total_actions += 1
                if np.random.random() < self.epsilon:
                    exploration_count += 1
                
                # Step the training environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)
                
                # Check if pole is starting to fall
                if not falling and abs(next_state[2]) > 0.2:  # If pole angle is significant
                    falling = True
                
                # If pole is falling, continue for a few more steps
                if falling:
                    fall_steps += 1
                    if fall_steps >= 10:  # Continue for 10 steps after falling starts
                        done = True
                
                done = done or terminated or truncated
                
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                score += reward
                steps += 1
                
                # Render every render_interval episodes
                if episode % render_interval == 0 and not render_done:
                    _, _, render_terminated, render_truncated, _ = render_env.step(action)
                    render_done = render_terminated or render_truncated
                    
                    # Adjust visualization timing
                    if falling:
                        time.sleep(0.05)  # Slower when falling to see the motion
                    elif render_done:
                        time.sleep(0.2)  # Pause at the end to see final state
                    else:
                        time.sleep(0.02)  # Normal speed during operation
            
            if episode % render_interval == 0:
                print(f"Episode {episode} completed in {steps} steps with score {score}")
                time.sleep(0.3)  # Longer pause before closing to see final state
                render_env.close()
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            scores.append(score)
            self.episode_rewards.append(score)
            self.epsilon_history.append(self.epsilon)
            self.exploration_ratio.append(exploration_count / total_actions)
            
            # Update plots every 10 episodes
            if episode % 10 == 0:
                self._update_plots(ax1, ax2, episode)
                plt.pause(0.01)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Average Score: {np.mean(scores[-100:]):.2f}")
        
        plt.ioff()  # Turn off interactive mode
        return scores

    def _update_plots(self, ax1, ax2, episode):
        """Update the real-time plots"""
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Rewards and Epsilon
        ax1.plot(self.episode_rewards, label='Reward', alpha=0.6)
        ax1.plot(self.epsilon_history, label='Epsilon', color='red', alpha=0.6)
        ax1.set_title(f'Training Progress (Episode {episode})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Value')
        ax1.legend()
        
        # Plot 2: Exploration Ratio
        ax2.plot(self.exploration_ratio, label='Exploration Ratio', color='green')
        ax2.set_title('Exploration vs Exploitation')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        
        plt.tight_layout()

    def visualize_q_values(self):
        """Create a heatmap of Q-values for a specific state dimension"""
        # Convert Q-table to numpy array for visualization
        q_values = np.zeros((self.buckets[2], self.buckets[3]))  # Using pole angle and angular velocity
        
        for state, actions in self.q_table.items():
            if len(state) == 4:  # Ensure state has all dimensions
                pole_angle_idx = min(state[2], self.buckets[2] - 1)  # Ensure index is within bounds
                pole_velocity_idx = min(state[3], self.buckets[3] - 1)  # Ensure index is within bounds
                q_values[pole_angle_idx, pole_velocity_idx] = np.max(actions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(q_values, cmap='viridis', annot=True, fmt='.2f')
        plt.title('Q-Value Heatmap (Pole Angle vs Angular Velocity)')
        plt.xlabel('Pole Angular Velocity Bucket')
        plt.ylabel('Pole Angle Bucket')
        plt.savefig('q_value_heatmap.png')
        plt.close()

    def test(self, episodes=10, render=True):
        """Test the trained agent with visualization"""
        print("\nTesting the trained agent:")
        for episode in range(episodes):
            # Create a new environment for rendering
            if render:
                test_env = gym.make('CartPole-v1', render_mode='human')
            else:
                test_env = self.env
            
            state = self.discretize_state(test_env.reset()[0])
            score = 0
            done = False
            
            while not done:
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                next_state = self.discretize_state(next_state)
                done = terminated or truncated
                state = next_state
                score += reward
                
                if render:
                    if done:
                        time.sleep(0.1)  # Brief pause when pole falls to see the final state
                    else:
                        time.sleep(0.02)  # Slightly slower during normal operation
            
            if render:
                time.sleep(0.2)  # Brief pause before closing the render window
                test_env.close()
            
            print(f"Test Episode {episode + 1}, Score: {score}")

def plot_scores(scores):
    """Plot training scores with enhanced visualization"""
    plt.figure(figsize=(12, 6))
    
    # Plot raw scores
    plt.plot(scores, alpha=0.3, label='Raw Scores')
    
    # Plot moving average
    window_size = 100
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(scores)), moving_avg, 
             label=f'{window_size}-Episode Moving Average', linewidth=2)
    
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_scores.png')
    plt.close()

def main():
    # Create environment
    env = gym.make('CartPole-v1', render_mode=None)
    
    # Create agent
    agent = QLearningAgent(env)
    
    # Check if there's a saved model
    if not agent.load_model():
        print("No saved model found. Starting training...")
        # Train the agent with more frequent visualization
        scores = agent.train(episodes=1000, render_interval=50)  # Show visualization every 50 episodes
        
        # Plot training scores
        plot_scores(scores)
        
        # Save the trained model
        agent.save_model()
    else:
        print("Loaded existing model. Skipping training.")
    
    # Visualize Q-values
    agent.visualize_q_values()
    
    # Test the trained agent with visualization and challenging initial states
    agent.test(episodes=10, render=True)
    
    env.close()

if __name__ == "__main__":
    main() 