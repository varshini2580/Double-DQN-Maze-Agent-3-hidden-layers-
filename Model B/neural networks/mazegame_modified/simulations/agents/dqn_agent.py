# double_dqn_maze_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DoubleDQNAgent:
    def __init__(self, maze_size, goal, action_dim, gamma=0.99, lr=5e-4, batch_size=128, buffer_size=100000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999):
        self.maze_size = maze_size
        self.goal = goal
        self.state_size = maze_size * maze_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = 100  
        self.target_update_counter = 0 
        self.steps_done = 0  # Track total steps
        self.episode_steps = 0  # Track steps in current episode
        self.episode = 0  # Track current episode
        self.max_episodes = 5000  # Maximum number of episodes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.state_size, action_dim).to(self.device)
        self.target_net = DQNNetwork(self.state_size, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Epsilon-greedy parameters
        self.epsilon_start = float(epsilon_start)  # Ensure float type
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(self.epsilon_start)  # Initialize epsilon
        
        # Verify epsilon starts at 1.0
        print(f"Initialized epsilon: {self.epsilon}")
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        import os
        # Model saving parameters
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_save_path = os.path.join(current_dir, "models/")
        self.model_filename = "dqn_model.pth"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)

    def encode_state(self, flat_index):
        one_hot_tensor = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
        one_hot_tensor[flat_index] = 1
        target_index = self.goal[0] * self.maze_size + self.goal[1]
        one_hot_tensor[target_index] = 2
        return one_hot_tensor

    def select_action(self, state):
        # Update episode steps
        self.episode_steps += 1
        self.steps_done += 1
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            return random.randrange(self.action_dim)
        else:
            # Greedy action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).argmax().item()
    
    def update_epsilon(self, episode):
        """Update epsilon using exponential decay"""
        # Only update epsilon if this is not the first call
        if episode > 0:
            old_epsilon = self.epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if episode % 10 == 0:  # Print every 10 episodes to avoid too much output
                print(f"Episode {episode}: Epsilon updated from {old_epsilon:.4f} to {self.epsilon:.4f} (min: {self.epsilon_end:.4f})")
        return self.epsilon




    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # Return 0 loss if not enough samples

        # Sample batch from replay memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q values for chosen actions
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute next Q values using target network
        with torch.no_grad():
            # Double DQN: select best actions using policy net but evaluate using target net
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            
            # Compute the target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_frequency:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_update_counter = 0
            
        return loss.item()  # Return the loss value

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """
        Save the policy network weights to a file.
        """
        torch.save(self.policy_net.state_dict(), self.model_save_path + self.model_filename)
        print(f'Info: The model has been saved...')
