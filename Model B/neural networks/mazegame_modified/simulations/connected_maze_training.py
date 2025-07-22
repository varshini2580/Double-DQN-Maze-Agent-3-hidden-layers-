import pygame
import numpy as np
import threading
import time
import queue
import seaborn as sns
import torch  
from envs.maze_env import MazeEnv
from agents.dqn_agent import DoubleDQNAgent
from Utils.utils import load_maze_data, astar
import pandas as pd
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

def save_training_data_to_excel(data_dict, episode, goal_reached=False, steps_to_goal=np.nan, saved_model=False):
    """
    Save training metrics to Excel file with the following columns:
    - Run_ID: Unique identifier for this training run
    - Episode: Episode number within this run
    - Episode_Reward: Total reward for the episode
    - Episode_Length: Number of steps in the episode
    - Epsilon: Exploration rate used
    - TD_Error: Temporal Difference error
    - Goal_Reached: Whether the goal was reached
    - Steps_to_Goal: Steps taken to reach goal (NaN if not reached)
    - Saved_Model: Whether model was saved this episode
    - Timestamp: When the episode was completed
    - Cumulative_Reward: Running total of rewards for this run
    """
    excel_path = "training_data.xlsx"
    
    try:
        # Get or create run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create new row with all metrics including Run_ID
        new_row = {
            'Run_ID': run_id,
            'Episode': episode,
            'Episode_Reward': data_dict['returns'],
            'Episode_Length': data_dict['steps'],
            'Epsilon': data_dict['epsilon'],
            'TD_Error': data_dict['training_error'],
            'Goal_Reached': goal_reached,
            'Steps_to_Goal': steps_to_goal,
            'Saved_Model': saved_model,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Cumulative_Reward': data_dict['returns']
        }
        
        # Create DataFrame with new row
        df = pd.DataFrame([new_row])
        
        # If file exists, read existing data and append new row
        if os.path.exists(excel_path):
            try:
                # Read existing data with error handling for empty files
                try:
                    existing_df = pd.read_excel(excel_path)
                    if existing_df.empty:
                        existing_df = pd.DataFrame()
                except:
                    existing_df = pd.DataFrame()
                
                # If existing data is not empty, append new data
                if not existing_df.empty:
                    # Add Run_ID column to existing data if it doesn't exist
                    if 'Run_ID' not in existing_df.columns:
                        existing_df['Run_ID'] = 'OLD_RUN'
                    
                    # Combine with new data
                    df = pd.concat([existing_df, df], ignore_index=True)
                    
                    # Sort by Run_ID and then Episode
                    df = df.sort_values(['Run_ID', 'Episode']).reset_index(drop=True)
                    
                    # Calculate cumulative reward for this run
                    run_mask = df['Run_ID'] == run_id
                    if run_mask.any():
                        # Calculate running sum for the current run
                        df.loc[run_mask, 'Cumulative_Reward'] = df[run_mask]['Episode_Reward'].cumsum()
            except Exception as e:
                print(f"Warning: Could not read existing Excel file, creating new: {e}")
        
        # Save the data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use mode='a' to append if file exists, but we're handling appending manually
                # to have better control over the data structure
                df.to_excel(excel_path, index=False)
                if attempt > 0:
                    print(f"Successfully saved after {attempt + 1} attempts")
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    print("Error: Could not save training data - file may be open in another program")
                    # Try to save to a backup file
                    backup_path = f"training_data_backup_{int(time.time())}.xlsx"
                    df.to_excel(backup_path, index=False)
                    print(f"Saved backup to {backup_path}")
                    return
                time.sleep(0.5)  # Wait a bit before retrying
        
        # Print progress every 10 episodes or on the first episode
        if episode % 10 == 0 or episode == 1:
            print(f"Saved metrics for episode {episode} of run {run_id}")
            
    except Exception as e:
        print(f"Error in save_training_data_to_excel: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        traceback.print_exc()

def check_saved_files():
    """Check if all files are properly saved"""
    print("\n=== Checking saved files ===")
    # Check Excel file
    excel_path = "training_data.xlsx"
    if os.path.exists(excel_path):
        print(f"Excel file found: {excel_path}")
        try:
            df = pd.read_excel(excel_path)
            print(f"Excel file contains {len(df)} rows of data")
            print(f"Last episode: {df['Episode'].iloc[-1]}")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
    else:
        print(f"Excel file not found: {excel_path}")
    
    # Check model file
    model_path = "dqn_model.pth"
    if os.path.exists(model_path):
        print(f"Model file found: {model_path}")
        try:
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            print(f"Model file size: {model_size:.2f} MB")
        except Exception as e:
            print(f"Error checking model file: {e}")
    else:
        print(f"Model file not found: {model_path}")
    
    # Check plots directory
    plots_dir = "custom_plots"
    if os.path.exists(plots_dir):
        print(f"Plots directory found: {plots_dir}")
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        print(f"Found {len(plot_files)} plot files:")
        for plot in plot_files:
            print(f" - {plot}")
    else:
        print(f"Plots directory not found: {plots_dir}")
    print("-" * 50)

def generate_plots():
    """Generate all training plots with consistent formatting using ALL historical data"""
    try:
        # Create custom_plots directory if it doesn't exist
        custom_plots_dir = "custom_plots"
        if not os.path.exists(custom_plots_dir):
            os.makedirs(custom_plots_dir)
        
        # Load data from Excel - this will contain ALL historical data
        excel_path = "training_data.xlsx"
        if not os.path.exists(excel_path):
            print("No training data available to plot")
            return
            
        # Read all data from Excel
        df = pd.read_excel(excel_path)
        if len(df) == 0:
            print("No data available in the Excel file")
            return
            
        # Ensure we have the latest data by forcing a reload
        df = pd.read_excel(excel_path)
        
        # Standardize column names if needed
        if 'Episode_Reward' in df.columns and 'Returns_Per_Episode' not in df.columns:
            df['Returns_Per_Episode'] = df['Episode_Reward']
        
        # Set common plot parameters
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.alpha'] = 0.5
        
        # Get episode range
        episodes = range(len(df))
        
        # Update plot parameters based on actual data ranges
        max_episodes = len(df)
        episode_ticks = range(0, max_episodes + 1, max(1, max_episodes // 10))
        
        # 1. Temporal Difference per Episode - Show ALL episodes
        plt.figure(figsize=(14, 6))
        plt.plot(df['Episode'], df['Training_Error'], color='green', linewidth=1, alpha=0.7)
        plt.title('Temporal Difference per Episode (All Data)', pad=15)
        plt.xlabel('Episode Number', labelpad=10)
        plt.ylabel('Temporal Difference', labelpad=10)
        
        # Auto-adjust x-axis based on data range
        max_episode = df['Episode'].max()
        x_ticks = list(range(0, max_episode + 1, max(1, max_episode // 10)))
        if len(x_ticks) > 10:  # Limit number of x-ticks for readability
            x_ticks = list(range(0, max_episode + 1, max(1, max_episode // 5)))
        plt.xticks(x_ticks)
        
        # Auto-adjust y-axis while keeping the scale you wanted
        y_max = max(3000, df['Training_Error'].max() * 1.1)  # Add 10% padding
        plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        plt.ylim(0, y_max)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(custom_plots_dir, 'temporal_difference.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Episode Length per Episode
        plt.figure()
        plt.plot(episodes, df['Steps_Per_Episode'], color='red', linewidth=0.5, alpha=0.7)
        plt.title('Episode Length per Episode', pad=15)
        plt.xlabel('Episodes', labelpad=10)
        plt.ylabel('Episode Length', labelpad=10)
        plt.xticks(range(0, 10001, 1000))
        plt.yticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
        plt.ylim(0, 200)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(custom_plots_dir, 'episode_length.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Returns per Episode
        plt.figure()
        plt.plot(episodes, df['Returns_Per_Episode'], color='blue', linewidth=0.5, alpha=0.7)
        plt.title('Returns per Episode', pad=15)
        plt.xlabel('Episodes', labelpad=10)
        plt.ylabel('Returns', labelpad=10)
        plt.xticks(range(0, 10001, 1000))
        plt.yticks([-20, -15, -10, -5, 0, 5])
        plt.ylim(-20, 5)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(custom_plots_dir, 'returns_per_episode.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Epsilon Decay per Episode
        plt.figure()
        plt.plot(episodes, df['Epsilon_History'], color='purple', linewidth=2)
        plt.title('Epsilon Decay per Episode', pad=15)
        plt.xlabel('Episodes', labelpad=10)
        plt.ylabel('Epsilon', labelpad=10)
        plt.xticks(range(0, 10001, 1000))
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(custom_plots_dir, 'epsilon_decay.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Cumulative Reward per Episode
        plt.figure()
        cumulative_rewards = df['Returns_Per_Episode'].cumsum()
        plt.plot(episodes, cumulative_rewards, color='orange', linewidth=1.5)
        plt.title('Cumulative Reward per Episode', pad=15)
        plt.xlabel('Episodes', labelpad=10)
        plt.ylabel('Cumulative Returns', labelpad=10)
        plt.xticks(range(0, 10001, 1000))
        plt.yticks([-10000, -8000, -6000, -4000, -2000, 0])
        plt.ylim(-10000, 0)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(custom_plots_dir, 'cumulative_reward.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("All plots generated successfully in the 'custom_plots' directory")
        
    except KeyError as e:
        print(f"Error: Missing required column in data - {e}")
        print("Available columns:", df.columns.tolist())
    except Exception as e:
        print(f"An error occurred while generating plots: {e}")
        print(f"Error generating plots: {e}")

# Window dimensions
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# Simulation settings
FPS = 50

print("Connected Maze Training System started")

# Shared data structures
movement_queue = queue.Queue()
training_data = {
    'current_state': None,
    'action': None,
    'reward': None,
    'next_state': None,
    'done': False,
    'step_count': 0
}

# Global variables for thread safety
training_active = False
simulation_complete = False

# Load maze data
try:
    maze, start, goal = load_maze_data("maze_grid.npy", "source_destination.npy")
    print(f"Loaded maze: {maze.shape}, Start: {start}, Goal: {goal}")
except Exception as e:
    print(f"Error loading maze data: {e}")
    # Fallback maze
    maze = np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 0, 0]
    ])
    start = (0, 0)
    goal = (3, 3)
    print("Using fallback maze")

# Create environment and path
# Load maze and create environment
maze, start, goal = load_maze_data("maze_grid.npy", "source_destination.npy")
env = MazeEnv(maze, start, goal)

# Get A* path for guidance
path = astar(maze, start, goal)
try:
    path = astar(maze, start, goal)
    print(f"Path found with {len(path)} steps: {path[:5]}..." if len(path) > 5 else f"Path: {path}")
except Exception as e:
    print(f"Error finding path: {e}")
    path = [start, goal]  # Simple fallback path

# Pygame setup
pygame.init()
WIDTH = HEIGHT = 500  # Larger window for better visibility
ROWS, COLS = maze.shape
CELL_SIZE = min(WIDTH // COLS, HEIGHT // ROWS)
ACTUAL_WIDTH = COLS * CELL_SIZE
ACTUAL_HEIGHT = ROWS * CELL_SIZE

win = pygame.display.set_mode((ACTUAL_WIDTH, ACTUAL_HEIGHT + 150))  # Extra space for info
pygame.display.set_caption("Maze Simulation with DQN Training")
font = pygame.font.Font(None, 20)

def calculate_action(current_pos, next_pos):
    """Calculate action based on position change"""
    if current_pos is None or next_pos is None or current_pos == next_pos:
        return -1  # No movement
    
    dr = next_pos[0] - current_pos[0]
    dc = next_pos[1] - current_pos[1]
    
    # Actions: 0=up, 1=down, 2=left, 3=right
    if dr == -1 and dc == 0:
        return 0  # up
    elif dr == 1 and dc == 0:
        return 1  # down
    elif dr == 0 and dc == -1:
        return 2  # left
    elif dr == 0 and dc == 1:
        return 3  # right
    else:
        return -1  # Invalid movement

def calculate_reward(current_pos, next_pos, goal_pos, maze):
    """Calculate reward for the movement"""
    if next_pos is None or current_pos is None:
        return 0
    
    try:
        if next_pos == goal_pos:
            return 100  # Reached goal
        elif next_pos[0] < 0 or next_pos[0] >= maze.shape[0] or next_pos[1] < 0 or next_pos[1] >= maze.shape[1]:
            return -10  # Out of bounds
        elif maze[next_pos[0]][next_pos[1]] == 1:
            return -10  # Hit obstacle
        else:
            # Distance-based reward
            current_dist = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
            next_dist = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])
            return -1 + (current_dist - next_dist) * 0.5
    except Exception as e:
        print(f"Error calculating reward: {e}")
        return -1

def draw_grid(win, maze, path, agent_pos, step_info=None):
    try:
        # Draw maze cells
        for i in range(ROWS):
            for j in range(COLS):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Draw cell
                if maze[i][j] == 1:
                    pygame.draw.rect(win, (0, 0, 0), rect)  # Black for walls
                else:
                    pygame.draw.rect(win, (255, 255, 255), rect)  # White for paths
                # Draw cell border
                #pygame.draw.rect(win, (128, 128, 128), rect, 1)  # Gray grid lines

        # Draw path
        if path and len(path) > 1:
            try:
                path_points = [(p[1] * CELL_SIZE + CELL_SIZE // 2, 
                               p[0] * CELL_SIZE + CELL_SIZE // 2) for p in path]
                pygame.draw.lines(win, (0, 0, 255), False, path_points, 2)
            except Exception as e:
                print(f"Error drawing path: {e}")

        # Draw start point
        if start is not None:
            start_center = (start[1] * CELL_SIZE + CELL_SIZE // 2,
                          start[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(win, (0, 255, 0), start_center, CELL_SIZE // 3)

        # Draw goal point
        if goal is not None:
            goal_center = (goal[1] * CELL_SIZE + CELL_SIZE // 2,
                         goal[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(win, (255, 0, 0), goal_center, CELL_SIZE // 3)

        # Draw agent
        if agent_pos is not None:
            agent_center = (agent_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                          agent_pos[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(win, (255, 165, 0), agent_center, CELL_SIZE // 3)

        # Draw info text
        if step_info:
            info_y = ACTUAL_HEIGHT + 10
            
            # Handle action gracefully
            action = step_info.get('action', 0)
            if isinstance(action, int) and 0 <= action < 4:
                action_text = ['Up', 'Down', 'Left', 'Right'][action]
            else:
                action_text = 'None'
            
            # Handle all values safely
            texts = [
                f"Step: {step_info.get('step_count', 0)}",
                f"Action: {action_text}",
                f"Reward: {step_info.get('reward', 0) if step_info.get('reward') is not None else 0:.2f}",
                f"Done: {str(step_info.get('done', False)).upper()}"
            ]
            
            for i, text in enumerate(texts):
                try:
                    text_surface = font.render(text, True, (0, 0, 0))
                    win.blit(text_surface, (10 + (i * 200), info_y))
                except Exception as e:
                    print(f"Error rendering text: {e}")
                    continue
                
    except Exception as e:
        print(f"Error in draw_grid: {e}")
        import traceback
        traceback.print_exc()  # This will help debug any remaining issues

def dqn_training_thread():
    # Track epsilon values for plotting (using a Python list for appending)
    epsilon_history = []
    """Thread function for Double DQN training that processes movement data"""
    global training_active, simulation_complete
    
    # Load epsilon parameters from config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    epsilon_params = config.get('training', {
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.9995
    })
    
    agent = DoubleDQNAgent(
        env.state_size, 
        env.action_size, 
        epsilon_start=epsilon_params['epsilon_start'],
        epsilon_end=epsilon_params['epsilon_end'],
        epsilon_decay=epsilon_params['epsilon_decay']
    )
    episode = 0
    max_episodes = 5000  # Set to exactly 5000 episodes
    episode_rewards = []
    episode_steps = []
    
    try:
        while episode < max_episodes and not simulation_complete:
            if not training_active:
                time.sleep(0.1)
                continue
            
            episode += 1
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                # Check for new movement data
                if not movement_queue.empty():
                    movement_data = movement_queue.get()
                    state = movement_data['next_state']
                    done = movement_data['done']
                    
                # Generate plots when agent reaches goal
                if done and total_reward > 0:  # Assuming positive reward for reaching goal
                    print(f"Agent reached goal in episode {episode}!")
            
            # Save training data after every episode
            training_data = {
                'returns': total_reward,
                'training_error': agent.get_training_error() if hasattr(agent, 'get_training_error') else 0.0,
                'loss': training_data.get('loss', 0.0),
                'steps': steps,
                'epsilon': agent.epsilon
            }
            
            # Check if goal was reached
            goal_reached = done and total_reward > 0
            steps_to_goal = steps if goal_reached else np.nan
            
            # Save to Excel
            save_training_data_to_excel(
                training_data, 
                episode, 
                goal_reached=goal_reached,
                steps_to_goal=steps_to_goal
            )
            
            # Print episode summary
            print(f"Episode {episode}: "
                  f"Reward={total_reward:.2f}, "
                  f"Steps={steps}, "
                  f"Loss={training_data.get('loss', 0.0):.4f}, "
                  f"Epsilon={agent.epsilon:.4f}, "
                  f"Goal Reached: {goal_reached}")
            
            # Save model periodically
            if episode % 100 == 0:  # Save more frequently
                model_path = f"dqn_model_ep{episode}.pth"
                torch.save(agent.qnetwork_local.state_dict(), model_path)
                print(f"Model saved at episode {episode} to {model_path}")
                
                # Also save a copy with the latest name
                torch.save(agent.qnetwork_local.state_dict(), "dqn_model_latest.pth")
                
            # Store experience in replay buffer and train the agent
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train the agent and get the loss
            loss = agent.optimize()
            
            # Update target network periodically
            if agent.steps_done % 100 == 0:
                agent.update_target()
                
            # Track loss for logging
            if loss is not None:
                training_data['loss'] = loss
                
            # Update previous position
            previous_pos = current_pos
            
            # Update training data for visualization
            training_data.update({
                'current_state': previous_pos,
                'action': action,
                'reward': reward,
                'next_state': current_pos,
                'done': done,
                'step_count': step_count
            })
            
            # Print training information
            if episode % 100 == 0:
                print(f"Episode {episode} completed")
                print(f"Double DQN Step {step_count}:")
                print(f"  State: {previous_pos} -> {current_pos}")
                action_name = ['Up', 'Down', 'Left', 'Right'][action]
                print(f"  Action: {action} ({action_name})")
                print(f"  Reward: {reward:.2f}")
                print(f"  Done: {done}")
                
                # Get Q-values for current state
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    q_values = agent.policy_net(state_tensor).squeeze().cpu().numpy()
                    print(f"  Q-values: {q_values}")
                print("-" * 50)
                
                step_count += 1
                
                if done:
                    episode_count += 1
                    print(f"Episode {episode_count} / {max_episodes} completed")
                    episode_rewards.append(reward)
                    if episode_count < max_episodes:
                        print(f"Episode {episode_count + 1} / {max_episodes} started")
                    if not training_active:  # Check if user wants to stop
                        print(f"Training stopped at episode {episode_count}")
                        break
                    env.reset()
                    
            previous_pos = current_pos
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error in training thread: {e}")
    finally:
        # Final save when training completes
        if episode > 0:
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            training_data = {
                'returns': total_reward,
                'training_error': agent.get_training_error(),
                'steps': steps,
                'epsilon': agent.epsilon
            }
            
            # Save metrics with saved_model status and timestamp
            save_training_data_to_excel(
                training_data,
                episode + 1,  # Changed to display episodes as 1-5000 instead of 0-4999
                goal_reached=goal_reached[episode],
                steps_to_goal=steps_per_episode[episode] if goal_reached[episode] else np.nan,
                saved_model=saved_models[episode]
            )
            generate_plots()
            torch.save(agent.qnetwork_local.state_dict(), "dqn_model_final.pth")
            print("Final model saved")
        
        training_active = False
        print("Training thread completed")
    
    print("Training thread finished")

# Main simulation function
def maze_simulation():
    global training_active, simulation_complete
    
    print("Starting Maze Simulation with Double DQN Agent and A* path visualization...")
    
    # Initialize metrics tracking
    # We need 5001 elements (indices 0-5000) to store 5000 episodes
    epochs = 5001  # This allows storing data for episodes 0-5000 (5001 total)
    returns_per_episode = np.zeros(epochs)
    steps_per_episode = np.zeros(epochs)
    training_error = np.zeros(epochs)
    time_steps = np.zeros(epochs)
    saved_models = np.zeros(epochs, dtype=bool)  # Boolean array for saved models
    goal_reached = np.zeros(epochs, dtype=bool)  # Boolean array for goal reaching
    epsilon_history = []  # Will store epsilon values for each episode
    current_time_step = 0
    
    # Track when the model was last saved
    last_saved_episode = -1
    
    pygame.init()
    pygame.display.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Maze Simulation with Double DQN Training")
    win.fill((255, 255, 255))
    pygame.display.flip()
    clock = pygame.time.Clock()
    run = True

    # Initialize agent and environment variables
    agent = DoubleDQNAgent(
        maze_size=maze.shape[0],
        goal=goal,
        action_dim=4,  # Up, Down, Left, Right
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        epsilon_start=1.0,    # Start with full exploration
        epsilon_end=0.01,     # Minimum exploration rate
        epsilon_decay=0.995   # Decay rate per episode
    )

    episode = 0
    max_episodes = 5000
    max_steps = 500

    # Compute A* path for visualization
    astar_path = astar(maze, tuple(start), tuple(goal))

    training_active = True
    simulation_complete = False

    # Initialize first state
    current_pos = env.agent_pos
    flat_index = current_pos[0] * maze.shape[1] + current_pos[1]
    state_tensor = agent.encode_state(flat_index)

    # Main simulation loop
    while run and episode < max_episodes:  # Add episode check to prevent going over max_episodes
        episode_running = True
        episode_steps = 0
        episode_reward = 0
        episode_loss = 0
        time_steps = np.zeros(epochs, dtype=int)
        saved_model = np.zeros(epochs, dtype=bool)
        
        # Start from current position instead of resetting

        # Episode loop
        while episode_running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    episode_running = False
                    break
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("Simulation ended by user")
                    # Print current episode statistics
                    if episode_steps > 0:
                        print(f"Episode {episode + 1}/{max_episodes} - Steps: {episode_steps}, Return: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Loss: {episode_loss:.3f}, Reward: {episode_reward:.2f}")
                    run = False
                    episode_running = False
                    break
            if not run:
                break

            action = agent.select_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # For backward compatibility
            
            # env.agent_pos is updated by env.step internally

            # Experience replay and optimization
            agent.memory.push(
                state_tensor.cpu().numpy(),
                action,
                reward,
                agent.encode_state(next_state).cpu().numpy(),
                done
            )
            
            loss = agent.optimize()
            if loss is not None:
                episode_loss += loss

            # Update episode metrics
            episode_reward += reward
            episode_steps += 1
            training_data['step_count'] += 1
            
            # Check if we reached the goal
            info = {'Success': False}
            if done and env.agent_pos == goal:
                info['Success'] = True
                agent.save_model()
                saved_model[episode] = True
                print(f"\n=== Goal reached after {episode_steps} steps! ===")
                print(f"Episode {episode + 1}/{max_episodes}")
                
                # Update goal_reached status
                goal_reached[episode] = info['Success']
                print(f"Steps: {episode_steps}")
                print(f"Return: {episode_reward:.2f}")
                print(f"Loss: {episode_loss:.3f}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"-" * 50)

            # Rendering
            win.fill((255, 255, 255))
            draw_grid(win, maze, astar_path, env.agent_pos, training_data)
            
            # Cap the frame rate
            clock.tick(240)
            pygame.display.flip()
            
            # Check termination conditions
            if done or episode_steps >= max_steps:
                if episode_steps >= max_steps:
                    print("Max steps reached!")
                
                # If goal was reached, reset agent position
                if done:
                    env.agent_pos = start
                    print(f"Agent reset to start position: {start}")
                
                # Update metrics
                returns_per_episode[episode] = episode_reward
                steps_per_episode[episode] = episode_steps
                training_error[episode] = episode_loss / episode_steps if episode_steps > 0 else 0
                time_steps[episode] = current_time_step
                saved_model[episode] = True
                
                # Save metrics to Excel
                excel_path = "training_data.xlsx"
                print(f"Saving metrics to Excel at episode {episode}")
                
                # Create data dictionary
                # Update epsilon for next episode and track it
                current_epsilon = agent.update_epsilon(episode)
                if len(epsilon_history) < epochs:  # Prevent index out of bounds
                    epsilon_history.append(float(current_epsilon))  # Track epsilon as float
                
                # Prepare data for saving
                data_dict = {
                    'returns': float(episode_reward),
                    'training_error': float(training_error[episode]) if episode < len(training_error) else 0.0,
                    'steps': int(episode_steps),
                    'epsilon': float(current_epsilon),
                    'Epsilon_History': float(epsilon_history[-1]) if epsilon_history else 0.0  # Use last epsilon value
                }
                
                # Determine if goal was reached and steps to goal
                goal_reached_episode = done
                steps_to_goal = episode_steps if done else np.nan
                
                # Save metrics
                save_training_data_to_excel(data_dict, episode, goal_reached=False, steps_to_goal=steps_to_goal)
                
                # Save model if not saved yet
                if not np.any(saved_model):
                    agent.save_model()
                    time.sleep(2)
                
                # Generate plots only if goal was reached
                if info['Success']:
                    print("\n=== Generating updated plots... ===")
                    try:
                        # Import and call our custom plot function
                        from custom_plots_v2 import generate_custom_plots_v2
                        generate_custom_plots_v2()
                        print("Updated plots generated successfully!")
                    except Exception as e:
                        print(f"Error generating plots: {e}")
                
                # Print episode statistics
                if done and env.agent_pos == env.goal:
                    print(f"\n=== Goal reached at position {env.goal}! ===\n")
                print(f"Episode {episode + 1}/{max_episodes} - Steps: {episode_steps}, Return: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Loss: {episode_loss:.3f}, Reward: {episode_reward:.2f}")
                episode_running = False
                episode += 1
    
    # Save model if not saved yet
    if not np.any(saved_model):
        saved_model[episode] = True
        agent.save_model()
    
    print(f'-' * 147)

if __name__ == "__main__":
    try:
        maze_simulation()
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        
        pygame.quit()