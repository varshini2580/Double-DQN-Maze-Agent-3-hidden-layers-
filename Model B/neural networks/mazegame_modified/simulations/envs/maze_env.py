import numpy as np

class MazeEnv:
    def __init__(self, maze, start, goal, max_episode_steps=500):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.agent_pos = start
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

    def __init__(self, maze, start, goal, max_episode_steps=500):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.agent_pos = start
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.visited_states = set([start])  # Track visited states for exploration bonus
        self.last_distance_to_goal = self._distance_to_goal(start)  # Track progress to goal

    def _distance_to_goal(self, pos):
        """Calculate Manhattan distance to goal"""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        info = {'Success': False}
        
        # Calculate new position
        x, y = self.agent_pos
        if action == 0:  # up
            new_pos = (x - 1, y)
        elif action == 1:  # right
            new_pos = (x, y + 1)
        elif action == 2:  # down
            new_pos = (x + 1, y)
        elif action == 3:  # left
            new_pos = (x, y - 1)
        else:
            new_pos = (x, y)  # Invalid action, stay in place

        # Check if the new position is valid
        if (0 <= new_pos[0] < self.maze.shape[0] and 
            0 <= new_pos[1] < self.maze.shape[1] and 
            self.maze[new_pos[0]][new_pos[1]] == 0):
            
            # Update position
            prev_pos = self.agent_pos
            self.agent_pos = new_pos
            
            # Check if goal reached
            if new_pos == self.goal:
                reward = 5.0
                terminated = True
                info['Success'] = True
                self.current_step = 0
                return self._get_state(), reward, terminated, truncated, info
                
            # Small positive reward for valid move
            reward = 0.05
            
            # Check if max steps reached
            if self.current_step >= self.max_episode_steps:
                truncated = True
                self.current_step = 0
                return self._get_state(), reward, terminated, truncated, info
                
            # Exploration bonus for new states
            if new_pos not in self.visited_states:
                reward += 0.05
                self.visited_states.add(new_pos)
            
            # Progress-based rewards
            current_dist = self._distance_to_goal(new_pos)
            if current_dist < self.last_distance_to_goal:
                reward += 0.07  # Reward for getting closer to goal
            else:
                reward -= 0.10  # Small penalty for moving away
            self.last_distance_to_goal = current_dist
            
        else:  # Invalid move (hit wall or out of bounds)
            reward = -0.75
            terminated = True  # Terminate episode on invalid move
            self.current_step = 0
        
        return self._get_state(), reward, terminated, truncated, info

    def reset(self):
        self.agent_pos = self.start
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        x, y = self.agent_pos
        width = self.maze.shape[1]
        return x * width + y
