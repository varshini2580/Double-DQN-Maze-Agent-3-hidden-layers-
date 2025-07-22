import pygame
import numpy as np
from envs.maze_env import MazeEnv
from agents.dqn_agent import DoubleDQNAgent
import time
import sys

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 30
WINDOW_SIZE = 800
CELL_SIZE = WINDOW_SIZE // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Load maze data
maze = np.load('maze_grid.npy')
start_pos = tuple(np.load('source_destination.npy')[0])
goal_pos = tuple(np.load('source_destination.npy')[1])

# Create environment and agent
env = MazeEnv(maze, start_pos, goal_pos)
agent = DoubleDQNAgent(state_size=2, action_size=4)

# Initialize screen
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Maze Simulation")

def draw_maze():
    screen.fill(WHITE)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[y, x] == 1:  # Wall
                pygame.draw.rect(screen, BLACK, rect)
            elif (x, y) == start_pos:  # Start
                pygame.draw.rect(screen, GREEN, rect)
            elif (x, y) == goal_pos:  # Goal
                pygame.draw.rect(screen, RED, rect)
    pygame.display.flip()

def main():
    clock = pygame.time.Clock()
    running = True
    
    # Reset environment
    state = env.reset()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset environment
                    state = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Draw maze
        draw_maze()
        
        # Get action from agent
        action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, done = env.step(action)
        
        # Update agent
        agent.update(state, action, reward, next_state, done)
        
        # Update state
        state = next_state
        
        if done:
            state = env.reset()
        
        clock.tick(10)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
