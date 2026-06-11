#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pygame
import sys

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Import your custom wrapper here 
# (Assuming it's in the same file or imported from gymTools)
from gymTools import MarioGridWrapper 

# --- Pygame Visual Settings ---
CELL_SIZE = 40  # Pixel size of each matrix tile on your monitor
COLS = 16
ROWS = 13
SCREEN_WIDTH = COLS * CELL_SIZE
SCREEN_HEIGHT = ROWS * CELL_SIZE

# Color Palette for the Matrix Values
COLORS = {
     0: (100, 149, 237),  # Empty Space -> Cornflower Blue (Sky)
     1: (139, 69, 19),    # Tiles -> Saddle Brown (Solid blocks/bricks)
     2: (255, 0, 0),      # Mario -> Red
    -1: (0, 255, 0)       # Enemies -> Bright Green
}

def get_action_from_keyboard():
    """
    Maps keyboard state to gym-super-mario-bros SIMPLE_MOVEMENT indices:
    0: NOOP, 1: Right, 2: Right+Jump, 3: Right+Run, 4: Right+Run+Jump, 5: Jump, 6: Left
    """
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return 2  # Right + Jump
    elif keys[pygame.K_RIGHT]:
        return 1  # Move Right
    elif keys[pygame.K_LEFT]:
        return 6  # Move Left
    elif keys[pygame.K_SPACE]:
        return 5  # Jump straight up
    return 0      # Standing still (NOOP)

def draw_matrix(surface, matrix):
    """Renders the 13x16 numerical matrix as colored grid squares"""
    for r in range(ROWS):
        for c in range(COLS):
            val = matrix[r, c]
            color = COLORS.get(val, (255, 255, 255)) # Fallback to white if unknown value
            
            # Draw the block
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
            
            # Draw a subtle grid line around the block
            pygame.draw.rect(surface, (50, 50, 50), rect, 1)

def main():
    # 1. Initialize Gymnasium Environment with your Wrapper
    base_env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='rgb_array')
    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = MarioGridWrapper(env)
    
    # 2. Initialize Pygame Window
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Mario Matrix Viewport (Human Playable)")
    clock = pygame.time.Clock()
    
    # Reset Environment
    obs, info = env.reset()
    
    running = True
    while running:
        # Limit frame rate to 30 FPS so it's playable by humans
        clock.tick(30) 
        
        # Handle Window Close Exits
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # 3. Get Human Input and Step the Environment
        action = get_action_from_keyboard()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()

        # 4. Render the 13x16 wrapped matrix onto the Pygame window
        draw_matrix(screen, obs)
        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()