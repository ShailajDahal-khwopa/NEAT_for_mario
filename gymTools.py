#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pygame
import sys

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# --- Optimized Gymnasium Wrapper ---
class MarioGridWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.screen_size_x = 16     
        self.screen_size_y = 13     
        
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=2, 
            shape=(self.screen_size_y, self.screen_size_x), 
            dtype=np.int8
        )

    def observation(self, obs):
        ram = self.env.unwrapped.ram
        
        # 1. Cast pointers to standard ints
        mario_level_x = int(ram[0x6d]) * 256 + int(ram[0x86])
        mario_x = int(ram[0x3ad])  
        mario_y = int(ram[0x3b8]) + 16 
        
        x_start = mario_level_x - mario_x 
        screen_start = int(np.rint(x_start / 16)) % 32
        
        # 2. VECTORIZED BACKGROUND EXTRACTION (Replaces the 208-iteration loop)
        # Grab all 416 bytes of tile RAM memory at once (Page 0 and Page 1)
        bg_mem = ram[0x0500:0x06A0]
        
        # Reshape into layout: (2 pages, 13 rows, 16 columns)
        bg_grid = bg_mem.reshape(2, 13, 16)
        
        # Merge Page 0 (left) and Page 1 (right) horizontally into a single 13x32 room grid
        full_room = np.hstack((bg_grid[0], bg_grid[1]))
        
        # Duplicate horizontally to effortlessly handle screen wrapping edges (13x64)
        extended_room = np.hstack((full_room, full_room))
        
        # Slice out the exact 16-column window visible on screen
        rendered_tiles = extended_room[:, screen_start:screen_start + 16]
        
        # Convert any structural tile ID variations into a uniform '1'
        rendered_screen = (rendered_tiles != 0).astype(np.int8)
                    
        # 3. Add Mario (Value: 2)
        mario_grid_x = (mario_x + 8) // 16
        mario_grid_y = (mario_y - 32) // 16 
        if 0 <= mario_grid_x < 16 and 0 <= mario_grid_y < 13:
            rendered_screen[mario_grid_y, mario_grid_x] = 2
        
        # 4. Add Enemies (Value: -1)
        for i in range(5):
            if ram[0xF + i] == 1:  
                enemy_x = int(ram[0x6e + i]) * 256 + int(ram[0x87 + i]) - x_start
                enemy_y = int(ram[0xcf + i])
                enemy_grid_x = (enemy_x + 8) // 16
                enemy_grid_y = (enemy_y + 8 - 32) // 16

                if 0 <= enemy_grid_x < 16 and 0 <= enemy_grid_y < 13: 
                    rendered_screen[enemy_grid_y, enemy_grid_x] = -1
                    
        return rendered_screen

# --- Pygame Engine Setup ---
CELL_SIZE = 40  
COLS, ROWS = 16, 13
SCREEN_WIDTH, SCREEN_HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE

COLORS = {
     0: (100, 149, 237),  # Sky Blue
     1: (139, 69, 19),    # Blocks (Brown)
     2: (255, 0, 0),      # Mario (Red)
    -1: (0, 255, 0)       # Enemies (Green)
}

def get_action_from_keyboard():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]: return 2  # Right + Jump
    if keys[pygame.K_RIGHT]: return 1                           # Right
    if keys[pygame.K_LEFT]: return 6                            # Left
    if keys[pygame.K_SPACE]: return 5                           # Jump
    return 0                                                    # NOOP

def draw_matrix(surface, matrix):
    for r in range(ROWS):
        for c in range(COLS):
            val = matrix[r, c]
            color = COLORS.get(val, (255, 255, 255))
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (50, 50, 50), rect, 1)

def main():
    base_env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='rgb_array')
    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = MarioGridWrapper(env)
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Optimized Mario Matrix Viewport")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    
    running = True
    while running:
        # FIXED: Increased to 60 FPS to match standard NES execution speeds
        clock.tick(60) 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        action = get_action_from_keyboard()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()

        draw_matrix(screen, obs)
        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()