#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class MarioGridWrapper(gym.ObservationWrapper):
    def __init__(self, env, flatten=True):
        super().__init__(env)
        self.screen_size_x = 16     
        self.screen_size_y = 13     
        self.flatten = flatten
        
        # NEAT needs a 1D vector of 208 inputs; standard RL (like CNNs) might prefer 2D (13, 16)
        if self.flatten:
            self.observation_space = Box(
                low=-1, high=2, shape=(208,), dtype=np.int8
            )
        else:
            self.observation_space = Box(
                low=-1, high=2, shape=(self.screen_size_y, self.screen_size_x), dtype=np.int8
            )

    def observation(self, obs):
        ram = self.env.unwrapped.ram
        
        # 1. Coordinate tracking
        mario_level_x = int(ram[0x6d]) * 256 + int(ram[0x86])
        mario_x = int(ram[0x3ad])  
        mario_y = int(ram[0x3b8]) + 16 
        
        x_start = mario_level_x - mario_x 
        screen_start = int(np.rint(x_start / 16)) % 32
        
        # 2. Vectorized background matrix parsing
        bg_mem = ram[0x0500:0x06A0]
        bg_grid = bg_mem.reshape(2, 13, 16)
        full_room = np.hstack((bg_grid[0], bg_grid[1]))
        extended_room = np.hstack((full_room, full_room))
        
        rendered_tiles = extended_room[:, screen_start:screen_start + 16]
        rendered_screen = (rendered_tiles != 0).astype(np.int8)
                    
        # 3. Insert Mario (Value: 2)
        mario_grid_x = (mario_x + 8) // 16
        mario_grid_y = (mario_y - 32) // 16 
        if 0 <= mario_grid_x < 16 and 0 <= mario_grid_y < 13:
            rendered_screen[mario_grid_y, mario_grid_x] = 2
        
        # 4. Insert Enemies (Value: -1)
        for i in range(5):
            if ram[0xF + i] == 1:  
                enemy_x = int(ram[0x6e + i]) * 256 + int(ram[0x87 + i]) - x_start
                enemy_y = int(ram[0xcf + i])
                enemy_grid_x = (enemy_x + 8) // 16
                enemy_grid_y = (enemy_y + 8 - 32) // 16

                if 0 <= enemy_grid_x < 16 and 0 <= enemy_grid_y < 13: 
                    rendered_screen[enemy_grid_y, enemy_grid_x] = -1
                    
        # If flatten is True, returns a 1D array of 208 numbers directly usable by AI
        return rendered_screen.flatten() if self.flatten else rendered_screen