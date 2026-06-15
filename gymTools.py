#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Any


class MarioGridWrapper(gym.ObservationWrapper):
    def __init__(self, env, flatten=True):
        super().__init__(env)
        self.screen_size_x = 16
        self.screen_size_y = 13
        self.flatten = flatten

        # NEAT needs a 1D vector of 208 inputs
        if self.flatten:
            self.observation_space = Box(low=-1, high=2, shape=(208,), dtype=np.int8)
        else:
            self.observation_space = Box(
                low=-1,
                high=2,
                shape=(self.screen_size_y, self.screen_size_x),
                dtype=np.int8,
            )

    # FIX 1: Match the base class signature exactly
    def observation(self, observation: Any) -> np.ndarray:
        # FIX 2: Explicitly type-cast or tell Pyright that we are pulling from the NES RAM array
        # This bypasses the 'Unknown attribute' error safely.
        unwrapped_env: Any = self.env.unwrapped
        ran = unwrapped_env.ram

        # 1. Coordinate tracking
        mario_level_x = int(ran[0x006D]) * 256 + int(ran[0x0086])
        mario_x = int(ran[0x03AD])
        mario_y = int(ran[0x03B8]) - 16

        x_start = mario_level_x - mario_x
        screen_start = int(np.rint(x_start / 16)) % 32

        # 2. Vectorized background matrix parsing
        bg_ram = ran[0x0500:0x06A0]
        bg_grid = bg_ram.reshape(2, 13, 16)
        full_room = np.hstack((bg_grid[0], bg_grid[1]))
        extended_room = np.hstack((full_room, full_room))

        rendered_tiles = extended_room[:, screen_start : screen_start + 16]
        rendered_screen = (rendered_tiles != 0).astype(np.int8)

        # 3. Insert Mario (Value: 2)
        mario_grid_x = (mario_x + 8) // 16
        mario_grid_y = (mario_y - 32) // 16
        if 0 <= mario_grid_x < 16 and 0 <= mario_grid_y < 13:
            rendered_screen[mario_grid_y, mario_grid_x] = 2

        # 4. Insert Enemies (Value: -1)
        for i in range(5):
            if ran[0x000F + i] == 1:
                enemy_x = int(ran[0x006E + i]) * 256 + int(ran[0x0087 + i]) - x_start
                enemy_y = int(ran[0x00CF + i])
                enemy_grid_x = (enemy_x + 8) // 16
                enemy_grid_y = (enemy_y - 32) // 16

                if 0 <= enemy_grid_x < 16 and 0 <= enemy_grid_y < 13:
                    rendered_screen[enemy_grid_y, enemy_grid_x] = -1

        # If flatten is True, returns a 1D array of 208 numbers directly usable by AI
        if self.flatten:
            return rendered_screen.flatten()
        return rendered_screen
