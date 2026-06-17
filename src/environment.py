"""
NES RAM Grid Wrapper Module

Extracts a 16x13 block representation directly from NES WRAM memory,
processing tile data, Mario position, and active enemies.
Handles compatibility between Gym and Gymnasium API versions.
"""

import numpy as np

try:
    import nes_py
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
except ImportError as e:
    raise ImportError(
        f"Missing required libraries: {e}\n"
        "Please install with: pip install gym-super-mario-bros nes-py pygame numpy"
    )


class MarioRAMGridWrapper:
    """
    Wraps the Mario environment to extract a 16-column x 13-row block grid
    representation directly from the NES emulator's RAM (WRAM).

    This wrapper:
    - Extracts tile data from WRAM addresses 0x0500-0x06A0
    - Computes Mario's grid position from RAM registers
    - Identifies and marks active enemies on the grid
    - Maintains the original RGB frame for dual-viewport rendering
    """

    def __init__(self, env):
        """
        Initialize the wrapper.

        Args:
            env: A gym_super_mario_bros environment wrapped with JoypadSpace
        """
        self.env = env
        self.screen_size_x = 16
        self.screen_size_y = 13
        self.last_raw_frame = None

        # Forward Gym attributes
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation."""
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            obs, info = reset_result
            self.last_raw_frame = obs
            return self.process_observation(obs), info
        else:
            self.last_raw_frame = reset_result
            return self.process_observation(reset_result)

    def step(self, action):
        """Execute one step of the environment."""
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            self.last_raw_frame = obs
            return self.process_observation(obs), reward, done, info
        else:
            obs, reward, terminated, truncated, info = step_result
            self.last_raw_frame = obs
            return self.process_observation(obs), reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        """Render the environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Close the environment."""
        return self.env.close()

    def process_observation(self, obs):
        """
        Extract block grid from NES RAM.

        The grid represents:
        - 0: Empty space (sky)
        - 1: Solid blocks
        - 2: Mario (player)
        - -1: Enemies (Goomba, Koopa, etc.)

        Returns:
            np.ndarray: 13x16 integer array representing the block grid
        """
        ram = self.env.unwrapped.ram

        # =====================================================================
        # 1. COMPUTE MARIO'S POSITION IN THE LEVEL
        # =====================================================================
        # Mario X coordinate in the entire level (across multiple screens)
        mario_level_x = int(ram[0x6d]) * 256 + int(ram[0x86])
        mario_x = int(ram[0x3ad])  # Pixel position within current page
        mario_y = int(ram[0x3b8]) + 16  # Vertical position (adjusted for sprite height)

        # Calculate where the visible screen window starts in the level
        x_start = mario_level_x - mario_x
        screen_start = int(np.rint(x_start / 16)) % 32

        # =====================================================================
        # 2. EXTRACT TILE DATA FROM WRAM
        # =====================================================================
        # Grab all 416 bytes of tile data (two 208-byte pages of background)
        # WRAM Layout: 2 pages × 13 rows × 16 columns per page
        bg_mem = ram[0x0500:0x06A0]

        # Reshape into (2 pages, 13 rows, 16 columns)
        bg_grid = bg_mem.reshape(2, 13, 16)

        # Merge pages horizontally: (13 rows × 32 columns total)
        full_room = np.hstack((bg_grid[0], bg_grid[1]))

        # Extend horizontally for edge wrapping: (13 rows × 64 columns)
        extended_room = np.hstack((full_room, full_room))

        # Slice the 16-column visible window
        rendered_tiles = extended_room[:, screen_start : screen_start + 16]

        # Convert all non-zero tile IDs to 1 (block), keep 0 (empty)
        rendered_screen = (rendered_tiles != 0).astype(np.int8)

        # =====================================================================
        # 3. PLACE MARIO ON THE GRID (Value: 2)
        # =====================================================================
        mario_grid_x = (mario_x + 8) // 16
        mario_grid_y = (mario_y - 32) // 16
        if 0 <= mario_grid_x < 16 and 0 <= mario_grid_y < 13:
            rendered_screen[mario_grid_y, mario_grid_x] = 2

        # =====================================================================
        # 4. PLACE ENEMIES ON THE GRID (Value: -1)
        # =====================================================================
        # The NES stores up to 5 active enemies in RAM
        for i in range(5):
            if ram[0x0F + i] == 1:  # Enemy active flag
                enemy_x = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - x_start
                enemy_y = int(ram[0xCF + i])
                enemy_grid_x = (enemy_x + 8) // 16
                enemy_grid_y = (enemy_y + 8 - 32) // 16

                if 0 <= enemy_grid_x < 16 and 0 <= enemy_grid_y < 13:
                    rendered_screen[enemy_grid_y, enemy_grid_x] = -1

        return rendered_screen
