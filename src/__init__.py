"""
Mario NES RAM Visualizer Package

A clean, modular interface to the NES Super Mario Bros environment,
extracting game state directly from emulator RAM and rendering it
alongside the actual game screen.
"""

from src.environment import MarioRAMGridWrapper

__all__ = ["MarioRAMGridWrapper"]
__version__ = "1.0.0"
