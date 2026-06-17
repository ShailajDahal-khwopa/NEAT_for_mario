#!/usr/bin/env python3
"""
Main entry point for the Mario NES RAM Visualizer.

Launches the dual-viewport simulator showing the actual NES game frame
alongside a 16x13 block grid extracted from emulator RAM.

Run with:
    python main.py
"""

from src.visualizer import main

if __name__ == "__main__":
    main()
