# Mario NES RAM Visualizer

A clean, professional Python environment for exploring the NES Super Mario Bros emulator at the hardware level. This project renders a **dual-viewport display** showing:

- **Left Pane**: The actual NES game screen (pixel-perfect RGB buffer)
- **Right Pane**: An abstracted 16×13 block grid extracted directly from the emulator's RAM

Perfect for AI/ML experimentation, reverse-engineering game physics, or understanding how the NES stores game state in memory.

---

## Features

✨ **Hardware-Level Introspection**
- Extracts game state directly from NES WRAM (Work RAM) at addresses 0x0500–0x06A0
- Tile data parsing for background environment (16×13 visible grid)
- Real-time Mario position and velocity tracking
- Enemy sprite detection and mapping

🎮 **Dual-Mode Control**
- **Manual Mode**: Full keyboard control (arrow keys + spacebar)
- **Autoplay Mode**: Random agent actions for testing/analysis
- Toggle modes on-the-fly with [A] key

🎨 **Polished UI**
- Dark-mode color scheme (Zinc palette) with high contrast
- Side-by-side viewports for real-time comparison
- HUD dashboard with world/level, coordinates, and enemy count
- 60 FPS rendering to match NES timing

📦 **Modular Architecture**
- Clean separation: environment logic, rendering, and UI
- Easy to extend for reinforcement learning agents
- Compatible with both legacy Gym and modern Gymnasium APIs

---

## Installation

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)

### Setup

Clone or download this repository, then install dependencies:

```bash
cd mario_bros_neat
pip install -r requirements.txt
```

**Dependencies:**
- `gym-super-mario-bros>=7.4.0` — NES emulation and environment
- `nes-py>=8.2.1` — Low-level NES emulator
- `pygame>=2.5.0` — Display and input handling
- `numpy>=1.21.0` — Numerical computations

---

## Usage

### Quick Start

```bash
python main.py
```

The dual-viewport window will open (1100×560 pixels). 

### Controls

| Key(s) | Action |
|--------|--------|
| **Left / Right Arrows** | Move Mario left or right |
| **Spacebar / Up Arrow** | Jump vertically |
| **Left/Right + Space** | Jump while moving (diagonal) |
| **[A]** | Toggle between AUTOPLAY and MANUAL mode |
| **[ESC]** | Quit the application |

### Operating Modes

#### Manual Mode (Default)
Use your keyboard to directly control Mario. Useful for:
- Exploring game mechanics
- Understanding how the environment responds to input
- Verifying the RAM extraction is working correctly

#### Autoplay Mode
Press **[A]** to toggle. The environment will:
- Select random valid actions each frame
- Run indefinitely, resetting when episodes end
- Useful for stress-testing the renderer and environment state extraction

---

## Project Structure

```
mario_bros_neat/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Entry point
├── src/
│   ├── __init__.py              # Package initialization
│   ├── environment.py           # MarioRAMGridWrapper (RAM extraction)
│   └── visualizer.py            # Pygame rendering and keyboard handling
└── legacy/
    └── step2_processed_mario.py # Original tutorial script (archived)
```

---

## Understanding the Block Grid

The right viewport displays a **13 rows × 16 columns** grid representing the NES level data:

### Block Values & Colors

| Value | Color | Meaning |
|-------|-------|---------|
| **0** | Sky Blue | Empty space (passable) |
| **1** | Clay Brown | Solid block or obstacle |
| **2** | Vibrant Red | Mario (player character) |
| **-1** | Green | Active enemies (Goomba, Koopa, etc.) |

### How It Works

The NES stores background tile IDs in RAM pages:
- **Address 0x0500–0x052F** (Page 0): Left half of the level (13×16 tiles)
- **Address 0x0530–0x055F** (Page 1): Right half of the level (13×16 tiles)

Each tile occupies 16 pixels in-game. The wrapper:

1. **Reads all 416 bytes** from WRAM tile memory
2. **Parses Mario's position** from CPU registers (level X, screen X, Y)
3. **Computes the visible 16-col window** based on Mario's viewport
4. **Extracts & marks enemies** from RAM sprite tables (up to 5 active)
5. **Renders a clean, abstracted grid** for AI/analysis

This bypasses pixel-based screen analysis—pure hardware-level introspection.

---

## Architecture

### `environment.py` – `MarioRAMGridWrapper`

A wrapper around the gym environment that:
- Intercepts observations and converts RGB frames to block grids
- Parses NES RAM bytes into structured tile/sprite data
- Maintains backward compatibility with Gym and Gymnasium APIs
- Exposes `.last_raw_frame` for dual rendering

### `visualizer.py` – Pygame UI & Controls

Handles:
- Window creation and layout (1100×560 dual viewports)
- Keyboard input mapping to gym action space
- Real-time frame rendering with dual comparison
- HUD with metrics (world, coordinates, enemy count)
- Mode toggling between manual and autoplay

### `main.py` – Entry Point

Minimal launcher that imports and calls the visualizer's main loop.

---

## Example: Training an RL Agent

The abstracted block grid is ideal for training neural networks or Q-learning agents:

```python
from src.environment import MarioRAMGridWrapper
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Setup environment
raw_env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="rgb_array")
raw_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
env = MarioRAMGridWrapper(raw_env)

obs = env.reset()
for step in range(1000):
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, done, info = env.step(action)
    
    # obs is now a 13×16 integer array (clean, compact representation)
    # Perfect for feeding to a CNN or attention model
    
    if done:
        obs = env.reset()
```

---

## Troubleshooting

### Import Errors

```
ImportError: No module named 'gym_super_mario_bros'
```

**Solution**: Reinstall dependencies:
```bash
pip install --upgrade gym-super-mario-bros nes-py pygame numpy
```

### Pygame Window Won't Open

- Ensure you have a display (Wayland/X11 on Linux, native on macOS/Windows)
- Try running with: `SDL_VIDEODRIVER=dummy python main.py` (headless, no window)

### Slow Performance

- Verify your CPU supports hardware-accelerated NumPy operations
- Reduce rendering updates if needed (the main loop defaults to 60 FPS)

### RAM Data Looks Wrong

- The wrapper assumes **SuperMarioBros-1-1** level layout
- Advanced levels may require calibration of RAM address offsets
- This is a known limitation of direct RAM introspection

---

## Performance

- **Rendering**: 60 FPS (locked to NES speed)
- **Memory**: ~50 MB (environment + Pygame)
- **CPU**: Minimal; primarily emulation + rendering overhead
- **Suitable for**: Real-time RL training, live analysis, streaming

---

## References

- [NES Dev Wiki – Mario RAM Map](https://www.smwiki.net/wiki/RAM_map)
- [nes-py Documentation](https://github.com/jjroquo/nes-py)
- [gym-super-mario-bros GitHub](https://github.com/Kautenja/gym-super-mario-bros)
- [Gymnasium API](https://gymnasium.farama.org/)

---

## License

This project is provided as-is for educational and research purposes.

---

## Contributing

Improvements welcome! Some ideas:
- Additional levels beyond 1-1
- PowerUp detection in RAM
- Improved enemy sprite classification
- Integration with popular RL frameworks (Stable-Baselines3, Ray RLlib)
- Async multi-environment support

---

## Author

Developed as a minor project for understanding NES hardware-level game state introspection and real-time visualization.

---

**Made with ❤️ for retro computing and AI/ML enthusiasts.**
