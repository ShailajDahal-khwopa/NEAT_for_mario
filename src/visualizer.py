"""
Pygame Dual-Viewport Visualizer Module

Renders a side-by-side comparison of:
- Left: The actual NES game screen (RGB buffer)
- Right: The abstracted 16x13 block grid extracted from RAM

Provides keyboard controls, HUD metrics, and a polished dark-mode UI.
"""

import sys
import numpy as np
import pygame

from src.environment import MarioRAMGridWrapper

try:
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
except ImportError as e:
    raise ImportError(
        f"Missing required libraries: {e}\n"
        "Please install with: pip install gym-super-mario-bros nes-py pygame numpy"
    )

# ==============================================================================
# DISPLAY & LAYOUT CONSTANTS
# ==============================================================================

# Grid rendering
CELL_SIZE = 32
GRID_COLS, GRID_ROWS = 16, 13
GRID_WIDTH = GRID_COLS * CELL_SIZE  # 512
GRID_HEIGHT = GRID_ROWS * CELL_SIZE  # 416

# NES emulator output
GAME_WIDTH, GAME_HEIGHT = 512, 480  # NES scaled 2x (256x240 -> 512x480)

# Layout spacing
MARGIN_X = 24
MARGIN_Y = 40
GAP_X = 28

# Viewport positions
GAME_X = MARGIN_X
GAME_Y = MARGIN_Y
GRID_X = MARGIN_X + GAME_WIDTH + GAP_X
GRID_Y = MARGIN_Y

# Total screen dimensions
SCREEN_WIDTH = MARGIN_X + GAME_WIDTH + GAP_X + GRID_WIDTH + MARGIN_X  # 1100
SCREEN_HEIGHT = MARGIN_Y + GAME_HEIGHT + MARGIN_Y  # 560

# ==============================================================================
# COLOR PALETTE (Dark Mode Design)
# ==============================================================================

BACKGROUND_COLOR = (24, 24, 27)  # Zinc-900 (sleek dark)
CARD_BG_COLOR = (39, 39, 42)  # Zinc-800
BORDER_COLOR = (63, 63, 70)  # Zinc-700
TEXT_COLOR = (244, 244, 245)  # Zinc-100
ACCENT_COLOR = (241, 196, 15)  # Gold (labels)

GRID_COLORS = {
    0: (52, 152, 219),  # Sky Blue (empty)
    1: (186, 139, 87),  # Clay Brown (blocks)
    2: (231, 76, 60),  # Vibrant Red (Mario)
    -1: (46, 204, 113),  # Goomba Green (enemies)
}


# ==============================================================================
# KEYBOARD & INPUT HANDLING
# ==============================================================================


def get_action_from_keyboard():
    """
    Maps Pygame keyboard state to gym-super-mario-bros action indices.

    Action Mapping:
    - 0: NOOP
    - 1: Right
    - 2: Right + Jump
    - 5: Jump/A (vertical)
    - 6: Left

    Returns:
        int: Action index for gym environment
    """
    keys = pygame.key.get_pressed()

    if keys[pygame.K_RIGHT] and (keys[pygame.K_SPACE] or keys[pygame.K_UP]):
        return 2  # Right + A (Jump Right)
    if keys[pygame.K_RIGHT]:
        return 1  # Right
    if keys[pygame.K_LEFT]:
        return 6  # Left
    if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
        return 5  # A (Jump Vertical)
    return 0  # NOOP


# ==============================================================================
# RENDERING FUNCTIONS
# ==============================================================================


def draw_game_screen(surface, frame):
    """
    Render the raw NES frame buffer to the left viewport.

    Args:
        surface: Pygame display surface
        frame: NumPy array (H×W×C) of NES game pixels
    """
    if frame is not None:
        # Convert from NumPy (Height, Width, Channel) to Pygame surface layout
        transposed = np.swapaxes(frame, 0, 1)
        frame_surface = pygame.surfarray.make_surface(transposed)

        # Scale to target display size
        scaled_surf = pygame.transform.scale(frame_surface, (GAME_WIDTH, GAME_HEIGHT))
        surface.blit(scaled_surf, (GAME_X, GAME_Y))

        # Draw border
        pygame.draw.rect(
            surface,
            BORDER_COLOR,
            (GAME_X - 2, GAME_Y - 2, GAME_WIDTH + 4, GAME_HEIGHT + 4),
            2,
        )
    else:
        # Placeholder for missing frame
        pygame.draw.rect(surface, CARD_BG_COLOR, (GAME_X, GAME_Y, GAME_WIDTH, GAME_HEIGHT))
        pygame.draw.rect(
            surface,
            BORDER_COLOR,
            (GAME_X - 2, GAME_Y - 2, GAME_WIDTH + 4, GAME_HEIGHT + 4),
            2,
        )


def draw_matrix(surface, matrix):
    """
    Render the 16x13 block grid on the right viewport.

    Args:
        surface: Pygame display surface
        matrix: NumPy array (13×16) with block values
    """
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = matrix[r, c]
            color = GRID_COLORS.get(val, (255, 255, 255))
            rect = pygame.Rect(
                GRID_X + c * CELL_SIZE, GRID_Y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(surface, color, rect)

            # Subtle gridlines
            pygame.draw.rect(surface, (40, 40, 45), rect, 1)

    # Draw border
    pygame.draw.rect(
        surface,
        BORDER_COLOR,
        (GRID_X - 2, GRID_Y - 2, GRID_WIDTH + 4, GRID_HEIGHT + 4),
        2,
    )


def draw_hud(surface, ram, info, autoplay, font, bold_font):
    """
    Render the HUD panel with metrics and status indicators.

    Args:
        surface: Pygame display surface
        ram: NES RAM buffer
        info: Step info dict from environment
        autoplay: Boolean indicating autoplay mode
        font: Regular font
        bold_font: Bold font for emphasis
    """
    hud_y = GRID_Y + GRID_HEIGHT + 14
    card_rect = pygame.Rect(GRID_X, hud_y, GRID_WIDTH, 50)

    # Dark backing card
    pygame.draw.rect(surface, CARD_BG_COLOR, card_rect, 0, 8)
    pygame.draw.rect(surface, BORDER_COLOR, card_rect, 1, 8)

    # 1. Mode status
    mode_lbl = font.render("MODE: ", True, (160, 160, 165))
    mode_text = "AUTOPLAY [A]" if autoplay else "MANUAL [A]"
    mode_color = (46, 204, 113) if autoplay else (231, 76, 60)  # Green vs Red
    mode_val = bold_font.render(mode_text, True, mode_color)

    surface.blit(mode_lbl, (GRID_X + 16, hud_y + 16))
    surface.blit(mode_val, (GRID_X + 16 + mode_lbl.get_width(), hud_y + 16))

    # 2. World, coordinates, and enemy info
    if ram is not None:
        mario_level_x = int(ram[0x6D]) * 256 + int(ram[0x86])
        mario_y = int(ram[0x3B8])
        enemy_count = sum(1 for i in range(5) if ram[0x0F + i] == 1)
        world_num = int(ram[0x75F]) + 1
        level_num = int(ram[0x75C]) + 1
        world_text = f"W{world_num}-{level_num}"

        info_str = (
            f"{world_text}  |  X: {mario_level_x:<4} Y: {mario_y:<3}  |  Enemies: {enemy_count}"
        )
    else:
        info_str = "World: 1-1  |  X: - Y: -  |  Enemies: -"

    info_val = font.render(info_str, True, TEXT_COLOR)
    surface.blit(info_val, (GRID_X + GRID_WIDTH - info_val.get_width() - 16, hud_y + 16))

    # 3. Viewport labels
    label_left = bold_font.render("ACTUAL NES EMULATOR SCREEN", True, ACCENT_COLOR)
    label_right = bold_font.render("16x13 RAM BLOCKS VIEWPORT", True, ACCENT_COLOR)

    surface.blit(label_left, (GAME_X, GAME_Y - 25))
    surface.blit(label_right, (GRID_X, GRID_Y - 25))


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================


def main():
    """
    Main entry point for the Mario RAM visualizer.

    Initializes the environment, Pygame display, and runs the main loop
    with support for both manual keyboard control and autoplay mode.
    """
    print("Starting Mario NES RAM Visualizer...")

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    try:
        # Create base environment with RGB rendering
        try:
            raw_env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="rgb_array")
        except TypeError:
            # Fallback for older Gym versions
            raw_env = gym_super_mario_bros.make("SuperMarioBros-v0")

        # Wrap with JoypadSpace (provides simple discrete action space)
        raw_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)

        # Wrap with our RAM grid processor
        env = MarioRAMGridWrapper(raw_env)

    except Exception as e:
        print(f"\nError initializing environment: {e}")
        sys.exit(1)

    # =========================================================================
    # PYGAME INITIALIZATION
    # =========================================================================
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Mario NES RAM Visualizer - Dual Block Viewport")
    clock = pygame.time.Clock()

    # Fonts
    font = pygame.font.SysFont("Courier New", 14, bold=True)
    bold_font = pygame.font.SysFont("Courier New", 15, bold=True)

    # =========================================================================
    # STATE INITIALIZATION
    # =========================================================================
    obs = env.reset()
    info = None
    done = False

    autoplay = False  # Start in keyboard manual play mode
    step_count = 0

    print("\n" + "=" * 70)
    print("  MARIO NES RAM VISUALIZER ACTIVE")
    print("=" * 70)
    print("  CONTROLS:")
    print("    [A]           Toggle AUTOPLAY (random) vs MANUAL (keyboard) mode")
    print("    [LEFT/RIGHT]  Move Mario left or right")
    print("    [SPACE/UP]    Jump (vertical or combined with direction)")
    print("    [ESC]         Quit the application")
    print("=" * 70 + "\n")

    try:
        # =====================================================================
        # MAIN LOOP
        # =====================================================================
        while not done:
            clock.tick(60)  # NES runs at 60 FPS

            # Handle input events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key == pygame.K_a:
                        autoplay = not autoplay
                        mode_str = "AUTOPLAY (random actions)" if autoplay else "MANUAL (keyboard control)"
                        print(f"[Step {step_count}] Switched to: {mode_str}")

            # Select action based on mode
            if autoplay:
                action = env.action_space.sample()
            else:
                action = get_action_from_keyboard()

            # Execute environment step
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            # Reset on episode end
            if done:
                print(f"[Step {step_count}] Episode ended. Resetting environment...")
                obs = env.reset()
                done = False

            # ================================================================
            # RENDER FRAME
            # ================================================================
            screen.fill(BACKGROUND_COLOR)

            # Left panel: Real NES game frame
            draw_game_screen(screen, env.last_raw_frame)

            # Right panel: RAM block grid
            draw_matrix(screen, obs)

            # Bottom panel: HUD & metrics
            draw_hud(screen, env.env.unwrapped.ram, info, autoplay, font, bold_font)

            pygame.display.flip()
            step_count += 1

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    finally:
        env.close()
        pygame.quit()
        print(f"\nVisualizer closed after {step_count} steps. Goodbye!")


if __name__ == "__main__":
    main()
