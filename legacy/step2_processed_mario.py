import sys
import time
import numpy as np
import pygame

# Try to import Gym/Gymnasium and nes-py wrappers
try:
    import nes_py
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
except ImportError as e:
    print(f"\nImport Error: {e}")
    print("="*80)
    print("Missing required libraries! Please install them using pip:")
    print("  pip install gym-super-mario-bros nes-py pygame numpy")
    print("="*80 + "\n")
    sys.exit(1)

# ==============================================================================
# STEP 2: The RAM-based Observation Wrapper
# Extracts a 16 columns by 13 rows block representation from the NES WRAM.
# Also stores the original game frame for dual-window rendering.
# ==============================================================================
class MarioRAMGridWrapper:
    def __init__(self, env):
        self.env = env
        self.screen_size_x = 16     
        self.screen_size_y = 13     
        self.last_raw_frame = None
        
        # Forward Gym attributes so the environment behaves normally
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            obs, info = reset_result
            self.last_raw_frame = obs
            return self.process_observation(obs), info
        else:
            self.last_raw_frame = reset_result
            return self.process_observation(reset_result)

    def step(self, action):
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
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def process_observation(self, obs):
        ram = self.env.unwrapped.ram
        
        # 1. Cast pointers to standard ints
        # Mario X coordinates (screen page index * 256 + in-page offset)
        mario_level_x = int(ram[0x6d]) * 256 + int(ram[0x86])
        mario_x = int(ram[0x3ad])  
        mario_y = int(ram[0x3b8]) + 16 
        
        x_start = mario_level_x - mario_x 
        screen_start = int(np.rint(x_start / 16)) % 32
        
        # 2. Vectorized Background Extraction
        # Grab all 416 bytes of tile RAM memory at once (Page 0 and Page 1)
        bg_mem = ram[0x0500:0x06A0]
        
        # Reshape into layout: (2 pages, 13 rows, 16 columns)
        bg_grid = bg_mem.reshape(2, 13, 16)
        
        # Merge Page 0 (left) and Page 1 (right) horizontally into a single 13x32 room grid
        full_room = np.hstack((bg_grid[0], bg_grid[1]))
        
        # Duplicate horizontally to handle screen wrapping edges (13x64)
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
        # Check active enemy slots (up to 5 slots)
        for i in range(5):
            if ram[0xF + i] == 1:  
                enemy_x = int(ram[0x6e + i]) * 256 + int(ram[0x87 + i]) - x_start
                enemy_y = int(ram[0xcf + i])
                enemy_grid_x = (enemy_x + 8) // 16
                enemy_grid_y = (enemy_y + 8 - 32) // 16

                if 0 <= enemy_grid_x < 16 and 0 <= enemy_grid_y < 13: 
                    rendered_screen[enemy_grid_y, enemy_grid_x] = -1
                    
        return rendered_screen

# ==============================================================================
# PYGAME DUAL VIEWPORT SETUP & RENDERING
# ==============================================================================
CELL_SIZE = 32
GRID_COLS, GRID_ROWS = 16, 13
GRID_WIDTH, GRID_HEIGHT = GRID_COLS * CELL_SIZE, GRID_ROWS * CELL_SIZE # 512 x 416

GAME_WIDTH, GAME_HEIGHT = 512, 480 # NES scaled 2x (256x240 -> 512x480)

# Screen and margins layout
MARGIN_X = 24
MARGIN_Y = 40
GAP_X = 28

GAME_X = MARGIN_X
GAME_Y = MARGIN_Y

GRID_X = MARGIN_X + GAME_WIDTH + GAP_X
GRID_Y = MARGIN_Y

SCREEN_WIDTH = MARGIN_X + GAME_WIDTH + GAP_X + GRID_WIDTH + MARGIN_X # 24 + 512 + 28 + 512 + 24 = 1100
SCREEN_HEIGHT = MARGIN_Y + GAME_HEIGHT + MARGIN_Y # 40 + 480 + 40 = 560

# Harmonious visual design colors
BACKGROUND_COLOR = (24, 24, 27) # Sleek Dark Mode (Zinc-900)
CARD_BG_COLOR = (39, 39, 42)    # Zinc-800
BORDER_COLOR = (63, 63, 70)     # Zinc-700
TEXT_COLOR = (244, 244, 245)    # Zinc-100

GRID_COLORS = {
     0: (52, 152, 219),  # Sky Blue (Smooth flat blue)
     1: (186, 139, 87),  # Blocks (Warm clay brown)
     2: (231, 76, 60),   # Mario (Vibrant Red)
    -1: (46, 204, 113)   # Enemies (Goomba/Koopa Green)
}

def get_action_from_keyboard():
    """Maps Pygame keyboard state to gym-super-mario-bros Actions."""
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

def draw_game_screen(surface, frame):
    """Blits and scales the raw NES frame buffer to the left panel."""
    if frame is not None:
        # Swap axes to convert from NumPy (Height, Width, Channel) to Pygame surface layout (Width, Height, Channel)
        transposed = np.swapaxes(frame, 0, 1)
        frame_surface = pygame.surfarray.make_surface(transposed)
        
        # Scale to 512x480
        scaled_surf = pygame.transform.scale(frame_surface, (GAME_WIDTH, GAME_HEIGHT))
        surface.blit(scaled_surf, (GAME_X, GAME_Y))
        
        # Border
        pygame.draw.rect(surface, BORDER_COLOR, (GAME_X - 2, GAME_Y - 2, GAME_WIDTH + 4, GAME_HEIGHT + 4), 2)
    else:
        # Placeholder
        pygame.draw.rect(surface, CARD_BG_COLOR, (GAME_X, GAME_Y, GAME_WIDTH, GAME_HEIGHT))
        pygame.draw.rect(surface, BORDER_COLOR, (GAME_X - 2, GAME_Y - 2, GAME_WIDTH + 4, GAME_HEIGHT + 4), 2)

def draw_matrix(surface, matrix):
    """Draws the 16x13 block viewport based on RAM data on the right panel."""
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = matrix[r, c]
            color = GRID_COLORS.get(val, (255, 255, 255))
            rect = pygame.Rect(GRID_X + c * CELL_SIZE, GRID_Y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
            # Subtle gridlines
            pygame.draw.rect(surface, (40, 40, 45), rect, 1)
            
    # Border
    pygame.draw.rect(surface, BORDER_COLOR, (GRID_X - 2, GRID_Y - 2, GRID_WIDTH + 4, GRID_HEIGHT + 4), 2)

def draw_hud(surface, ram, info, autoplay, font, bold_font):
    """Draws details, controls, and active states at the bottom of the right panel."""
    hud_y = GRID_Y + GRID_HEIGHT + 14
    card_rect = pygame.Rect(GRID_X, hud_y, GRID_WIDTH, 50)
    
    # Draw dark backing container card
    pygame.draw.rect(surface, CARD_BG_COLOR, card_rect, 0, 8)
    pygame.draw.rect(surface, BORDER_COLOR, card_rect, 1, 8)
    
    # 1. Mode status
    mode_lbl = font.render("MODE: ", True, (160, 160, 165))
    mode_text = "AUTOPLAY [A]" if autoplay else "MANUAL [A]"
    mode_color = (46, 204, 113) if autoplay else (231, 76, 60) # Green vs Red
    mode_val = bold_font.render(mode_text, True, mode_color)
    
    surface.blit(mode_lbl, (GRID_X + 16, hud_y + 16))
    surface.blit(mode_val, (GRID_X + 16 + mode_lbl.get_width(), hud_y + 16))
    
    # 2. Coordinates and World Info
    if ram is not None:
        mario_level_x = int(ram[0x6d]) * 256 + int(ram[0x86])
        mario_y = int(ram[0x3b8])
        enemy_count = sum(1 for i in range(5) if ram[0xF + i] == 1)
        world_text = f"W{int(ram[0x75f])+1}-{int(ram[0x75c])+1}"
        
        info_str = f"{world_text}  |  X: {mario_level_x:<4} Y: {mario_y:<3}  |  Enemies: {enemy_count}"
    else:
        info_str = "World: 1-1  |  X: - Y: -  |  Enemies: -"
        
    info_val = font.render(info_str, True, TEXT_COLOR)
    surface.blit(info_val, (GRID_X + GRID_WIDTH - info_val.get_width() - 16, hud_y + 16))
    
    # Header Labels for viewports
    label_left = bold_font.render("ACTUAL NES EMULATOR SCREEN", True, (241, 196, 15))
    label_right = bold_font.render("16x13 RAM BLOCKS VIEWPORT", True, (241, 196, 15))
    
    surface.blit(label_left, (GAME_X, GAME_Y - 25))
    surface.blit(label_right, (GRID_X, GRID_Y - 25))

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("Starting Step 2: Observations & Wrappers with RAM Grid...")
    
    # Setup environments
    try:
        # Make the NES env headless-capable by rendering in rgb_array mode
        try:
            raw_env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='rgb_array')
        except TypeError:
            raw_env = gym_super_mario_bros.make('SuperMarioBros-v0')
            
        raw_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        env = MarioRAMGridWrapper(raw_env)
        
    except Exception as e:
        print(f"\nError initializing environment: {e}")
        sys.exit(1)

    # Init pygame for rendering
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Mario RAM Block Viewport vs Real Game Sandbox")
    clock = pygame.time.Clock()
    
    # Fonts
    font = pygame.font.SysFont("Courier New", 14, bold=True)
    bold_font = pygame.font.SysFont("Courier New", 15, bold=True)

    obs = env.reset()
    info = None
    done = False
    
    autoplay = False # Start in keyboard manual play mode
    step_count = 0
    
    print("\n" + "="*60)
    print("  MARIO DUAL VIEWPORT SIMULATOR RUNNING!")
    print("="*60)
    print("  Controls:")
    print("    - Press [A] to toggle Autoplay (Random moves) vs Manual Keyboard Play")
    print("    - In Manual Play: Use [LEFT] and [RIGHT] arrows to run")
    print("                      Use [SPACE] or [UP] arrow to jump")
    print("    - Press [ESC] or close the window to quit")
    print("="*60 + "\n")

    try:
        while not done:
            clock.tick(60) # NES matches 60 FPS
            
            # Event processing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key == pygame.K_a:
                        autoplay = not autoplay
                        print(f"Switched mode to: {'AUTOPLAY' if autoplay else 'MANUAL PLAY'}")
            
            # Action selection
            if autoplay:
                action = env.action_space.sample()
            else:
                action = get_action_from_keyboard()

            # Execute step
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                
            if done:
                print("Episode ended. Resetting environment...")
                obs = env.reset()
                done = False
            
            # Draw everything
            screen.fill(BACKGROUND_COLOR)
            
            # Left pane: Real Game frame
            draw_game_screen(screen, env.last_raw_frame)
            
            # Right pane: RAM block grid
            draw_matrix(screen, obs)
            
            # Bottom pane: HUD & Metrics
            draw_hud(screen, env.env.unwrapped.ram, info, autoplay, font, bold_font)
            
            pygame.display.flip()
            step_count += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()
        print("\nSimulator Stopped. Step 2 Complete!")

if __name__ == "__main__":
    main()
