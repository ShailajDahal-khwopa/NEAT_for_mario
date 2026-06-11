import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Import your optimized wrapper
from gymTools import MarioGridWrapper

def eval_genome(genome, config):
    """
    Example NEAT evaluation function running at uncapped CPU speed
    """
    # 1. Initialize environment headlessly
    base_env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode=None)
    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = MarioGridWrapper(env, flatten=True) # Flattened for neural network input layer
    
    # Create your network from genome structure
    # net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    obs, info = env.reset()
    fitness = 0
    done = False
    
    while not done:
        # obs is a flat NumPy array of length 208 containing only -1, 0, 1, 2
        # outputs = net.activate(obs)
        # action = np.argmax(outputs)
        
        action = env.action_space.sample() # Placeholder execution step
        obs, reward, terminated, truncated, info = env.step(action)
        
        fitness += reward
        done = terminated or truncated
        
        # Performance/Stagnation guardrails
        if fitness < -50: # Example breaking condition if Mario gets stuck
            break
            
    env.close()
    return fitness