#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
import time
import neat
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Import your custom modules
from gymTools import MarioGridWrapper
from train import Train


def run_training(generations, parallel, level, config_file):
    """Launches the parallelized evolutionary training loop"""
    print("\n==================================================")
    # Corrected timestamp reference logic for standard file creation logs
    print(f"STARTING NEAT TRAINING FOR MARIO LEVEL {level}")
    print(f"Generations: {generations} | Parallel Workers: {parallel}")
    print(f"Configuration File: {config_file}")
    print("==================================================\n")

    # Initialize your training orchestration engine
    trainer = Train(generations=generations, parallel=parallel, level=level)
    trainer.main(config_file=config_file)


def play_winner(pickle_path, config_file, level):
    """Loads a saved genome and renders its gameplay on screen"""
    target_fps = 30
    frame_delay = 1.0 / target_fps

    if not os.path.exists(pickle_path):
        print(f"Error: Saved genome file '{pickle_path}' not found!")
        sys.exit(1)

    print(f"\nLoading genome from: {pickle_path}...")
    with open(pickle_path, "rb") as f:
        genome = pickle.load(f)

    # Reconstruct the NEAT configuration setup
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Build the network brain from the saved structural data
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Initialize environment with human rendering mode enabled
    print(
        f"Launching visual playback for Level {level}. Press CTRL+C inside terminal to exit."
    )
    base_env = gym_super_mario_bros.make(
        f"SuperMarioBros-{level}-v0", render_mode="human"
    )
    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = MarioGridWrapper(env, flatten=True)

    # Define macro actions mapping array matching training logic
    actions_map = [1, 3, 4]

    state, _info = env.reset()  # FIX: Prefixed unused info
    done = False

    try:
        while not done:
            # Process sensory matrix elements through the neural network
            inputs = state.flatten()
            output = net.activate(inputs)

            # Select the macro action index with the highest activation node response
            chosen_action = actions_map[output.index(max(output))]

            state, _reward, terminated, truncated, _info = env.step(
                chosen_action
            )  # FIX: Prefixed unused assets
            done = terminated or truncated

            # Render the frame
            env.unwrapped.render()
            time.sleep(frame_delay)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Playback closed.")


def main():
    parser = argparse.ArgumentParser(
        description="NEAT Super Mario Bros Project Orchestrator"
    )

    # Core operational mode switch
    parser.add_argument(
        "mode",
        choices=["train", "play"],
        help="Choose 'train' to evolve networks, or 'play' to view an evolved network.",
    )

    # Configuration parameter adjustments
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Path to the NEAT configuration file (default: 'config')",
    )
    parser.add_argument(
        "--gen",
        type=int,
        default=1000,
        help="Number of generations to execute during training (default: 1000)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=4,
        help="Number of parallel CPU worker processes for training (default: 4)",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="1-1",
        help="The Mario level to load, e.g., '1-1', '1-2' (default: '1-1')",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="winner.pkl",
        help="The saved pickle filename to evaluate during 'play' mode (default: 'winner.pkl')",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training(
            generations=args.gen,
            parallel=args.cores,
            level=args.level,
            config_file=args.config,
        )
    elif args.mode == "play":
        play_winner(pickle_path=args.file, config_file=args.config, level=args.level)


if __name__ == "__main__":
    main()

