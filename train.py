import os
import neat
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pickle
import multiprocessing as mp
import logging
import visualize

# Import your custom wrapper from gymTools.py
from gymTools import MarioGridWrapper

# Fixed: Gymnasium modern logger property format update
gym.logger.min_level = 40 


class Train:
    def __init__(self, generations, parallel=2, level="1-1"):
        # Use a small forward-focused action set with explicit jump choices.
        # SIMPLE_MOVEMENT indices: 1=right, 3=right+B, 4=right+A+B
        self.actions = [1, 3, 4]
        self.generations = generations
        self.lock = mp.Lock()
        self.par = parallel
        self.level = level

    def _get_actions(self, a):
        return self.actions[a.index(max(a))]

    def _fitness_func_no_parallel(self, genomes, config):
        base_env = gym_super_mario_bros.make(f'SuperMarioBros-{self.level}-v0', render_mode=None)
        env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
        env = MarioGridWrapper(env, flatten=True)
        
        idx, genomes = zip(*genomes)
        for genome in genomes:
            try:
                state, info = env.reset()
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                done = False
                i = 0
                old = 40
                
                while not done:
                    state = state.flatten() 
                    output = net.activate(state)
                    output = self._get_actions(output)
                    
                    state, reward, terminated, truncated, info = env.step(output)
                    done = terminated or truncated
                    
                    i += 1
                    if i % 50 == 0:
                        if old == info['x_pos']:
                            break
                        else:
                            old = info['x_pos']

                fitness = -1 if info['x_pos'] <= 40 else info['x_pos']
                genome.fitness = fitness
                
            except KeyboardInterrupt:
                env.close()
                return
        env.close()

    def _fitness_func(self, genome, config, o):
        base_env = gym_super_mario_bros.make(f'SuperMarioBros-{self.level}-v0', render_mode=None)
        env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
        env = MarioGridWrapper(env, flatten=True)
        
        try:
            state, info = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            stagnant_frames = 0
            max_stagnant_frames = 150
            old_x = info['x_pos']
            
            while not done:
                state = state.flatten()
                output = net.activate(state)
                output = self._get_actions(output)
                
                state, reward, terminated, truncated, info = env.step(output)
                done = terminated or truncated

                if info['x_pos'] > old_x:
                    old_x = info['x_pos']
                    stagnant_frames = 0
                else:
                    stagnant_frames += 1

                # End hopeless runs, but give enough time to discover jump timing.
                if stagnant_frames >= max_stagnant_frames:
                    break

            fitness = -1 if info['x_pos'] <= 40 else info['x_pos']
            
            if fitness >= 3252 or info.get('flag_get', False):
                pickle.dump(genome, open("finisher.pkl", "wb"))
                print("\nDone! Level Cleared by a Genome!")
                # FIXED: Return data to queue instead of forcing a dead exit to prevent hang
                o.put(fitness) 
                env.close()
                return
                
            o.put(fitness)
            env.close()
            
        except KeyboardInterrupt:
            env.close()
            o.put(-1)
        except Exception as e:
            env.close()
            o.put(-1)

    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)

        for i in range(0, len(genomes), self.par):
            output = mp.Queue()
            chunk = genomes[i:i + self.par]

            processes = [mp.Process(target=self._fitness_func, args=(genome, config, output)) for genome in chunk]

            [p.start() for p in processes]
            [p.join() for p in processes]

            # Fixed: Safely drain the queue without stalling if errors occurred
            results = []
            for _ in range(len(processes)):
                if not output.empty():
                    results.append(output.get())
                else:
                    results.append(-1)

            for n, r in enumerate(results):
                if i + n < len(genomes):
                    genomes[i + n].fitness = r

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.Checkpointer(5))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        winner = p.run(self._eval_genomes, n)
        
        # Save output pickle assets cleanly
        pickle.dump(winner, open('winner.pkl', 'wb'))

        # Fixed: Pass prune_unused=True to keep Graphviz from choking on 208 inputs
        try:
            visualize.draw_net(config, winner, view=False, filename="winner_net", prune_unused=True)
            visualize.plot_stats(stats, ylog=False, view=False, filename="avg_fitness.svg")
            visualize.plot_species(stats, view=False, filename="speciation.svg")
        except Exception as e:
            print(f"Skipped rendering diagrams due to layout environment variables: {e}")

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)


if __name__ == "__main__":
    t = Train(generations=1000, parallel=4)
    t.main()