#!/usr/bin/python3
""" main.py

Closed loop simulator for testing reinforcement learning models with Sionna. 

"""

import tensorflow as tf
import sionna
from time import perf_counter
from hydra import compose, initialize 
import numpy as np
import os
import matplotlib.pyplot as plt

from RL_simulator import SionnaEnv
from DQN_agent import Agent, ReplayBuffer

def main(cfg):
    """Run the simulator."""
    # Starting simulator
    env = SionnaEnv(cfg)
    buffer = ReplayBuffer(cfg.buffer_max_size)
    agent = Agent(cfg,
                  state_shape=env.observation_space.shape,
                  action_shape=env.action_space.shape)
    done = False

    for e in range(int(cfg.episodes)):
        print("Starting Episode: ", e)
        observation = env.reset(seed=cfg.random_seed)
        env.render()
        step = 0

        while True:
            print("Step: ", env.timestep)
            start = perf_counter()        
            action = env.action_space.sample()


            # Taking action
            agent.act(observation)


            print("Action: ", action)
            next_observation, reward, terminated, truncated, info = env.step(action)
            buffer.add((observation, action, reward, next_observation, terminated))
            observation = next_observation
            env.render()

            agent.train(buffer, cfg.training_batch_size)

            print(reward)

            
            
            # Clearing up
            if terminated or truncated:
                print("Episode terminated or truncated. Resetting Env.")   
                break

            # Noting run time
            step += 1
            end = perf_counter()
            print(f"\t{round(end-start, 5)}s elapsed.")

        if e % cfg.target_update_freq == 0:
            agent.update_target()

    return

if __name__ == "__main__":

    with initialize(version_base=None, config_path="conf", job_name="simulation"):
        config = compose(config_name="simulation")
        #print(OmegaConf.to_yaml(config))
        sionna.config.xla_compat=True
        sionna.config.seed=config.random_seed
        #tf.random.set_seed(config.random_seed)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    main(config)