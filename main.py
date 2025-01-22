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
from logger import logger

def main(cfg):
    """Run the simulator."""
    # Starting simulator
    env = SionnaEnv(cfg)
    buffer = ReplayBuffer(cfg.buffer_max_size, cfg.log_path)
    agent = Agent(cfg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  path=cfg.models_path)

    for e in range(int(cfg.episodes)):
        logger.info(f"Starting Episode: {e}")
        observation = env.reset() #seed=cfg.random_seed)

        while True:
            start = perf_counter()        
            # Taking action
            logger.info(f"Step: {env.timestep}") 
            action, action_id = agent.act(observation)
            logger.info(f"Action: {action}")
            next_observation, reward, terminated, truncated, _ = env.step(action) 
            print(truncated)
            buffer.add((observation, action_id, reward, next_observation, terminated), env.timestep)
            logger.info(f"Reward: {reward}")
            observation = next_observation
            env.render() # rendering post action, images show end of round

            agent.train(buffer, cfg.training_batch_size, env.timestep)

            # Clearing up
            if terminated or truncated:
                logger.warning("Episode terminated or truncated. Resetting Env.") 
                env.truncated = False # resetting the false flag
                env.terminated = False  
                end = perf_counter()
                logger.info(f"{round(end-start, 5)}s elapsed.")
                break

            # Noting run time
            end = perf_counter()
            logger.info(f"{round(end-start, 5)}s elapsed.")

            if env.timestep % cfg.target_network_update_freq == 0:
                logger.info("Updating target network.")
                agent.update_target_network()

            env.timestep += 1

        continue

    logger.critical(f"Completed {e} episodes. Exiting.")
    return

if __name__ == "__main__":
    with initialize(version_base=None, config_path="conf", job_name="simulation"):
        config = compose(config_name="simulation")
        sionna.config.xla_compat=True
        sionna.config.seed=config.random_seed
        # tf.random.set_seed(config.random_seed)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        # np.random.seed(config.random_seed)        
        logger.info(f"Config:\n{config}\n")

    main(config)