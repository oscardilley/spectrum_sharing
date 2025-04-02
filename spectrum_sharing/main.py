#!/usr/bin/python3
""" main.py

Closed loop simulator for testing reinforcement learning models with Sionna. 

Must run as a module: python3 -m spectrum_sharing.main

"""

import tensorflow as tf
import sionna
from time import perf_counter
from hydra import compose, initialize 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING not displayed
import matplotlib.pyplot as plt

from spectrum_sharing.RL_simulator import SionnaEnv
from spectrum_sharing.DQN_agent import Agent, ReplayBuffer
from spectrum_sharing.logger import logger
from spectrum_sharing.plotting import plot_total_rewards

def main(cfg):
    """Run the simulator."""
    # Starting simulator
    env = SionnaEnv(cfg)
    buffer = ReplayBuffer(cfg.buffer_max_size, cfg.buffer_path)
    agent = Agent(cfg,
                  num_tx=len(cfg.transmitters),
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  possible_actions=env.possible_actions,
                  num_possible_actions=env.num_actions,
                  path=cfg.models_path)
    reward_per_episode = [[None for _ in range(cfg.step_limit + 1)] for e in range(int(cfg.episodes))]
    avg_throughput_per_episode = [0.0 for e in range(int(cfg.episodes))]
    avg_fairness_per_episode = [0.0 for e in range(int(cfg.episodes))]
    avg_pe_per_episode = [0.0 for e in range(int(cfg.episodes))]
    avg_se_per_episode = [0.0 for e in range(int(cfg.episodes))]
    avg_su_per_episode = [0.0 for e in range(int(cfg.episodes))]

    for e in range(int(cfg.episodes)):
        logger.info(f"Starting Episode: {e}")
        observation = env.reset() #seed=cfg.random_seed)

        while True:
            start = perf_counter()        
            # Taking action
            logger.info(f"Step: {env.timestep}") 
            action, action_id = agent.act(observation)
            logger.info(f"Action: {action}")
            next_observation, reward, terminated, truncated, info = env.step(action) 

            if next_observation is None: 
                logger.critical("Exiting episode after error to prevent propagation.")
                break

            buffer.add((observation, action_id, reward, next_observation, terminated), env.timestep) # consider adding truncated OR terminated
            logger.info(f"Reward: {reward}")
            observation = next_observation
            env.render(episode=e) # rendering post action, images show end of round

            agent.train(buffer, cfg.training_batch_size, env.timestep)

            # Storing and plotting reward information
            reward_per_episode[e][env.timestep] = reward 
            avg_throughput_per_episode[e] += info["rewards"][0].numpy() / float(cfg.step_limit)
            avg_fairness_per_episode += info["rewards"][1].numpy() / float(cfg.step_limit)
            avg_se_per_episode[e] += info["rewards"][2].numpy() / float(cfg.step_limit)
            avg_pe_per_episode[e] += info["rewards"][3].numpy() / float(cfg.step_limit)
            avg_su_per_episode[e] += info["rewards"][4].numpy() / float(cfg.step_limit)

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

        # Visualisation to track training performance
        plot_total_rewards(episode=e,
                           reward=np.array(reward_per_episode),
                           throughput=avg_throughput_per_episode,
                           fairness=avg_fairness_per_episode,
                           se=avg_se_per_episode,
                           pe=avg_pe_per_episode,
                           su=avg_su_per_episode,
                           save_path=cfg.images_path)
        continue

    logger.critical(f"Completed {e} episodes. Exiting.")
    return

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f'Number of GPUs available : {len(gpus)}')
    if gpus:
        gpu_num = 0 # Index of the GPU to be used
        try:
            tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
            logger.warning(f'Only GPU number {gpu_num} used.')
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True) # manages memory growth
        except RuntimeError as e:
            logger.critical(e)
    with initialize(version_base=None, config_path="Config", job_name="simulation"):
        config = compose(config_name="simulation")
        sionna.config.xla_compat=True
        # # Use for determinism:
        # sionna.config.seed=config.random_seed
        # tf.random.set_seed(config.random_seed)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        # np.random.seed(config.random_seed)        
        logger.info(f"Config:\n{config}\n")

    main(config)