#!/usr/bin/python3
""" benchmark.py

Script to test RL model with zero exploration against baseline scenarios. 
Runs a single episode deterministically.

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
from plotting import plot_total_rewards

def main(cfg):
    """Run the simulator."""
    # Starting simulator
    env = SionnaEnv(cfg)
    # buffer = ReplayBuffer(cfg.buffer_max_size, cfg.log_path) # not required as no training
    agent = Agent(cfg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  path=cfg.test_models_path,
                  test=True)

    test = {"Agent": None, 
            "TX 0 ON, Max": (np.array([1, 0], dtype=np.int8), 34, 34), 
            "TX 0 ON, Min": (np.array([1, 0], dtype=np.int8), 25, 25), 
            "TX 1 ON, Max": (np.array([0, 1], dtype=np.int8), 34, 34), 
            "TX 1 ON, Min": (np.array([0, 1], dtype=np.int8), 25, 25)}
    
    reward_per_test = [[] for e in range(len(test))] # need to change to not be per episode
    avg_reward_per_test = [0.0 for e in range(len(test))] # need to change to not be per episode
    min_reward_per_test = [10000 for e in range(len(test))]
    max_reward_per_test = [0.0 for e in range(len(test))]
    avg_throughput_per_test = [0.0 for e in range(len(test))]
    avg_pe_per_test = [0.0 for e in range(len(test))]
    avg_se_per_test = [0.0 for e in range(len(test))]
    avg_su_per_test = [0.0 for e in range(len(test))]
            
    for test_id, test_name in enumerate(test.keys()):
        logger.info(f"Starting test {test_id}.")
        observation = env.reset(seed=cfg.random_seed) # deterministic
        e = test_name
        while True:
            start = perf_counter()        
            # Taking action
            logger.info(f"Step: {env.timestep}") 
            if test[test_name] is None:
                action, action_id = agent.act(observation)
            else:
                action = test[test_name]
            logger.info(f"Action: {action}")
            next_observation, reward, terminated, truncated, info = env.step(action) 

            if next_observation is None: 
                logger.critical("Exiting episode after error to prevent propagation.")
                break

            logger.info(f"Reward: {reward}")
            observation = next_observation
            env.render(episode=e) # rendering post action, images show end of round


            # Storing and plotting reward information
            reward_per_test[test_id].append(reward)
            avg_reward_per_test[test_id] += reward / float(cfg.step_limit)
            min_reward_per_test[test_id] = min(reward, min_reward_per_test[test_id])
            max_reward_per_test[test_id] = max(reward, max_reward_per_test[test_id])
            avg_throughput_per_test[test_id] += info["rewards"][0].numpy() / float(cfg.step_limit)
            avg_se_per_test[test_id] += info["rewards"][1].numpy() / float(cfg.step_limit)
            avg_pe_per_test[test_id] += info["rewards"][2].numpy() / float(cfg.step_limit)
            avg_su_per_test[test_id] += info["rewards"][3].numpy() / float(cfg.step_limit)
          
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
        # plot_total_rewards(episode=e,
        #                     reward=avg_reward_per_episode,
        #                     reward_min=min_reward_per_episode,
        #                     reward_max=max_reward_per_episode,
        #                     throughput=avg_throughput_per_episode,
        #                     se=avg_se_per_episode,
        #                     pe=avg_pe_per_episode,
        #                     su=avg_su_per_episode,
        #                     save_path=cfg.images_path)
    

    logger.info(f"Reward per test: {reward_per_test}")
    logger.info(f"Avg Reward per test: {avg_reward_per_test}")
    logger.info(f"Min Reward per test: {min_reward_per_test}")
    logger.info(f"Max Reward per test: {max_reward_per_test}")
    logger.info(f"Avg throughput per test: {avg_throughput_per_test}")
    logger.info(f"Avg se per test: {avg_se_per_test}")
    logger.info(f"Avg pe per test: {avg_pe_per_test}")
    logger.info(f"Avg su per testt: {avg_su_per_test}")
    logger.critical(f"Completed test. Exiting.")
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