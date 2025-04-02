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
import sys

from RL_simulator import SionnaEnv
from DQN_agent import Agent
from logger import logger

def main(cfg, test_index):
    """Run the simulator."""
    # Starting simulator
    env = SionnaEnv(cfg)
    # buffer = ReplayBuffer(cfg.buffer_max_size, cfg.log_path) # not required as no training
    agent = Agent(cfg,
                  num_tx=len(cfg.transmitters),
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  possible_actions=env.possible_actions,
                  num_possible_actions=env.num_actions,
                  path=cfg.test_models_path,
                  test=True)

    tests = {"Agent": None,
            "TX 0 ON, Avg": ((1, 1), (0, 1)), 
            #"TX 0 ON, Min": (np.array([1, 1], dtype=np.int8), np.array([0, 1], dtype=np.int8)), 
            "TX 1 ON, Avg": ((0, 1), (1, 1)), 
            #"TX 1 ON, Min": (np.array([0, 1], dtype=np.int8), np.array([1, 1], dtype=np.int8)),
            #"Both ON, Min": (np.array([1, 1], dtype=np.int8), np.array([1, 1], dtype=np.int8)),
            "Both ON, Avg": ((1, 1), (1, 1))}
    
    values = list(tests.items())
    if values[test_index][1] is None:
        test = None
    else:
        test = values[test_index][1]
    e = values[test_index][0]
    
    avg_reward_per_test = 0.0 # need to change to not be per episode
    min_reward_per_test = 10000
    max_reward_per_test = 0.0
    avg_throughput_per_test = 0.0
    avg_pe_per_test = 0.0
    avg_se_per_test = 0.0
    avg_su_per_test = 0.0
            
    logger.info(f"Starting test {test_index}.")
    observation = env.reset(seed=cfg.random_seed) # deterministic
    while True:
        start = perf_counter()        
        # Taking action
        logger.info(f"Step: {env.timestep}") 
        if test is None:
            action, action_id = agent.act(observation)
        else:
            action = test
        logger.info(f"Action: {action}")
        next_observation, reward, terminated, truncated, info = env.step(action) 

        if next_observation is None: 
            logger.critical("Exiting episode after error to prevent propagation.")
            break

        logger.info(f"Reward: {reward}")
        observation = next_observation
        env.render(episode=e) # rendering post action, images show end of round


        # Storing and plotting reward information
        avg_reward_per_test += reward / float(cfg.step_limit)
        min_reward_per_test = min(reward, min_reward_per_test)
        max_reward_per_test = max(reward, max_reward_per_test)
        avg_throughput_per_test += info["rewards"][0].numpy() / float(cfg.step_limit)
        avg_se_per_test += info["rewards"][1].numpy() / float(cfg.step_limit)
        avg_pe_per_test += info["rewards"][2].numpy() / float(cfg.step_limit)
        avg_su_per_test += info["rewards"][3].numpy() / float(cfg.step_limit)
        
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
 
    logger.info(f"Avg Reward: {avg_reward_per_test}")
    logger.info(f"Min Reward: {min_reward_per_test}")
    logger.info(f"Max Reward: {max_reward_per_test}")
    logger.info(f"Avg throughput: {avg_throughput_per_test}")
    logger.info(f"Avg se: {avg_se_per_test}")
    logger.info(f"Avg pe: {avg_pe_per_test}")
    logger.info(f"Avg su: {avg_su_per_test}")
    logger.critical(f"Completed test. Exiting.")

    return

if __name__ == "__main__":
    with initialize(version_base=None, config_path="conf", job_name="simulation"):
        config = compose(config_name="simulation")
        sionna.config.xla_compat=True
        sionna.config.seed=config.random_seed
        # Below all helps tensorflow determinism with the GPU
        tf.random.set_seed(config.random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'  # Ensures single-threaded execution
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
        np.random.seed(config.random_seed)        
        logger.info(f"Config:\n{config}\n")

    test_id = int(sys.argv[1])
    main(config, test_id)