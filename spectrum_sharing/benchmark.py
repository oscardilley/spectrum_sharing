""" benchmark.py

Script to test RL model with zero exploration against baseline scenarios. 
Runs a single episode deterministically. Needs to be separated for repeatability.

Run as module: python3 -m spectrum_sharing.benchmark <test_index> <seed>

"""

import tensorflow as tf
import sionna
from time import perf_counter
from hydra import compose, initialize 
import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime
import pathlib

from spectrum_sharing.RL_simulator import SionnaEnv, PrecomputedEnv
from spectrum_sharing.DQN_agent import Agent
from spectrum_sharing.logger import logger

CONFIG_NAME = "simulation5" # the only config selection in the script

tests = {"Agent": None, 
        "TX 0 ON, Avg": ((1, 1), (0, 1)), 
        "TX 1 ON, Avg": ((0, 1), (1, 1)), 
        "Both ON, Avg": ((1, 1), (1, 1))} # static power levels determined by start levels in config

def main(cfg, test_index):
    """Run the simulator."""
    # Starting simulator
    if bool(cfg.use_pre_gen_maps):
        logger.warning("Using precomputed maps.")
        env = PrecomputedEnv(cfg, True)
    else:
        env = SionnaEnv(cfg, True)
    agent = Agent(cfg,
                  num_tx=len(cfg.transmitters),
                  observation_space=env.observation_space,
                  possible_actions=env.possible_actions,
                  num_possible_actions=env.num_actions,
                  path=cfg.test_models_path,
                  test=True)

    values = list(tests.items())
    if values[test_index][1] is None:
        test = None
    else:
        test = values[test_index][1]
    label = values[test_index][0]
    e = 0 # episode zero, triggers timestamping
    
    avg_reward_per_test = 0.0 # need to change to not be per episode
    min_reward_per_test = 10000
    max_reward_per_test = 0.0
    avg_throughput_per_test = 0.0
    avg_fairness_per_test = 0.0
    avg_pe_per_test = 0.0
    avg_se_per_test = 0.0
    avg_su_per_test = 0.0
            
    logger.info(f"Starting test {label}, {test_index}.")
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
        # env.render(episode=test_index) 

        # Storing and plotting reward information
        avg_reward_per_test += reward / float(cfg.step_limit)
        min_reward_per_test = min(reward, min_reward_per_test)
        max_reward_per_test = max(reward, max_reward_per_test)
        avg_throughput_per_test += info["rewards"][0].numpy() / float(cfg.step_limit)
        avg_fairness_per_test += info["rewards"][1].numpy() / float(cfg.step_limit)
        avg_se_per_test += info["rewards"][2].numpy() / float(cfg.step_limit)
        avg_pe_per_test += info["rewards"][3].numpy() / float(cfg.step_limit)
        avg_su_per_test += info["rewards"][4].numpy() / float(cfg.step_limit)

        # Clearing up
        if terminated or truncated:
            logger.warning("Episode terminated or truncated. Resetting Env.") 
            env.truncated = False # resetting the false flag
            env.terminated = False  
            end = perf_counter()
            logger.info(f"{round(end-start, 5)}s elapsed.")
            break

        env.timestep += 1

        # Noting run time
        end = perf_counter()
        logger.info(f"{round(end-start, 5)}s elapsed.")

    logger.info(f"Avg Reward: {avg_reward_per_test}")
    logger.info(f"Min Reward: {min_reward_per_test}")
    logger.info(f"Max Reward: {max_reward_per_test}")
    logger.info(f"Avg throughput: {avg_throughput_per_test}")
    logger.info(f"Avg se: {avg_se_per_test}")
    logger.info(f"Avg pe: {avg_pe_per_test}")
    logger.info(f"Avg su: {avg_su_per_test}")
    logger.critical(f"Completed test. Saving and Exiting.")

    results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avg_throughput": avg_throughput_per_test,
            "avg_fairness": avg_fairness_per_test,
            "avg_se": avg_se_per_test,
            "avg_pe": avg_pe_per_test,
            "avg_su": avg_su_per_test,
            "avg_reward": avg_reward_per_test.numpy(),
            "min_reward": min_reward_per_test.numpy(),
            "max_reward": max_reward_per_test.numpy(), 
            "seed": cfg.random_seed,
            "test_label": label,
        }

    # Convert to a single-row DataFrame
    df = pd.DataFrame([results])

    # Create aggregated results file
    pathlib.Path(cfg.test_path).mkdir(parents=True, exist_ok=True)
    # Write results to temp file to prevent partial writes etc when running in parallel
    result_file_path = os.path.join(cfg.test_path, f"temp_seed{cfg.random_seed}_{label}.csv")
    df.to_csv(result_file_path, index=False)

    logger.critical(f"Appended results to {result_file_path}")

    return

if __name__ == "__main__":
    test_id = int(sys.argv[1])
    seed = int(sys.argv[2])
    with initialize(version_base=None, config_path="Config", job_name=CONFIG_NAME):
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
        config = compose(config_name=CONFIG_NAME)
        config.random_seed = seed # overwriting random seed
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

    main(config, test_id)