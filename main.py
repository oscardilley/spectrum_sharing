#!/usr/bin/python3
""" main.py

Closed loop simulator for testing reinforcement learning models with Sionna. 

"""

import tensorflow as tf
import sionna
from time import perf_counter
from hydra import compose, initialize 
import numpy as np
# import gymnasium as gym

# from plotting import plot_motion, plot_performance, plot_rewards
# from utils import update_users, get_throughput, get_spectral_efficiency, get_power_efficiency, get_spectrum_utility
# from scenario_simulator import FullSimulator
from RL_simulator import SionnaEnv

def main(cfg):
    """Run the simulator."""
    # Initalisation
    # e = 0
    # users={}
    # performance=[]
    # rewards = tf.zeros(shape=(cfg.episodes, 4), dtype=tf.float32)
    # fig_0, fig_1, fig_2 = None, None, None
    # ax_0, ax_1, ax_2 = None, None, None
    # transmitters = dict(cfg.transmitters)
    # num_tx = len(transmitters)
    # sharing_state = tf.ones(shape=(num_tx), dtype=tf.bool)
    # max_results_length = cfg.max_results_length
    # primary_bandwidth = cfg.primary_fft_size * cfg.primary_subcarrier_spacing
    # sharing_bandwidth = cfg.primary_fft_size * cfg.primary_subcarrier_spacing

    # Starting simulator
    env = SionnaEnv(cfg)
    state = env.reset()
    print("Initial state: ", state)
    for e in range(cfg.episodes):
        print("Starting Episode: ", e)
        start = perf_counter()
        while(True):
            random = tf.random.uniform(shape=(2,), minval=0, maxval=1)
            action = random >= 0.5    
            if np.all(action.numpy() == False):
                # in the future, shortcut this and go straight to rewards without calculation
                continue
            else:
                break
        
        print("Action: ", action)
        state, reward, done, _, _ = env.step(action)
        env.render()
        end = perf_counter()
        print(f"\t{round(end-start, 5)}s elapsed.")


    return

if __name__ == "__main__":
    # Configuration
    random_seed = 40
    sionna.config.xla_compat=True
    sionna.config.seed=random_seed

    with initialize(version_base=None, config_path="conf", job_name="simulation"):
        config = compose(config_name="simulation")
        #print(OmegaConf.to_yaml(config))
    main(config)