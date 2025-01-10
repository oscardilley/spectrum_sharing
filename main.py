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

def main(cfg):
    """Run the simulator."""
    # Starting simulator
    env = SionnaEnv(cfg)
    initial_state = env.reset(seed=cfg.random_seed)
    action = tf.convert_to_tensor([True for _ in range(len(cfg.transmitters))], dtype=tf.bool)
    print("Initial transmitter state: ", action)
    for e in range(cfg.episodes):
        print("Starting Episode: ", e)
        start = perf_counter()
        state, reward, done, _, _ = env.step(action)
        while(True):
            random = tf.random.uniform(shape=(len(cfg.transmitters),), minval=0, maxval=1)
            action = random >= 0.5    
            if np.all(action.numpy() == False):
                # in the future, shortcut this and go straight to rewards without calculation
                continue
            else:
                # Used to ensure number of matplotlib figures is managed
                # print(plt.get_fignums())
                # for fig_num in plt.get_fignums():
                #     print(f"FIGURE {fig_num}")
                #     fig = plt.figure(fig_num)  # Access the figure by its number
                #     title = fig._suptitle.get_text() if fig._suptitle else "No Title"
                #     for ax in plt.figure(fig_num).axes:
                #         print(ax.get_title())
                #     print(title)
                break
        
        print("Next Action: ", action)
        env.render()
        end = perf_counter()
        print(f"\t{round(end-start, 5)}s elapsed.")


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