#!/usr/bin/python3
""" main.py

Closed loop simulator for testing reinforcement learning models with Sionna. 

"""

import tensorflow as tf
import sionna
from time import perf_counter
from hydra import compose, initialize 
import numpy as np
import gymnasium as gym

from plotting import plot_motion, plot_performance, plot_rewards
from utils import update_users, get_throughput, get_spectral_efficiency, get_power_efficiency, get_spectrum_utility
from scenario_simulator import FullSimulator

def main(cfg):
    """Run the simulator."""
    # Initalisation
    e = 0
    users={}
    performance=[]
    rewards = tf.zeros(shape=(cfg.episodes, 4), dtype=tf.float32)
    fig_0, fig_1, fig_2 = None, None, None
    ax_0, ax_1, ax_2 = None, None, None
    transmitters = dict(cfg.transmitters)
    num_tx = len(transmitters)
    sharing_state = tf.ones(shape=(num_tx), dtype=tf.bool)
    max_results_length = cfg.max_results_length
    primary_bandwidth = cfg.primary_fft_size * cfg.primary_subcarrier_spacing
    sharing_bandwidth = cfg.primary_fft_size * cfg.primary_subcarrier_spacing

    # Starting simulator
    while e < cfg.episodes:
        start = perf_counter()
        print(f"Episode {e}")

        if e == 0:
            print("Initialising")
            # Setting up the primary coverage bands for each transmitter - note this all needs adapting for > 2 transmitters.
            primaryBand1 = FullSimulator(prefix="primary",
                                         scene_name=sionna.rt.scene.simple_street_canyon,
                                         carrier_frequency=cfg.primary_carrier_freq_1,
                                         bandwidth=primary_bandwidth,
                                         pmax=50, # maximum power
                                         transmitters=transmitters,
                                         num_rx = cfg.num_rx,
                                         max_depth=cfg.max_depth,
                                         cell_size=cfg.cell_size,
                                         initial_state = tf.convert_to_tensor([True, False], dtype=tf.bool),
                                         subcarrier_spacing = cfg.primary_subcarrier_spacing,
                                         fft_size = cfg.primary_fft_size,
                                         batch_size=cfg.batch_size,
                                         )
            primaryBand2 = FullSimulator(prefix="primary",
                                         scene_name=sionna.rt.scene.simple_street_canyon,
                                         carrier_frequency=cfg.primary_carrier_freq_2,
                                         bandwidth=primary_bandwidth,
                                         pmax=50, # maximum power
                                         transmitters=transmitters,
                                         num_rx = cfg.num_rx,
                                         max_depth=cfg.max_depth,
                                         cell_size=cfg.cell_size,
                                         initial_state = tf.convert_to_tensor([False, True], dtype=tf.bool),
                                         subcarrier_spacing = cfg.primary_subcarrier_spacing,
                                         fft_size = cfg.primary_fft_size,
                                         batch_size=cfg.batch_size,
                                         )
            # Setting up the sharing band
            sharingBand = FullSimulator(prefix="sharing",
                                        scene_name=sionna.rt.scene.simple_street_canyon,
                                        carrier_frequency=cfg.sharing_carrier_freq,
                                        bandwidth=sharing_bandwidth,
                                        pmax=50, # maximum power
                                        transmitters=transmitters,
                                        num_rx = cfg.num_rx,
                                        max_depth=cfg.max_depth,
                                        cell_size=cfg.cell_size,
                                        initial_state = sharing_state,
                                        subcarrier_spacing = cfg.sharing_subcarrier_spacing,
                                        fft_size = cfg.sharing_fft_size,
                                        batch_size=cfg.batch_size,
                                        )
            
            valid_area = tf.math.logical_or(tf.math.logical_or(primaryBand1.grid, primaryBand2.grid), sharingBand.grid) # shape [y_max, x_max]
            
        # Generating the initial user positions based on logical OR of validity matrices
        users = update_users(valid_area, cfg.num_rx, users)

        # Running the simulation - with separated primary bands
        primaryOutput1 = primaryBand1(users, tf.convert_to_tensor([True, False], dtype=tf.bool), transmitters) 
        primaryOutput2 = primaryBand2(users, tf.convert_to_tensor([False, True], dtype=tf.bool), transmitters)
        sharingOutput = sharingBand(users, sharing_state, transmitters)

        # Combining the primary bands for the different transmitters:
        primaryOutput = {"bler": tf.stack([primaryOutput1["bler"][0,:], primaryOutput2["bler"][1,:]]), "sinr": tf.stack([primaryOutput1["sinr"][0,:], primaryOutput2["sinr"][1,:]])}
        performance.append({"Primary": primaryOutput, "Sharing": sharingOutput})

        # Plotting the performance and motion
        if len(performance) > max_results_length: # managing stored results size
            performance = performance[-1*max_results_length:]

        # Plotting the performance
        if e >= 1:
            plot_performance(episode=e,
                            users=users,
                            performance=performance, 
                            save_path=cfg.images_path)
            
        primary_sinr_map_1 = primaryBand1.sinr
        primary_sinr_map_2 = primaryBand2.sinr
        sharing_sinr_map = sharingBand.sinr
        fig_0, ax_0 = plot_motion(episode=e, 
                                  id="Primary Band 1, SINR", 
                                  grid=valid_area, 
                                  cm=tf.reduce_max(primary_sinr_map_1, axis=0), 
                                  color="inferno",
                                  users=users, 
                                  transmitters=transmitters, 
                                  cell_size=cfg.cell_size, 
                                  fig=fig_0,
                                  ax=ax_0, 
                                  save_path=cfg.images_path)
        fig_1, ax_1 = plot_motion(episode=e, 
                                  id="Primary Band 2, SINR", 
                                  grid=valid_area, 
                                  cm=tf.reduce_max(primary_sinr_map_2, axis=0), 
                                  color="inferno",
                                  users=users, 
                                  transmitters=transmitters, 
                                  cell_size=cfg.cell_size, 
                                  fig=fig_1,
                                  ax=ax_1, 
                                  save_path=cfg.images_path)
        fig_2, ax_2  = plot_motion(episode=e, 
                                  id="Sharing Band, Max SINR", 
                                  grid=valid_area, 
                                  cm=tf.reduce_max(sharing_sinr_map, axis=0), 
                                  color="viridis",
                                  users=users, 
                                  transmitters=transmitters, 
                                  cell_size=cfg.cell_size, 
                                  fig=fig_2,
                                  ax=ax_2, 
                                  save_path=cfg.images_path)

        # Calculating rewards
        rates = tf.stack([primaryOutput1["rate"], primaryOutput2["rate"], sharingOutput["rate"]])
        throughput, per_ue_throughput, per_ap_per_band_throughput = get_throughput(rates)

        primary_power = tf.convert_to_tensor(np.power(10, (np.array([tx["primary_power"] for tx in transmitters.values()]) - 30) / 10), dtype=tf.float32)
        sharing_power = tf.convert_to_tensor(np.power(10, (np.array([tx["sharing_power"] for tx in transmitters.values()]) - 30) / 10), dtype=tf.float32)
        mu_pa = tf.convert_to_tensor([tx["mu_pa"] for tx in transmitters.values()])
        pe, per_ap_pe = get_power_efficiency(primary_bandwidth, # integral over power efficiency over time is energy efficiency
                                                   sharing_bandwidth,
                                                   sharing_state,
                                                   primary_power,
                                                   sharing_power,
                                                   mu_pa)

        se, per_ap_se = get_spectral_efficiency(primary_bandwidth, 
                                                      sharing_bandwidth,
                                                      per_ap_per_band_throughput)
        
        su = get_spectrum_utility(primary_bandwidth,
                                  sharing_bandwidth,
                                  sharing_state,
                                  throughput)
        
        # Plotting objectives/ rewards
        indices = tf.constant([[e, 0], [e, 1], [e, 2], [e, 3]])
        updates = tf.stack([throughput, se, pe, su], axis=0)
        rewards = tf.tensor_scatter_nd_update(rewards, indices, tf.reshape(updates, (4,)))
        
        plot_rewards(episode=e,
                     rewards=rewards,
                     save_path=cfg.images_path)
        



        # Decision making:

        # NB: we don't need to take a decision every time

        # NB: need to consider the impact of time

        if e == 2:
            sharing_state = tf.convert_to_tensor([False, True], dtype=tf.bool)
        elif e == 4: 
            sharing_state = tf.convert_to_tensor([True, False], dtype=tf.bool)
        transmitters = transmitters

        # Iterate to next episode
        e += 1
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