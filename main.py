#!/usr/bin/python3
""" main.py

Closed loop simulator for testing reinforcement learning models with Sionna. 

"""

import tensorflow as tf
import sionna
from time import perf_counter
from hydra import compose, initialize 

from plotting import plot_motion, plot_performance
from utils import update_users
from scenario_simulator import FullSimulator

def main(cfg):
    """Run the simulator."""
    # Initalisation
    e = 0
    users={}
    performance=[]
    fig_0, fig_1 = None, None
    ax_0, ax_1 = None, None
    transmitters = dict(cfg.transmitters)
    num_tx = len(transmitters)
    primaryState = tf.ones(shape=(num_tx), dtype=tf.bool)
    sharingState = tf.ones(shape=(num_tx), dtype=tf.bool)
    max_results_length = cfg.max_results_length

    # Starting simulator
    while e < 10:
        start = perf_counter()
        print(f"Episode {e}")

        if e == 0:
            print("Initialising")
            primaryBand = FullSimulator(scene_name=sionna.rt.scene.simple_street_canyon,
                                        carrier_frequency=cfg.prim_carrier_freq,
                                        bandwidth=cfg.prim_bandwidth,
                                        pmax=50, # maximum power
                                        transmitters=transmitters,
                                        num_rx = cfg.num_rx,
                                        max_depth=cfg.max_depth,
                                        cell_size=cfg.cell_size,
                                        initial_state = primaryState,
                                        subcarrier_spacing = cfg.prim_subcarrier_spacing,
                                        fft_size = cfg.prim_fft_size,
                                        )
            sharingBand = FullSimulator(scene_name=sionna.rt.scene.simple_street_canyon,
                                        carrier_frequency=cfg.sharing_carrier_freq,
                                        bandwidth=cfg.sharing_bandwidth,
                                        pmax=50, # maximum power
                                        transmitters=transmitters,
                                        num_rx = cfg.num_rx,
                                        max_depth=cfg.max_depth,
                                        cell_size=cfg.cell_size,
                                        initial_state = sharingState,
                                        subcarrier_spacing = cfg.sharing_subcarrier_spacing,
                                        fft_size = cfg.sharing_fft_size,
                                        )
            valid_area = tf.math.logical_or(primaryBand.grid, sharingBand.grid) # shape [y_max, x_max]
            
        # Generating the initial user positions based on logical OR of validity matrices
        users = update_users(valid_area, cfg.num_rx, users)
        # Update the transmitters here if required for power etc.
        primary_sinr_map = primaryBand.cm.sinr
        sharing_sinr_map = sharingBand.cm.sinr
        fig_0, ax_0 = plot_motion(episode=e, 
                                  id="Primary Band, Max SINR - ", 
                                  grid=valid_area, 
                                  cm=tf.reduce_max(primary_sinr_map, axis=0), 
                                  color="inferno",
                                  users=users, 
                                  transmitters=transmitters, 
                                  cell_size=cfg.cell_size, 
                                  fig=fig_0,
                                  ax=ax_0, 
                                  save_path=cfg.images_path)
        fig_1, ax_1  = plot_motion(episode=e, 
                                  id="Sharing Band, Max SINR - ", 
                                  grid=valid_area, 
                                  cm=tf.reduce_max(sharing_sinr_map, axis=0), 
                                  color="viridis",
                                  users=users, 
                                  transmitters=transmitters, 
                                  cell_size=cfg.cell_size, 
                                  fig=fig_1,
                                  ax=ax_1, 
                                  save_path=cfg.images_path)
       
        # Running the simulation
        primaryOutput = primaryBand(users, primaryState) # optionally feed in transmitters if changing
        sharingOutput = sharingBand(users, sharingState)
        performance.append({"Primary": primaryOutput, "Sharing": sharingOutput})
        if len(performance) > max_results_length: # managing stored results size
            performance = performance[-1*max_results_length:]

        # Plotting the performance
        if e >= 1:
            plot_performance(episode=e,
                            users=users,
                            performance=performance, 
                            save_path=cfg.images_path)

        # Processing the outputs (over time) and determining next state

        # Decision making:
        primaryState = tf.ones(shape=(num_tx), dtype=tf.bool)
        sharingState = tf.ones(shape=(num_tx), dtype=tf.bool)

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