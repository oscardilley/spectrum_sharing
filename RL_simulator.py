""" RL_simulator.py

Wrapping Sionna logic inside a gymnasium wrapper for reinforcement learning.

"""

import tensorflow as tf
import sionna
from time import perf_counter
from hydra import compose, initialize 
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from plotting import plot_motion, plot_performance, plot_rewards
from utils import update_users, get_throughput, get_spectral_efficiency, get_power_efficiency, get_spectrum_utility
from scenario_simulator import FullSimulator

class SionnaEnv(gym.Env):
    """ Sionna environment for reinforcement learning. """

    def __init__(self, cfg):
        """ Initialisation of the environment. """
        self.cfg = cfg
        self.transmitters = dict(self.cfg.transmitters)
        self.num_tx = len(self.transmitters)
        self.sharing_state = tf.ones(shape=(self.num_tx), dtype=tf.bool)
        self.max_results_length = self.cfg.max_results_length
        self.primary_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.sharing_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.primaryBand1 = FullSimulator(prefix="primary",
                                         scene_name=sionna.rt.scene.simple_street_canyon,
                                         carrier_frequency=self.cfg.primary_carrier_freq_1,
                                         bandwidth=self.primary_bandwidth,
                                         pmax=50, # maximum power
                                         transmitters=self.transmitters,
                                         num_rx = self.cfg.num_rx,
                                         max_depth=self.cfg.max_depth,
                                         cell_size=self.cfg.cell_size,
                                         initial_state = tf.convert_to_tensor([True, False], dtype=tf.bool),
                                         subcarrier_spacing = self.cfg.primary_subcarrier_spacing,
                                         fft_size = self.cfg.primary_fft_size,
                                         batch_size=self.cfg.batch_size,
                                         )
        self.primaryBand2 = FullSimulator(prefix="primary",
                                        scene_name=sionna.rt.scene.simple_street_canyon,
                                        carrier_frequency=self.cfg.primary_carrier_freq_2,
                                        bandwidth=self.primary_bandwidth,
                                        pmax=50, # maximum power
                                        transmitters=self.transmitters,
                                        num_rx = self.cfg.num_rx,
                                        max_depth=self.cfg.max_depth,
                                        cell_size=self.cfg.cell_size,
                                        initial_state = tf.convert_to_tensor([False, True], dtype=tf.bool),
                                        subcarrier_spacing = self.cfg.primary_subcarrier_spacing,
                                        fft_size = self.cfg.primary_fft_size,
                                        batch_size=self.cfg.batch_size,
                                        )
        # Setting up the sharing band
        self.sharingBand = FullSimulator(prefix="sharing",
                                    scene_name=sionna.rt.scene.simple_street_canyon,
                                    carrier_frequency=self.cfg.sharing_carrier_freq,
                                    bandwidth=self.sharing_bandwidth,
                                    pmax=50, # maximum power
                                    transmitters=self.transmitters,
                                    num_rx = self.cfg.num_rx,
                                    max_depth=self.cfg.max_depth,
                                    cell_size=self.cfg.cell_size,
                                    initial_state = self.sharing_state,
                                    subcarrier_spacing = self.cfg.sharing_subcarrier_spacing,
                                    fft_size = self.cfg.sharing_fft_size,
                                    batch_size=self.cfg.batch_size,
                                    )
        
    def reset(self, seed=None, options=None):
        """ Reset the environment to its initial state. """
        self.e = 0
        self.users={}
        self.performance=[]
        self.fig_0, self.fig_1, self.fig_2 = None, None, None
        self.ax_0, self.ax_1, self.ax_2 = None, None, None
        self.rewards = tf.zeros(shape=(self.cfg.episodes, 4), dtype=tf.float32)
        self.primaryBand1.reset()
        self.primaryBand2.reset()
        self.sharingBand.reset()
        self.valid_area = tf.math.logical_or(tf.math.logical_or(self.primaryBand1.grid, self.primaryBand2.grid), self.sharingBand.grid) # shape [y_max, x_max]

        return self.sharing_state

    def step(self, action):
        """ Step through the environment. """
        self.sharing_state = action

        # Update the transmission power (up to the max, in fix granuals)


        # Generating the initial user positions based on logical OR of validity matrices
        self.users = update_users(self.valid_area, self.cfg.num_rx, self.users)

        # Running the simulation - with separated primary bands
        primaryOutput1 = self.primaryBand1(self.users, tf.convert_to_tensor([True, False], dtype=tf.bool), self.transmitters) 
        primaryOutput2 = self.primaryBand2(self.users, tf.convert_to_tensor([False, True], dtype=tf.bool), self.transmitters)
        sharingOutput = self.sharingBand(self.users, self.sharing_state, self.transmitters)

        # Combining the primary bands for the different transmitters:
        primaryOutput = {"bler": tf.stack([primaryOutput1["bler"][0,:], primaryOutput2["bler"][1,:]]), "sinr": tf.stack([primaryOutput1["sinr"][0,:], primaryOutput2["sinr"][1,:]])}
        self.performance.append({"Primary": primaryOutput, "Sharing": sharingOutput})

        # Calculating rewards
        rates = tf.stack([primaryOutput1["rate"], primaryOutput2["rate"], sharingOutput["rate"]])
        throughput, per_ue_throughput, per_ap_per_band_throughput = get_throughput(rates)

        primary_power = tf.convert_to_tensor(np.power(10, (np.array([tx["primary_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32)
        sharing_power = tf.convert_to_tensor(np.power(10, (np.array([tx["sharing_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32)
        mu_pa = tf.convert_to_tensor([tx["mu_pa"] for tx in self.transmitters.values()])
        pe, per_ap_pe = get_power_efficiency(self.primary_bandwidth, # integral over power efficiency over time is energy efficiency
                                                   self.sharing_bandwidth,
                                                   self.sharing_state,
                                                   primary_power,
                                                   sharing_power,
                                                   mu_pa)

        se, per_ap_se = get_spectral_efficiency(self.primary_bandwidth, 
                                                self.sharing_bandwidth,
                                                per_ap_per_band_throughput)
        
        su = get_spectrum_utility(self.primary_bandwidth,
                                  self.sharing_bandwidth,
                                  self.sharing_state,
                                  throughput)
        
        # Plotting objectives/ rewards
        indices = tf.constant([[self.e, 0], [self.e, 1], [self.e, 2], [self.e, 3]])
        updates = tf.stack([throughput, se, pe, su], axis=0)
        self.rewards = tf.tensor_scatter_nd_update(self.rewards, indices, tf.reshape(updates, (4,)))

        # updates should influence the state

        # integral of the rewards needs computing, relative to the time length of the episode

        self.e += 1



        # To model as an MDP, we need to return a state which captures all necessary information for , not dependent on history
            # Unsure if it satisfies the Markov property, as there is a stochastic element
            # as long as the future state stochastic element is independent of the past state, it should be Markov
            # everything in state and action should be sufficient to calculate the reward

            # it is MDP - POMDP would be if the base stations make their own decisions - distributed RL

        # define the action space using spaces 



        # returns the 5-tuple (observation, reward, terminated, truncated, info)
            # need to add logic to detect if final episode and return done or truncated if ended early e.g. due to a limit
        return self.sharing_state, updates, False, False, {"rewards": self.rewards}

    def render(self):
        """ Visualising the performance. """
        # Plotting the performance and motion
        if len(self.performance) > self.max_results_length: # managing stored results size
            self.performance = self.performance[-1*self.max_results_length:]

        # Plotting the performance
        if self.e >= 1:
            plot_performance(episode=self.e,
                             users=self.users,
                             performance=self.performance, 
                             save_path=self.cfg.images_path)
            
        primary_sinr_map_1 = self.primaryBand1.sinr
        primary_sinr_map_2 = self.primaryBand2.sinr
        sharing_sinr_map = self.sharingBand.sinr
        fig_0, ax_0 = plot_motion(episode=self.e, 
                                  id="Primary Band 1, SINR", 
                                  grid=self.valid_area, 
                                  cm=tf.reduce_max(primary_sinr_map_1, axis=0), 
                                  color="inferno",
                                  users=self.users, 
                                  transmitters=self.transmitters, 
                                  cell_size=self.cfg.cell_size, 
                                  fig=self.fig_0,
                                  ax=self.ax_0, 
                                  save_path=self.cfg.images_path)
        fig_1, ax_1 = plot_motion(episode=self.e, 
                                  id="Primary Band 2, SINR", 
                                  grid=self.valid_area, 
                                  cm=tf.reduce_max(primary_sinr_map_2, axis=0), 
                                  color="inferno",
                                  users=self.users, 
                                  transmitters=self.transmitters, 
                                  cell_size=self.cfg.cell_size, 
                                  fig=self.fig_1,
                                  ax=self.ax_1, 
                                  save_path=self.cfg.images_path)
        fig_2, ax_2  = plot_motion(episode=self.e, 
                                  id="Sharing Band, Max SINR", 
                                  grid=self.valid_area, 
                                  cm=tf.reduce_max(sharing_sinr_map, axis=0), 
                                  color="viridis",
                                  users=self.users, 
                                  transmitters=self.transmitters, 
                                  cell_size=self.cfg.cell_size, 
                                  fig=self.fig_2,
                                  ax=self.ax_2, 
                                  save_path=self.cfg.images_path)
        
        plot_rewards(episode=self.e,
                     rewards=self.rewards,
                     save_path=self.cfg.images_path)

        return
