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
        print("Num Transmitters: ", self.num_tx)
        self.sharing_state = tf.ones(shape=(self.num_tx), dtype=tf.bool)
        self.max_results_length = self.cfg.max_results_length
        self.primary_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.sharing_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.primaryBands = {}
        self.initial_states = {}
        self.norm_ranges= {"throughput": (0, 50), "se": (0, 10), "pe": (0, 20), "su": (0, 10)}

        for id, tx in enumerate(self.transmitters.values()):
            self.initial_states["PrimaryBand"+str(id)] = tf.cast(tf.one_hot(id, self.num_tx, dtype=tf.int16), dtype=tf.bool)
            self.primaryBands["PrimaryBand"+str(id)] = FullSimulator(prefix="primary",
                                                                     scene_name=sionna.rt.scene.simple_street_canyon,
                                                                     carrier_frequency=tx["primary_carrier_freq"],
                                                                     bandwidth=self.primary_bandwidth,
                                                                     pmax=50, # maximum power
                                                                     transmitters=self.transmitters,
                                                                     num_rx = self.cfg.num_rx,
                                                                     max_depth=self.cfg.max_depth,
                                                                     cell_size=self.cfg.cell_size,
                                                                     initial_state = self.initial_states["PrimaryBand"+str(id)],
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
        # Set up gym standard attributes
        # action space - all valid actions - defined based on the number of transmitters
        # observation space - all valid observations



        
    def reset(self, seed=None, options=None):
        """ Reset the environment to its initial state. """
        super().reset(seed=seed)
        self.e = 0
        self.users={}
        self.performance=[]
        self.fig_0 = None
        self.ax_0 = None
        self.primary_figs = [None for _ in range(self.num_tx)]
        self.primary_axes = [None for _ in range(self.num_tx)]
        self.rewards = tf.zeros(shape=(self.cfg.episodes, 4), dtype=tf.float32)
        self.norm_rewards = tf.zeros(shape=(self.cfg.episodes, 4), dtype=tf.float32)
        [primaryBand.reset() for primaryBand in self.primaryBands.values()]
        self.sharingBand.reset()
        grids = [primaryBand.grid for primaryBand in self.primaryBands.values()]
        self.valid_area = self.sharingBand.grid
        for i in range(len(grids)):
            self.valid_area = tf.math.logical_or(self.valid_area, grids[i]) # shape [y_max, x_max]
        self.users = update_users(self.valid_area, self.cfg.num_rx, self.users) # getting initial user positions.
        self.primary_sinr_maps = [primaryBand.sinr for primaryBand in self.primaryBands.values()]    
        self.sharing_sinr_map = self.sharingBand.sinr

        return self._get_state()

    def step(self, action):
        """ Step through the environment. """
        self.sharing_state = action

        # Update the transmission power (up to the max, in fix granuals)


        # Generating the initial user positions based on logical OR of validity matrices
        if self.e > 0:
            self.users = update_users(self.valid_area, self.cfg.num_rx, self.users)

        # Running the simulation - with separated primary bands
        primaryOutputs = [primaryBand(self.users, state, self.transmitters) for primaryBand, state in zip(self.primaryBands.values(), self.initial_states.values())]
        sharingOutput = self.sharingBand(self.users, self.sharing_state, self.transmitters)

        # Combining the primary bands for the different transmitters:
        primaryOutput = {"bler": tf.stack([primaryOutput["bler"][i,:] for primaryOutput, i in zip(primaryOutputs, range(len(self.initial_states.values())))]), 
                         "sinr": tf.stack([primaryOutput["sinr"][i,:] for primaryOutput, i in zip(primaryOutputs, range(len(self.initial_states.values())))])}
        self.performance.append({"Primary": primaryOutput, "Sharing": sharingOutput})

        # Calculating rewards
        rates = tf.stack([primaryOutput["rate"] for primaryOutput in primaryOutputs] + sharingOutput["rate"])
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
        updates = tf.stack([throughput, 
                            se, 
                            pe, 
                            su], axis=0)
        norm_updates = tf.stack([self._norm(throughput,self.norm_ranges["throughput"][0],self.norm_ranges["throughput"][1]), 
                                 self._norm(se,self.norm_ranges["se"][0],self.norm_ranges["se"][1]), 
                                 self._norm(pe,self.norm_ranges["pe"][0],self.norm_ranges["pe"][1]), 
                                 self._norm(su,self.norm_ranges["su"][0],self.norm_ranges["su"][1])], axis=0)
        self.rewards = tf.tensor_scatter_nd_update(self.rewards, indices, tf.reshape(updates, (4,)))
        self.norm_rewards = tf.tensor_scatter_nd_update(self.norm_rewards, indices, tf.reshape(norm_updates, (4,)))
        reward = tf.reduce_sum(self.norm_rewards)

        self.e += 1



        # To model as an MDP, we need to return a state which captures all necessary information for , not dependent on history
            # Unsure if it satisfies the Markov property, as there is a stochastic element
            # as long as the future state stochastic element is independent of the past state, it should be Markov
            # everything in state and action should be sufficient to calculate the reward

            # it is MDP - POMDP would be if the base stations make their own decisions - distributed RL

        # define the action space using spaces 



        # returns the 5-tuple (observation, reward, terminated, truncated, info)
            # need to add logic to detect if final episode and return done or truncated if ended early e.g. due to a limit
        return self._get_state(), reward, False, False, {"rewards": self.rewards}
        # return None, reward, False, False, {"rewards": self.rewards}

    def _get_state(self):
        """ Getting the data for the current state. """
         # Adding normalised values to the state array
        state = []
        for user in self.users.values():
            x = user["position"][1]  # x position
            y = user["position"][0]  # y position

            norm_x = self._norm(tf.cast(x, tf.float32), 0, self.valid_area.shape[1])
            norm_y = self._norm(tf.cast(y, tf.float32), 0, self.valid_area.shape[0])

            primary_sinrs = [self._norm(primary_sinr[[0]][y][x], -1e5, 1e5) for primary_sinr in self.primary_sinr_maps]
            sharing_sinr = [self._norm(self.sharing_sinr_map[0][y][x], -1e5, 1e5)]


            state.extend([norm_x, norm_y] + primary_sinrs + sharing_sinr)

        return tf.convert_to_tensor(state, dtype=tf.float32)
    
    def _norm(self, value, min_val, max_val):
        """Min Max Normalisation of value to range [0,1] given a range. """

        return (value - min_val) / (max_val - min_val)
    
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
            
        self.fig_0, self.ax_0  = plot_motion(episode=self.e, 
                                  id="Sharing Band, Max SINR", 
                                  grid=self.valid_area, 
                                  cm=tf.reduce_max(self.sharing_sinr_map, axis=0), 
                                  color="viridis",
                                  users=self.users, 
                                  transmitters=self.transmitters, 
                                  cell_size=self.cfg.cell_size, 
                                  fig=self.fig_0,
                                  ax=self.ax_0, 
                                  save_path=self.cfg.images_path)
        
        for id, primary_sinr_map in enumerate(self.primary_sinr_maps):
            self.primary_figs[id], self.primary_axes[id] = plot_motion(episode=self.e, 
                                                                       id=f"Primary Band {id}, SINR", 
                                                                       grid=self.valid_area, 
                                                                       cm=tf.reduce_max(primary_sinr_map, axis=0), 
                                                                       color="inferno",
                                                                       users=self.users, 
                                                                       transmitters=self.transmitters, 
                                                                       cell_size=self.cfg.cell_size, 
                                                                       fig=self.primary_figs[id],
                                                                       ax=self.primary_axes[id], 
                                                                       save_path=self.cfg.images_path)
        
        plot_rewards(episode=self.e,
                     rewards=self.rewards,
                     save_path=self.cfg.images_path)

        return
