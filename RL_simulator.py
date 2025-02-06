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
import itertools

from plotting import plot_motion, plot_performance, plot_rewards
from utils import update_users, get_throughput, get_spectral_efficiency, get_power_efficiency, get_spectrum_utility
from scenario_simulator import FullSimulator
from logger import logger

class SionnaEnv(gym.Env):
    """ Sionna environment for reinforcement learning. """

    def __init__(self, cfg):
        """ Initialisation of the environment. """
        self.cfg = cfg
        self.limit = cfg.step_limit
        self.transmitters = dict(self.cfg.transmitters)
        self.num_tx = len(self.transmitters)
        self.max_results_length = self.cfg.max_results_length
        self.primary_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.sharing_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.primaryBands = {}
        self.initial_states = {}
        self.norm_ranges= {"throughput": (0, 30), "se": (0, 6), "pe": (0, 6), "su": (0, 6)}

        # Set up gym standard attributes
        self.truncated = False
        self.terminated = False # not used
        on_off_action = spaces.Discrete(2, seed=self.cfg.random_seed) # 0 = OFF, 1 = ON
        power_action = spaces.Discrete(3, seed=self.cfg.random_seed)  # 0 = decrease, 1 = stay, 2 = increase
        self.action_space = gym.vector.utils.batch_space(spaces.Tuple((on_off_action, power_action)), self.num_tx)
        single_tx_actions = list(itertools.product(
            range(on_off_action.n),   # [0, 1] for ON/OFF
            range(power_action.n)     # [0, 1, 2] for power actions
        ))
        self.possible_actions = list(itertools.product(single_tx_actions, single_tx_actions))
        single_ue_observation = spaces.Dict({
            "ue_pos": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32, seed=self.cfg.random_seed),
            "ue_sinr": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32, seed=self.cfg.random_seed),
            "ue_bler":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32, seed=self.cfg.random_seed),
        })
        self.num_actions = len(self.possible_actions)
        tx_observation = spaces.Dict({
            "tx_pos": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32, seed=self.cfg.random_seed),
            "tx_power": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32, seed=self.cfg.random_seed),
            "tx_state": spaces.Discrete(4, seed=self.cfg.random_seed),
            "ues_primary": spaces.Tuple((single_ue_observation for _ in range(cfg.num_rx)), seed=cfg.random_seed),
            "ues_sharing": spaces.Tuple((single_ue_observation for _ in range(cfg.num_rx)), seed=cfg.random_seed),
        })
        self.observation_space = spaces.Tuple((tx_observation for _ in range(self.num_tx)), seed=self.cfg.random_seed)

        # Initialising the transmitters, ensuring atleast one transmitter is active
        self.initial_action = self.action_space.sample()
        self.sharing_state = tf.convert_to_tensor([bool(tx_action[0]) for tx_action in self.initial_action], dtype=tf.bool)

        for id, tx in enumerate(self.transmitters.values()):
            self.initial_states["PrimaryBand"+str(id)] = tf.cast(tf.one_hot(id, self.num_tx, dtype=tf.int16), dtype=tf.bool)
            self.primaryBands["PrimaryBand"+str(id)] = FullSimulator(prefix="primary",
                                                                     scene_name= cfg.scene_path + "simple_OSM_scene.xml", #sionna.rt.scene.simple_street_canyon,
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
                                    scene_name=cfg.scene_path + "simple_OSM_scene.xml",
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
        super().reset(seed=seed)

        # Initialising data structures
        self.timestep = 0
        self.users={}
        self.performance=[]
        self.rates = None
        self.fig_0 = None
        self.ax_0 = None
        self.primary_figs = [None for _ in range(self.num_tx)]
        self.primary_axes = [None for _ in range(self.num_tx)]
        self.rewards = tf.zeros(shape=(self.cfg.step_limit + 1, 4), dtype=tf.float32)
        self.norm_rewards = tf.zeros(shape=(self.cfg.step_limit + 1, 4), dtype=tf.float32)

        # Resetting key attributes
        initial_action = self.action_space.sample()
        self.sharing_state = tf.convert_to_tensor([bool(tx_action[0]) for tx_action in self.initial_action], dtype=tf.bool)
        [primaryBand.reset() for primaryBand in self.primaryBands.values()]
        self.sharingBand.reset()
        grids = [primaryBand.grid for primaryBand in self.primaryBands.values()]
        self.valid_area = self.sharingBand.grid
        for i in range(len(grids)):
            self.valid_area = tf.math.logical_or(self.valid_area, grids[i]) # shape [y_max, x_max]
        self.users = update_users(self.valid_area, self.cfg.num_rx, self.users) # getting initial user positions.
        # Updating SINR maps
        self.primary_sinr_maps = [primaryBand.sinr for primaryBand in self.primaryBands.values()]    
        self.sharing_sinr_map = self.sharingBand.sinr

        return self._get_obs()

    def step(self, action):
        """ Step through the environment. """
        self.sharing_state = tf.convert_to_tensor([bool(tx_action[0]) for tx_action in action], dtype=tf.bool) # action in (array(tx_0_on/off, tx_0_power_decrease/stay/increase) for tx in transmitters)

        # Updating the transmitters
        for id, tx in enumerate(self.transmitters.values()):
            match action[id][1]:
                # Applying actions - restriction of actions is handled through masking in the Q-network
                case 0:
                    tx["sharing_power"] = tx["sharing_power"] - 1
                case 1:
                    pass
                case 2:
                    tx["sharing_power"] = tx["sharing_power"] + 1
                case _:
                    logger.critical("Invalid action.")
                    raise ValueError("Invalid action.")
            
            if (tx["sharing_power"] > tx["max_power"]) or (tx["sharing_power"] < tx["min_power"]):
                logger.critical("Out of power range, this should not be possible if masking is properly applied.")
                raise ValueError("Out of power range.")
            tx["state"] = action[id][0] # updating the stored state value

        # Generating the initial user positions based on logical OR of validity matrices
        if self.timestep > 0:
            self.users = update_users(self.valid_area, self.cfg.num_rx, self.users)

        # Running the simulation - with separated primary bands
        primaryOutputs = [primaryBand(self.users, state, self.transmitters) for primaryBand, state in zip(self.primaryBands.values(), self.initial_states.values())]
        sharingOutput = self.sharingBand(self.users, self.sharing_state, self.transmitters, self.timestep, self.cfg.images_path)

        # Printing simulation details:
        # if self.timestep == 1:
        #     [band.simulator.pusch_config.show() for band in self.primaryBands.values()]
        #     self.sharingBand.simulator.pusch_config.show()

        # Updating SINR maps
        self.primary_sinr_maps = [primaryBand.sinr for primaryBand in self.primaryBands.values()]    
        self.sharing_sinr_map = self.sharingBand.sinr

        # Combining the primary bands for the different transmitters:
        primaryOutput = {"bler": tf.stack([primaryOutput["bler"][i,:] for primaryOutput, i in zip(primaryOutputs, range(len(self.initial_states.values())))]), 
                         "sinr": tf.stack([primaryOutput["sinr"][i,:] for primaryOutput, i in zip(primaryOutputs, range(len(self.initial_states.values())))])}
        self.performance.append({"Primary": primaryOutput, "Sharing": sharingOutput})

        # Calculating rewards
        self.rates = tf.stack([primaryOutput["rate"] for primaryOutput in primaryOutputs] + sharingOutput["rate"])
        throughput, per_ue_throughput, per_ap_per_band_throughput = get_throughput(self.rates)

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
        indices = tf.constant([[self.timestep, 0], [self.timestep, 1], [self.timestep, 2], [self.timestep, 3]])
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
        reward = tf.reduce_sum(norm_updates)

        if (np.isnan(reward.numpy())):
            logger.critical("Reward NAN")
            return None, None, None, None, None

        # Infinite-horizon problem so we terminate at an arbitraty point - the agent does not know about this limit
        if self.timestep == self.limit:
            logger.warning("Last step of episode, Truncated.")
            self.truncated = True

        # returns the 5-tuple (observation, reward, terminated, truncated, info)
        return self._get_obs(), reward, self.terminated, self.truncated, {"rewards": norm_updates}

    def _get_obs(self):
        """ Getting the data for the current state. """
         # Adding normalised values to the state array
        state = []

        # Iterate over each transmitter
        for tx_id, tx in enumerate(self.transmitters.values()):
            # Normalize transmitter position
            norm_tx_x = self._norm(tx["position"][1], 0, self.valid_area.shape[1])
            norm_tx_y = self._norm(tx["position"][0], 0, self.valid_area.shape[0])

            # Normalize power level (assuming min/max power are defined)
            norm_tx_power = self._norm(tx["sharing_power"], self.cfg.min_power, self.cfg.max_power)

            tx_on_off = int(tx["state"]) 

            # Primary and sharing UEs
            primary_ues = []
            sharing_ues = []

            for user_id, user in enumerate(self.users.values()):
                x = user["position"][1].numpy()
                y = user["position"][0].numpy()

                norm_x = self._norm(x, 0, self.valid_area.shape[1])
                norm_y = self._norm(y, 0, self.valid_area.shape[0])

                # Convert SINR to dB and normalize
                primary_sinr = [self._norm((10 * tf.math.log(self.primary_sinr_maps[tx_id][0][y][x]) / tf.math.log(10.0)).numpy(), -100, 100)]
                if tx_on_off == 0:
                    sharing_sinr = [self._norm((10 * tf.math.log(0.0) / tf.math.log(10.0)).numpy(), -100, 100)]
                else:
                    index = min(self.sharing_sinr_map.shape[0] - 1, tx_id)
                    sharing_sinr = [self._norm((10 * tf.math.log(self.sharing_sinr_map[index][y][x]) / tf.math.log(10.0)).numpy(), -100, 100)]

                # Normalize BLER (assuming 0-1 range)
                if self.rates is None:
                    norm_bler = self._norm(1, 0, 1)
                else:
                    norm_bler = self._norm(tf.reduce_sum(self.rates[tx_id,:,user_id]).numpy(), 0, 1)

                # Construct UE observation dictionary
                prim = {
                    "ue_pos": np.array([norm_x, norm_y]),
                    "ue_sinr": np.array(primary_sinr),
                    "ue_bler": np.array(norm_bler),
                }
                shar = {
                    "ue_pos":  np.array([norm_x, norm_y]),
                    "ue_sinr":  np.array(sharing_sinr),
                    "ue_bler": np.array(norm_bler),
                }

                primary_ues.append(prim)
                sharing_ues.append(shar)

            # Construct transmitter observation dictionary
            tx_obs = {
                "tx_pos": np.array([norm_tx_x, norm_tx_y]),
                "tx_power": np.array([norm_tx_power]),
                "tx_state": tx_on_off,
                "ues_primary": primary_ues,
                "ues_sharing": sharing_ues,
            }

            state.append(tx_obs)

# Need to fix either the state or the get_obs for how the bler, sinr etc are added

        return state
    
    def _norm(self, value, min_val, max_val):
        """Min Max Normalisation of value to range [0,1] given a range. """
        value_clipped = tf.clip_by_value(value, min_val, max_val) # avoiding inf

        return (value_clipped - min_val) / (max_val - min_val)
    
    def render(self, episode):
        """ Visualising the performance. """
        # Plotting the performance and motion
        if len(self.performance) > self.max_results_length: # managing stored results size
            self.performance = self.performance[-1*self.max_results_length:]

        # Plotting the performance
        if self.timestep >= 1:
            plot_performance(step=self.timestep,
                             users=self.users,
                             performance=self.performance, 
                             save_path=self.cfg.images_path)
            plot_rewards(episode=episode,
                         step=self.timestep,
                         rewards=self.rewards,
                         save_path=self.cfg.images_path)
            
        self.fig_0, self.ax_0  = plot_motion(step=self.timestep, 
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
            self.primary_figs[id], self.primary_axes[id] = plot_motion(step=self.timestep, 
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
        

        return
