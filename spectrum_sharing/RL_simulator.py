""" RL_simulator.py

Wrapping Sionna logic inside a gymnasium wrapper for reinforcement learning.

"""

import tensorflow as tf
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import product
import math
import matplotlib.pyplot as plt

from spectrum_sharing.plotting import plot_motion, plot_performance, plot_rewards, prop_fair_plotter
from spectrum_sharing.utils import update_users, get_throughput, get_spectral_efficiency, get_power_efficiency, get_spectrum_utility, get_power_efficiency_bounds, get_fairness
from spectrum_sharing.scenario_simulator import FullSimulator
from spectrum_sharing.logger import logger

class PrecomputedEnv(gym.Env):
    """ Environment inheriting from OpenAI Gymnasium for training
    reinforcement learning models in spectrum sharing. Uses precomputed coverage
    maps for expedited training without having to load Sionna.

    Parameters
    ----------
    cfg : dict
        Top level configuration dictionary.

    test : bool
        Flag indicating if in test mode. Changes what is plotted.

    Usage
    ------
    Call reset() to initialise episode.
    Call step() to advance episode.
    Call render() to visualise.
    """
    def __init__(self, cfg, test=False):
        """ Initialisation of the environment. """
        self.cfg = cfg
        self.limit = cfg.step_limit
        self.transmitters = dict(self.cfg.transmitters)
        self.num_tx = len(self.transmitters)
        self.max_results_length = self.cfg.max_results_length
        self.primary_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.sharing_bandwidth = self.cfg.sharing_fft_size * self.cfg.primary_subcarrier_spacing
        self.primaryBands = {}
        self.initial_states = {}
        self.test = test
        if self.test:
            self.images_path = self.cfg.test_images_path
        else:
            self.images_path = self.cfg.images_path

        # Set up gym standard attributes
        self.truncated = False
        self.terminated = False # not used
        on_off_action = spaces.Discrete(2, seed=self.cfg.random_seed) # 0 = OFF, 1 = ON
        power_action = spaces.Discrete(3, seed=self.cfg.random_seed)  # 0 = decrease, 1 = stay, 2 = increase
        self.action_space = gym.vector.utils.batch_space(spaces.Tuple((on_off_action, power_action)), self.num_tx)
        single_tx_actions = list(product(
            range(on_off_action.n),   # [0, 1] for ON/OFF
            range(power_action.n)     # [0, 1, 2] for power actions
        ))
        self.possible_actions = list(product(single_tx_actions, single_tx_actions))

        single_ue_observation = spaces.Dict({
            "ue_pos": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32, seed=self.cfg.random_seed),
            "ue_sinr": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32, seed=self.cfg.random_seed),
            "ue_rate":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32, seed=self.cfg.random_seed),
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

        # Creating an array of all of the different power options.
        powers = [[0] for _ in range(self.num_tx)]
        for id, tx in enumerate(self.transmitters.values()):
            powers[id] = powers[id] + list(range(int(tx["min_power"]), int(tx["max_power"]) + 1))
        self.powers = list(product(*[powers[id] for id in range(self.num_tx)])) # all possible combinations of powers

        initial_action = self.possible_actions[np.random.randint(0,self.num_actions)] 
        self.sharing_state = tf.convert_to_tensor([bool(tx_action[0]) for tx_action in initial_action], dtype=tf.bool)
        for id, a in enumerate(initial_action):
            self.transmitters[f"tx{id}"]["state"] = int(a[0]) # apply the state
        
        # Initialising Sionna just to get meta data
        for id, tx in enumerate(self.transmitters.values()):
            self.initial_states["PrimaryBand"+str(id)] = tf.cast(tf.one_hot(id, self.num_tx, dtype=tf.int16), dtype=tf.bool)
            self.primaryBands["PrimaryBand"+str(id)] = FullSimulator(cfg=self.cfg,
                                                                     prefix="primary",
                                                                     scene_name= cfg.scene_path + "simple_OSM_scene.xml", #sionna.rt.scene.simple_street_canyon,
                                                                     carrier_frequency=tx["primary_carrier_freq"],
                                                                     pmax=self.cfg.max_power, # global maximum power
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
        self.sharingBand = FullSimulator(cfg=self.cfg,
                                         prefix="sharing",
                                         scene_name= cfg.scene_path + "simple_OSM_scene.xml",
                                         carrier_frequency=self.cfg.sharing_carrier_freq,
                                         pmax=self.cfg.max_power, # maximum power for initial mapping of coverage area
                                         transmitters=self.transmitters,
                                         num_rx = self.cfg.num_rx,
                                         max_depth=self.cfg.max_depth,
                                         cell_size=self.cfg.cell_size,
                                         initial_state = self.sharing_state,
                                         subcarrier_spacing = self.cfg.sharing_subcarrier_spacing,
                                         fft_size = self.cfg.sharing_fft_size,
                                         batch_size=self.cfg.batch_size,
                                         )
        
        # Getting scheduler function for later
        self.scheduler = self.sharingBand.proportional_fair_scheduler
        
        # Getting min and max and key attribute from theoretical calculations
        global_centre = self.sharingBand.center_transform
        self.global_max = (global_centre[0:2] + (self.sharingBand.global_size[0:2]  / 2)).astype(int)
        self.global_min = (global_centre[0:2] - (self.sharingBand.global_size[0:2]  / 2)).astype(int)
        primary_max_rates = sum([band.max_data_rate for band in self.primaryBands.values()])
        max_throughput = ((self.num_tx * self.sharingBand.max_data_rate) + primary_max_rates)
        max_se = max(max_throughput / (self.primary_bandwidth + self.sharing_bandwidth), primary_max_rates / self.primary_bandwidth)
        max_su = max_se # max_su will be < max_se by definition

        # Prop fair scheduler params
        primary_number_rbs = np.array([band.number_rbs for band in self.primaryBands.values()])
        primary_num_slots = np.array([band.num_slots for band in self.primaryBands.values()])
        primary_max_per_rb = np.array([band.max_data_sent_per_rb for band in self.primaryBands.values()])
        if not np.all(primary_number_rbs == primary_number_rbs[0]):
            raise Exception("Inconsistent number_rbs in primaryBands")
        self.primary_number_rbs = primary_number_rbs[0]
        if not np.all(primary_num_slots == primary_num_slots[0]):
            raise Exception("Inconsistent num_slots in primaryBands")
        self.primary_num_slots = primary_num_slots[0]
        if not np.all(primary_max_per_rb == primary_max_per_rb[0]):
            raise Exception("Inconsistent max_data_sent_per_rb in primaryBands")
        self.primary_max_per_rb = primary_max_per_rb[0]
        self.sharing_number_rbs = self.sharingBand.number_rbs
        self.sharing_num_slots = self.sharingBand.num_slots
        self.sharing_max_per_rb = self.sharingBand.max_data_sent_per_rb


        primary_power = tf.convert_to_tensor(np.power(10, (np.array([tx["primary_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32) # in W
        sharing_power_min = tf.convert_to_tensor(np.power(10, (np.array([tx["min_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32) # in W
        sharing_power_max = tf.convert_to_tensor(np.power(10, (np.array([tx["max_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32)
        mu_pa = tf.convert_to_tensor([tx["mu_pa"] for tx in self.transmitters.values()])
        min_pe, max_pe, _, _ = get_power_efficiency_bounds(self.primary_bandwidth, self.sharing_bandwidth, primary_power, mu_pa,
                                                           sharing_power_min, sharing_power_max)
        
        self.norm_ranges= {"throughput": (0, math.ceil(max_throughput / 1e6)), # automate this generation based on theoretical calculations
                           "se": (0, math.ceil(max_se)), 
                           "pe": (min_pe, math.ceil(max_pe)),
                           "su": (0, math.ceil(max_su)),
                           "sinr": (math.floor(self.cfg.min_sinr), math.ceil(self.cfg.max_sinr))} 
        
        # Loading the precomputed maps
        self.primary_maps = np.load(self.cfg.assets_path + "primary_maps.npy")
        self.sharing_maps = np.load(self.cfg.assets_path + "sharing_maps.npy")
        self.valid_area = np.load(self.cfg.assets_path + "grid.npy")

         # Normalising the coverage maps
        self.norm_primary_sinr_maps = (np.clip(self.primary_maps[0], a_min=self.cfg.min_sinr, a_max=self.cfg.max_sinr) - self.cfg.min_sinr) / (self.cfg.max_sinr - self.cfg.min_sinr) # min max between +- 100
        self.norm_primary_bler_maps = self.primary_maps[1] # bler already normalised
        self.norm_sharing_sinr_maps = (np.clip(self.sharing_maps[0], a_min=self.cfg.min_sinr, a_max=self.cfg.max_sinr) - self.cfg.min_sinr) / (self.cfg.max_sinr - self.cfg.min_sinr)
        self.norm_sharing_bler_maps = self.sharing_maps[1]
        self.primary_sinr_maps = np.clip(self.primary_maps[0], a_min=self.cfg.min_sinr, a_max=self.cfg.max_sinr)
        self.primary_bler_maps = self.primary_maps[1] # bler already normalised
        self.sharing_sinr_maps = np.clip(self.sharing_maps[0], a_min=self.cfg.min_sinr, a_max=self.cfg.max_sinr)
        self.sharing_bler_maps = self.sharing_maps[1]

        # Clearing up unnecessary Sionna components after they have been used for precalculation
        del self.primaryBands
        del self.sharingBand

        
    def reset(self, seed=None):
        """ 
        Reset the environment to its initial state.
        
        Parameters
        ----------
        seed : int
            A random seed for determinism. Defaults to None.

        Returns
        -------
        observation : list
            Environment observation. The list of dict returned by _get_obs() 
        
        """
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
        self.rewards = tf.zeros(shape=(self.cfg.step_limit + 1, 5), dtype=tf.float32)
        self.norm_rewards = tf.zeros(shape=(self.cfg.step_limit + 1, 5), dtype=tf.float32)

        # Resetting key attributes
        self.users = update_users(self.valid_area, self.cfg.num_rx, self.users) # getting initial user positions.

        initial_action = self.possible_actions[np.random.randint(0,self.num_actions)] 
        self.sharing_state = tf.convert_to_tensor([bool(tx_action[0]) for tx_action in initial_action], dtype=tf.bool)
        for id, a in enumerate(initial_action):
            self.transmitters[f"tx{id}"]["state"] = int(a[0]) # apply the state
        powers_index = tuple([int(tx["sharing_power"] * initial_action[tx_id][0]) for tx_id, tx in enumerate(self.transmitters.values())])
        self.state = self.powers.index(powers_index)

        return self._get_obs()

    def step(self, action):
        """ 
        Step through the environment, after applying an action.
        
        Parameters
        ----------
        action : list
            The action taken for the next timestep.

        Returns
        -------
        observation : list
            Environment observation. The list of dict returned by _get_obs() 

        reward : float
            Normalised float representation of the reward the agent receieves
            from this episode.

        terminated : bool
            Flag indicating if this is the final timestep of the episode.

        truncated : bool
            Flag indicated if the episode is being cut short.

        info : dict
            Key value pairs of additional information.           
        """
        self.sharing_state = tf.convert_to_tensor([bool(tx_action[0]) for tx_action in action], dtype=tf.bool) # action in (array(tx_0_on/off, tx_0_power_decrease/stay/increase) for tx in transmitters)

        # Getting new random user positions
        if self.timestep > 0:
            self.users = update_users(self.valid_area, self.cfg.num_rx, self.users)

        # Mapping the effect of the action to the state via search and alignment
        for id, a in enumerate(action):
            self.transmitters[f"tx{id}"]["state"] = int(a[0]) # apply the state
            if a[1] == 0: # reduce power, action mask should protect from exceeding valid actions
                self.transmitters[f"tx{id}"]["sharing_power"] -= 1
            elif a[1] == 2: # increase power
                self.transmitters[f"tx{id}"]["sharing_power"] += 1
        powers_index = tuple([int(tx["sharing_power"] * action[tx_id][0]) for tx_id, tx in enumerate(self.transmitters.values())])
        try:
            self.state = self.powers.index(powers_index)
        except:
            logger.critical("Power index not found.")
            return None, None, None, None, None # truncates episode and stops error propagation without exiting
        
        self.sharing_sinr_map = self.sharing_sinr_maps[self.state]
        self.sharing_bler_map = self.sharing_bler_maps[self.state]
        # Calculating the simulator outputs from pre-generated SINR/BLER maps
        user_positions = tf.convert_to_tensor([user["position"][0:2] for user in self.users.values()])
        for band_id, (sinr_map, bler_map) in enumerate(zip([self.sharing_sinr_map, self.primary_sinr_maps], [self.sharing_bler_map, self.primary_bler_maps])):
            blers = []
            sinrs = []
            rates = []
            for tx_id, tx in enumerate(self.transmitters.values()):
                sinr_image = sinr_map[tx_id]
                bler_image = bler_map[tx_id]

                height, width = sinr_image.shape

                plt.figure(figsize=(12, 5))

                # --- SINR Plot ---
                plt.subplot(1, 2, 1)
                plt.imshow(sinr_image, origin='lower', cmap='viridis', vmin=self.cfg.min_sinr, vmax=self.cfg.max_sinr)
                plt.colorbar(label='SINR (dB)')
                plt.title(f'SINR Map - Band {band_id}, TX {tx_id}')
                plt.xlim(0, width)
                plt.ylim(0, height)

                # Scatter user positions
                plt.scatter(user_positions[:, 1], user_positions[:, 0], c='white', s=20, edgecolors='black')
                for y, x in user_positions:
                    x = float(x)
                    y = float(y)
                    plt.text(x + 0.5, y + 0.5, f'({int(x)},{int(y)})', color='white', fontsize=6)

                # --- BLER Plot ---
                plt.subplot(1, 2, 2)
                plt.imshow(bler_image, origin='lower', cmap='cool', vmin=0, vmax=1)
                plt.colorbar(label='BLER')
                plt.title(f'BLER Map - Band {band_id}, TX {tx_id}')
                plt.xlim(0, width)
                plt.ylim(0, height)

                # Scatter user positions
                plt.scatter(user_positions[:, 1], user_positions[:, 0], c='white', s=20, edgecolors='black')
                for y, x in user_positions:
                    x = float(x)
                    y = float(y)
                    plt.text(x + 0.5, y + 0.5, f'({int(x)},{int(y)})', color='white', fontsize=6)

                # Save and close
                plt.tight_layout()
                plt.savefig(self.images_path + f"Plot for band {band_id}, tx{tx_id}.png", dpi=400)
                plt.close()

                sinr = tf.gather_nd(sinr_map[tx_id], user_positions)
                sinrs.append(sinr)
                bler = tf.gather_nd(bler_map[tx_id], user_positions)
                blers.append(bler)
                # Scheduling
                if band_id == 0: # sharing
                    grid_alloc, user_rates = self.scheduler(bler, self.sharing_max_per_rb, self.sharing_num_slots, self.sharing_number_rbs)
                    # prop_fair_plotter(self.timestep, 
                    #                     tx["name"],
                    #                     grid_alloc, 
                    #                     self.cfg.num_rx,
                    #                     user_rates, 
                    #                     self.sharing_max_per_rb,
                    #                     save_path=self.images_path)
                elif band_id == 1: # primary
                    grid_alloc, user_rates = self.scheduler(bler, self.primary_max_per_rb, self.primary_num_slots, self.primary_number_rbs)
                    # prop_fair_plotter(self.timestep, # adds significant time constraint
                    #                     tx["name"],
                    #                     grid_alloc, 
                    #                     self.cfg.num_rx,
                    #                     user_rates, 
                    #                     self.primary_max_per_rb,
                    #                     save_path=self.images_path)
                else:
                    raise Exception("Out of range")
                rates.append(user_rates)

            if band_id == 0: # sharing
                sharingOutput = {"bler": tf.stack(blers, axis=0), "sinr": tf.stack(sinrs, axis=0), "rate": tf.stack(rates, axis=0)}
            elif band_id == 1: # primary
                primaryOutput = {"bler": tf.stack(blers, axis=0), "sinr": tf.stack(sinrs, axis=0), "rate": tf.stack(rates, axis=0)}
            else:
                raise Exception("Out of range")
            
        self.performance.append({"Primary": primaryOutput, "Sharing": sharingOutput})
        # print({"Primary": primaryOutput, "Sharing": sharingOutput})

        # Calculating rewards
        self.rates = tf.concat([
                                tf.cast(tf.expand_dims(primaryOutput["rate"], axis=1), dtype=tf.float32),  
                                tf.cast(tf.expand_dims(sharingOutput["rate"], axis=1), dtype=tf.float32)  # Expanding to [2,1,num_ues]
                                ], axis=1)  # Concatenating along axis 1 to make it [Transmitters, Bands, UEs]
        throughput, per_ue_throughput, per_ap_per_band_throughput = get_throughput(self.rates)

        fairness = get_fairness(per_ue_throughput)

        primary_power = tf.convert_to_tensor(np.power(10, (np.array([tx["primary_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32) # in W
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

        # Checking validity of values before performing normalisation:
        if not (self.norm_ranges["throughput"][0] <= throughput <= self.norm_ranges["throughput"][1]):
            raise ValueError(f"Throughput value {throughput} is out of range: {self.norm_ranges['throughput']}")
        if not (self.norm_ranges["se"][0] <= se <= self.norm_ranges["se"][1]):
            raise ValueError(f"SE value {se} is out of range: {self.norm_ranges['se']}")
        if not (self.norm_ranges["pe"][0] <= pe <= self.norm_ranges["pe"][1]):
            raise ValueError(f"PE value {pe} is out of range: {self.norm_ranges['pe']}")
        if not (0 <= fairness <= 1):
            raise ValueError(f"Jain's fairness value {fairness} is out of range: [0, 1]")
        if not (self.norm_ranges["su"][0] <= su <= self.norm_ranges["su"][1]):
            raise ValueError(f"SU value {su} is out of range: {self.norm_ranges['su']}")

        # Processing the reward for the agent
        updates = tf.stack([throughput,
                            fairness, 
                            se, 
                            pe, 
                            su], axis=0)
        logger.info(f"Updates [throughput, fairness, se, pe, su]: {updates.numpy()}")

        norm_updates = tf.stack([self._norm(throughput, self.norm_ranges["throughput"][0], self.norm_ranges["throughput"][1]), 
                                 fairness, # designed to be bounded [0,1]
                                 self._norm(se, self.norm_ranges["se"][0], self.norm_ranges["se"][1]), 
                                 self._norm(1/pe, 1/self.norm_ranges["pe"][1], 1/self.norm_ranges["pe"][0]), # being minimised - careful in defining ranges to avoid division by zero
                                 self._norm(su, self.norm_ranges["su"][0], self.norm_ranges["su"][1])], axis=0)
        
        indices = tf.constant([[self.timestep, 0], [self.timestep, 1], [self.timestep, 2], [self.timestep, 3], [self.timestep, 4]]) # used for updating preallocated tensor
        self.rewards = tf.tensor_scatter_nd_update(self.rewards, indices, tf.reshape(updates, (5,)))

        self.norm_rewards = tf.tensor_scatter_nd_update(self.norm_rewards, indices, tf.reshape(norm_updates, (5,)))
        reward = tf.reduce_sum(norm_updates)

        if (np.isnan(reward.numpy())):
            logger.critical("Reward NAN")
            return None, None, None, None, None

        # Infinite-horizon problem so we terminate at an arbitraty point - the agent does not know about this limit
        if self.timestep == self.limit:
            logger.warning("Last step of episode, Truncated.")
            self.truncated = True

        # returns the 5-tuple (observation, reward, terminated, truncated, info)
        return self._get_obs(), reward, self.terminated, self.truncated, {"rewards": updates}

    def _get_obs(self):
        """ 
        Getting the data for the current state. The design of the observation depends on the agent input structure.

        Returns
        -------
        observation : list
            Environment observation. The list of dict returned by _get_obs()          
        
        """
        # start = perf_counter()
        state = [] # Adding normalised values to the state array

        # Iterate over each transmitter
        for tx_id, tx in enumerate(self.transmitters.values()):
            # Normalize transmitter position

            norm_tx_x = self._norm(tx["position"][0], self.global_min[0], self.global_max[0]) # Global system is in x,y,z, coverage map is in y,x 
            norm_tx_y = self._norm(tx["position"][1], self.global_min[1], self.global_max[1])

            # Normalize power level (assuming min/max power are defined)
            norm_tx_power = self._norm(tx["sharing_power"], self.cfg.min_power, self.cfg.max_power)

            tx_on_off = int(tx["state"]) 

            # Primary and sharing UEs
            primary_ues = []
            sharing_ues = []

            for user_id, user in enumerate(self.users.values()):
                x = user["position"][1] # uses the coverage map coordinate system
                y = user["position"][0]

                norm_x = self._norm(x, 0, self.valid_area.shape[1])
                norm_y = self._norm(y, 0, self.valid_area.shape[0])

                # Convert SINR to dB and normalize
                primary_sinr = [self.norm_primary_sinr_maps[tx_id][y][x]]
                if tx_on_off == 0:
                    sharing_sinr = [self._norm((10 * tf.math.log(0.0) / tf.math.log(10.0)).numpy(), *self.norm_ranges["sinr"]).numpy()]
                else:
                    sharing_sinr = [self.norm_sharing_sinr_maps[self.state][tx_id][y][x]]

                # Using rate instead of BLER
                if self.rates is None:
                    rate = np.array([0 for tx in range(self.num_tx + 1)], dtype=np.float32) # primaries and sharing will be in self.rates
                else:
                    rate = self.rates[tx_id,:,user_id].numpy() # note: this is probably being mapped to too small a range below in normalisation
                # Construct UE observation dictionary
                prim = {
                    "ue_pos": np.array([norm_x, norm_y]),
                    "ue_sinr": np.array(primary_sinr),
                    "ue_rate": np.array(self._norm(np.sum(rate[:-1]), self.norm_ranges["throughput"][0] * 1e6, self.norm_ranges["throughput"][1] * 1e6)), # normalised with max aggregate not max UE
                }
                shar = {
                    "ue_pos":  np.array([norm_x, norm_y]),
                    "ue_sinr":  np.array(sharing_sinr),
                    "ue_rate": np.array(self._norm(rate[-1], self.norm_ranges["throughput"][0] * 1e6, self.norm_ranges["throughput"][1] * 1e6)),
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
        # end = perf_counter()
        # print("State calculation time: ", round(end-start, 5), "s") # approx 0.2s
        return state
    
    def _norm(self, value, min_val, max_val):
        """
        Min Max Normalisation of value to range [0,1] given a range. 
        
        Parameters
        ----------
        value : float
            Value for normalisation.

        min_val : float
            Upper bound for value.

        max_val : float
            Lower bound for value.

        Returns
        -------
        value_norm : float
            Min-max normalised value between [0,1].      
        
        """
        value_clipped = tf.clip_by_value(value, min_val, max_val) # avoiding inf

        return (value_clipped - min_val) / (max_val - min_val)
    
    def render(self, episode):
        """ 
        Visualising the performance. Plots and saves graphs to directory at config save path for images. 

        Parameters
        ----------
        episode : int
            Episode reference number for file naming.
        """
        # Plotting the performance and motion
        if len(self.performance) > self.max_results_length: # managing stored results size
            self.performance = self.performance[-1*self.max_results_length:]

        # Plotting the performance
        if self.timestep >= 1:
            if self.test:
                # plot_performance(step=self.timestep,
                #                 users=self.users,
                #                 performance=self.performance, 
                #                 save_path=self.images_path)
                pass
            plot_rewards(episode=episode,
                         step=self.timestep,
                         rewards=self.rewards,
                         save_path=self.images_path)
        
        if self.test:
            self.fig_0, self.ax_0  = plot_motion(step=self.timestep, 
                                                id="Sharing Band, Max SINR", 
                                                grid=self.valid_area, 
                                                cm=tf.reduce_max(self.sharing_sinr_map, axis=0), 
                                                color="viridis",
                                                users=self.users, 
                                                transmitters=self.transmitters, 
                                                cell_size=self.cfg.cell_size, 
                                                sinr_range=self.norm_ranges["sinr"],
                                                fig=self.fig_0,
                                                ax=self.ax_0, 
                                                save_path=self.images_path)
            
            for id, primary_sinr_map in enumerate(self.primary_sinr_maps):
                self.primary_figs[id], self.primary_axes[id] = plot_motion(step=self.timestep, 
                                                                        id=f"Primary Band {id}, SINR", 
                                                                        grid=self.valid_area, 
                                                                        cm=primary_sinr_map, 
                                                                        color="inferno",
                                                                        users=self.users, 
                                                                        transmitters=self.transmitters, 
                                                                        cell_size=self.cfg.cell_size, 
                                                                        sinr_range=self.norm_ranges["sinr"],
                                                                        fig=self.primary_figs[id],
                                                                        ax=self.primary_axes[id], 
                                                                        save_path=self.images_path)


        return
    

class SionnaEnv(gym.Env):
    """ Sionna environment inheriting from OpenAI Gymnasium for training
    reinforcement learning models in spectrum sharing.

    Parameters
    ----------
    cfg : dict
        Top level configuration dictionary.

    test : bool
        Flag indicating if in test mode. Changes what is plotted.

    Usage
    ------
    Call reset() to initialise episode.
    Call step() to advance episode.
    Call render() to visualise.

    """
    def __init__(self, cfg, test=False):
        """ Initialisation of the environment. """
        self.cfg = cfg
        self.limit = cfg.step_limit
        self.transmitters = dict(self.cfg.transmitters)
        self.num_tx = len(self.transmitters)
        self.max_results_length = self.cfg.max_results_length
        self.primary_bandwidth = self.cfg.primary_fft_size * self.cfg.primary_subcarrier_spacing
        self.sharing_bandwidth = self.cfg.sharing_fft_size * self.cfg.primary_subcarrier_spacing
        self.primaryBands = {}
        self.initial_states = {}
        self.test = test
        if self.test:
            self.images_path = self.cfg.test_images_path
        else:
            self.images_path = self.cfg.images_path

        # Set up gym standard attributes
        self.truncated = False
        self.terminated = False # not used
        on_off_action = spaces.Discrete(2, seed=self.cfg.random_seed) # 0 = OFF, 1 = ON
        power_action = spaces.Discrete(3, seed=self.cfg.random_seed)  # 0 = decrease, 1 = stay, 2 = increase
        self.action_space = gym.vector.utils.batch_space(spaces.Tuple((on_off_action, power_action)), self.num_tx)
        single_tx_actions = list(product(
            range(on_off_action.n),   # [0, 1] for ON/OFF
            range(power_action.n)     # [0, 1, 2] for power actions
        ))
        self.possible_actions = list(product(single_tx_actions, single_tx_actions))

        single_ue_observation = spaces.Dict({
            "ue_pos": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32, seed=self.cfg.random_seed),
            "ue_sinr": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32, seed=self.cfg.random_seed),
            "ue_rate":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32, seed=self.cfg.random_seed),
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

        # Initialising the transmitters
        self.initial_action = self.action_space.sample() # note that sample gives [[tx_0_state, tx_1_state], [tx_0_power, tx_1_power]]
        self.sharing_state = tf.convert_to_tensor([bool(tx_action) for tx_action in self.initial_action[0]], dtype=tf.bool)

        for id, tx in enumerate(self.transmitters.values()):
            self.initial_states["PrimaryBand"+str(id)] = tf.cast(tf.one_hot(id, self.num_tx, dtype=tf.int16), dtype=tf.bool)
            self.primaryBands["PrimaryBand"+str(id)] = FullSimulator(cfg=self.cfg,
                                                                     prefix="primary",
                                                                    #  scene_name= cfg.scene_path + "empty_scene.xml", #cfg.scene_path + "simple_OSM_scene.xml", #sionna.rt.scene.simple_street_canyon,
                                                                     scene_name=cfg.scene_path + "simple_OSM_scene.xml",
                                                                     carrier_frequency=tx["primary_carrier_freq"],
                                                                     pmax=self.cfg.max_power, # global maximum power
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
        self.sharingBand = FullSimulator(cfg=self.cfg,
                                         prefix="sharing",
                                        #  scene_name=cfg.scene_path + "empty_scene.xml",#"simple_OSM_scene.xml",
                                         scene_name=cfg.scene_path + "simple_OSM_scene.xml",
                                         carrier_frequency=self.cfg.sharing_carrier_freq,
                                         pmax=self.cfg.max_power, # maximum power for initial mapping of coverage area
                                         transmitters=self.transmitters,
                                         num_rx = self.cfg.num_rx,
                                         max_depth=self.cfg.max_depth,
                                         cell_size=self.cfg.cell_size,
                                         initial_state = self.sharing_state,
                                         subcarrier_spacing = self.cfg.sharing_subcarrier_spacing,
                                         fft_size = self.cfg.sharing_fft_size,
                                         batch_size=self.cfg.batch_size,
                                         )
        
        # Getting min and max and key attribute from theoretical calculations
        global_centre = self.sharingBand.center_transform
        self.global_max = (global_centre[0:2] + (self.sharingBand.global_size[0:2]  / 2)).astype(int)
        self.global_min = (global_centre[0:2] - (self.sharingBand.global_size[0:2]  / 2)).astype(int)
        primary_max_rates = sum([band.max_data_rate for band in self.primaryBands.values()])
        max_throughput = ((self.num_tx * self.sharingBand.max_data_rate) + primary_max_rates)
        max_se = max(max_throughput / (self.primary_bandwidth + self.sharing_bandwidth), primary_max_rates / self.primary_bandwidth)
        max_su = max_se # max_su will be < max_se by definition

        primary_power = tf.convert_to_tensor(np.power(10, (np.array([tx["primary_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32) # in W
        sharing_power_min = tf.convert_to_tensor(np.power(10, (np.array([tx["min_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32) # in W
        sharing_power_max = tf.convert_to_tensor(np.power(10, (np.array([tx["max_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32)
        mu_pa = tf.convert_to_tensor([tx["mu_pa"] for tx in self.transmitters.values()])
        min_pe, max_pe, _, _ = get_power_efficiency_bounds(self.primary_bandwidth, self.sharing_bandwidth, primary_power, mu_pa,
                                                           sharing_power_min, sharing_power_max)
        
        self.norm_ranges= {"throughput": (0, math.ceil(max_throughput / 1e6)), # automate this generation based on theoretical calculations
                           "se": (0, math.ceil(max_se)), 
                           "pe": (min_pe, math.ceil(max_pe)),
                           "su": (0, math.ceil(max_su)),
                           "sinr": (math.floor(self.cfg.min_sinr), math.ceil(self.cfg.max_sinr))} 

        
    def reset(self, seed=None):
        """ 
        Reset the environment to its initial state.
        
        Parameters
        ----------
        seed : int
            A random seed for determinism. Defaults to None.

        Returns
        -------
        observation : list
            Environment observation. The list of dict returned by _get_obs() 
        
        """
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
        self.rewards = tf.zeros(shape=(self.cfg.step_limit + 1, 5), dtype=tf.float32)
        self.norm_rewards = tf.zeros(shape=(self.cfg.step_limit + 1, 5), dtype=tf.float32)

        # Resetting key attributes
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
        """ 
        Step through the environment, after applying an action.
        
        Parameters
        ----------
        action : list
            The action taken for the next timestep.

        Returns
        -------
        observation : list
            Environment observation. The list of dict returned by _get_obs() 

        reward : float
            Normalised float representation of the reward the agent receieves
            from this episode.

        terminated : bool
            Flag indicating if this is the final timestep of the episode.

        truncated : bool
            Flag indicated if the episode is being cut short.

        info : dict
            Key value pairs of additional information.           
        """
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

        sharingOutput = self.sharingBand(self.users, self.sharing_state, self.transmitters, self.timestep)

        # Updating SINR maps
        self.primary_sinr_maps = [primaryBand.sinr for primaryBand in self.primaryBands.values()]    
        self.sharing_sinr_map = self.sharingBand.sinr

        # Combining the primary bands for the different transmitters:
        primaryOutput = {"bler": tf.stack([primaryOutput["bler"][i,:] for primaryOutput, i in zip(primaryOutputs, range(len(self.initial_states.values())))]), 
                         "sinr": tf.stack([primaryOutput["sinr"][i,:] for primaryOutput, i in zip(primaryOutputs, range(len(self.initial_states.values())))]),
                         "rate": tf.stack([primaryOutput["rate"][i,:] for primaryOutput, i in zip(primaryOutputs, range(len(self.initial_states.values())))])}
        self.performance.append({"Primary": primaryOutput, "Sharing": sharingOutput})

        # print({"Primary": primaryOutput, "Sharing": sharingOutput})

        # Calculating rewards
        self.rates = tf.concat([
            tf.cast(tf.stack([primaryOutput["rate"] for primaryOutput in primaryOutputs], axis=1), dtype=tf.float32),  
            tf.cast(tf.expand_dims(sharingOutput["rate"], axis=1), dtype=tf.float32)  # Expanding to [2,1,20]
        ], axis=1)  # Concatenating along axis 1 to make it [Transmitters, Bands, UEs]
        throughput, per_ue_throughput, per_ap_per_band_throughput = get_throughput(self.rates)

        fairness = get_fairness(per_ue_throughput)

        primary_power = tf.convert_to_tensor(np.power(10, (np.array([tx["primary_power"] for tx in self.transmitters.values()]) - 30) / 10), dtype=tf.float32) # in W
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
    

        # Checking validity of values before performing normalisation:
        if not (self.norm_ranges["throughput"][0] <= throughput <= self.norm_ranges["throughput"][1]):
            raise ValueError(f"Throughput value {throughput} is out of range: {self.norm_ranges['throughput']}")
        if not (self.norm_ranges["se"][0] <= se <= self.norm_ranges["se"][1]):
            raise ValueError(f"SE value {se} is out of range: {self.norm_ranges['se']}")
        if not (self.norm_ranges["pe"][0] <= pe <= self.norm_ranges["pe"][1]):
            raise ValueError(f"PE value {pe} is out of range: {self.norm_ranges['pe']}")
        if not (self.norm_ranges["su"][0] <= su <= self.norm_ranges["su"][1]):
            raise ValueError(f"SU value {su} is out of range: {self.norm_ranges['su']}")

        # Processing the reward for the agent
        updates = tf.stack([throughput,
                            fairness, 
                            se, 
                            pe, 
                            su], axis=0)
        logger.info(f"Updates [throughput, fairness, se, pe, su]: {updates.numpy()}")

        norm_updates = tf.stack([self._norm(throughput, self.norm_ranges["throughput"][0], self.norm_ranges["throughput"][1]), 
                                 fairness, # designed to be bounded [0,1]
                                 self._norm(se, self.norm_ranges["se"][0], self.norm_ranges["se"][1]), 
                                 self._norm(1/pe, 1/self.norm_ranges["pe"][1], 1/self.norm_ranges["pe"][0]), # being minimised - careful in defining ranges to avoid division by zero
                                 self._norm(su, self.norm_ranges["su"][0], self.norm_ranges["su"][1])], axis=0)
        
        indices = tf.constant([[self.timestep, 0], [self.timestep, 1], [self.timestep, 2], [self.timestep, 3], [self.timestep, 4]]) # used for updating preallocated tensor
        self.rewards = tf.tensor_scatter_nd_update(self.rewards, indices, tf.reshape(updates, (5,)))

        self.norm_rewards = tf.tensor_scatter_nd_update(self.norm_rewards, indices, tf.reshape(norm_updates, (5,)))
        reward = tf.reduce_sum(norm_updates)

        # logger.info(f"Norm Updates [throughput, fairness, se, pe, su]: {norm_updates.numpy()}")

        if (np.isnan(reward.numpy())):
            logger.critical("Reward NAN")
            return None, None, None, None, None

        # Infinite-horizon problem so we terminate at an arbitraty point - the agent does not know about this limit
        if self.timestep == self.limit:
            logger.warning("Last step of episode, Truncated.")
            self.truncated = True

        # returns the 5-tuple (observation, reward, terminated, truncated, info)
        return self._get_obs(), reward, self.terminated, self.truncated, {"rewards": updates}

    def _get_obs(self):
        """ 
        Getting the data for the current state. The design of the observation depends on the agent input structure.

        Returns
        -------
        observation : list
            Environment observation. The list of dict returned by _get_obs()          
        
        """
        state = [] # Adding normalised values to the state array

        # Iterate over each transmitter
        for tx_id, tx in enumerate(self.transmitters.values()):
            # Normalize transmitter position

            norm_tx_x = self._norm(tx["position"][0], self.global_min[0], self.global_max[0]) # Global system is in x,y,z, coverage map is in y,x 
            norm_tx_y = self._norm(tx["position"][1], self.global_min[1], self.global_max[1])

            # Normalize power level (assuming min/max power are defined)
            norm_tx_power = self._norm(tx["sharing_power"], self.cfg.min_power, self.cfg.max_power)

            tx_on_off = int(tx["state"]) 

            # Primary and sharing UEs
            primary_ues = []
            sharing_ues = []

            for user_id, user in enumerate(self.users.values()):
                x = user["position"][1] # uses the coverage map coordinate system
                y = user["position"][0]

                norm_x = self._norm(x, 0, self.valid_area.shape[1])
                norm_y = self._norm(y, 0, self.valid_area.shape[0])

                # Convert SINR to dB and normalize
                primary_sinr = [self._norm((10 * tf.math.log(self.primary_sinr_maps[tx_id][0][y][x]) / tf.math.log(10.0)).numpy(), *self.norm_ranges["sinr"])]
                if tx_on_off == 0:
                    sharing_sinr = [self._norm((10 * tf.math.log(0.0) / tf.math.log(10.0)).numpy(), *self.norm_ranges["sinr"])]
                else:
                    index = min(self.sharing_sinr_map.shape[0] - 1, tx_id)
                    sharing_sinr = [self._norm((10 * tf.math.log(self.sharing_sinr_map[index][y][x]) / tf.math.log(10.0)).numpy(), *self.norm_ranges["sinr"])]

                # Using rate instead of BLER
                if self.rates is None:
                    rate = np.array([0 for tx in range(self.num_tx + 1)], dtype=np.float32) # primaries and sharing will be in self.rates
                else:
                    rate = self.rates[tx_id,:,user_id].numpy() # note: this is probably being mapped to too small a range below in normalisation

                # Construct UE observation dictionary
                prim = {
                    "ue_pos": np.array([norm_x, norm_y]),
                    "ue_sinr": np.array(primary_sinr),
                    "ue_rate": np.array(self._norm(np.sum(rate[:-1]), self.norm_ranges["throughput"][0] * 1e6, self.norm_ranges["throughput"][1] * 1e6)), # normalised with max aggregate not max UE
                }
                shar = {
                    "ue_pos":  np.array([norm_x, norm_y]),
                    "ue_sinr":  np.array(sharing_sinr),
                    "ue_rate": np.array(self._norm(rate[-1], self.norm_ranges["throughput"][0] * 1e6, self.norm_ranges["throughput"][1] * 1e6)),
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

        return state
    
    def _norm(self, value, min_val, max_val):
        """
        Min Max Normalisation of value to range [0,1] given a range. 
        
        Parameters
        ----------
        value : float
            Value for normalisation.

        min_val : float
            Upper bound for value.

        max_val : float
            Lower bound for value.

        Returns
        -------
        value_norm : float
            Min-max normalised value between [0,1].      
        
        """
        value_clipped = tf.clip_by_value(value, min_val, max_val) # avoiding inf

        return (value_clipped - min_val) / (max_val - min_val)
    
    def render(self, episode):
        """ 
        Visualising the performance. Plots and saves graphs to directory at config save path for images. 

        Parameters
        ----------
        episode : int
            Episode reference number for file naming.

        """
        # Plotting the performance and motion
        if len(self.performance) > self.max_results_length: # managing stored results size
            self.performance = self.performance[-1*self.max_results_length:]

        # Plotting the performance
        if self.timestep >= 1:
            if self.test:
                plot_performance(step=self.timestep,
                                users=self.users,
                                performance=self.performance, 
                                save_path=self.images_path)
                pass
            plot_rewards(episode=episode,
                         step=self.timestep,
                         rewards=self.rewards,
                         save_path=self.images_path)
        
        if self.test:
            self.fig_0, self.ax_0 = plot_motion(step=self.timestep, 
                                                id="Sharing Band, Max SINR", 
                                                grid=self.valid_area, 
                                                cm=tf.reduce_max(self.sharing_sinr_map, axis=0), 
                                                color="viridis",
                                                users=self.users, 
                                                transmitters=self.transmitters, 
                                                cell_size=self.cfg.cell_size, 
                                                sinr_range=self.norm_ranges["sinr"],
                                                fig=self.fig_0,
                                                ax=self.ax_0, 
                                                save_path=self.images_path)
            
            for id, primary_sinr_map in enumerate(self.primary_sinr_maps):
                self.primary_figs[id], self.primary_axes[id] = plot_motion(step=self.timestep, 
                                                                        id=f"Primary Band {id}, SINR", 
                                                                        grid=self.valid_area, 
                                                                        cm=tf.reduce_max(primary_sinr_map, axis=0), 
                                                                        color="inferno",
                                                                        users=self.users, 
                                                                        transmitters=self.transmitters, 
                                                                        cell_size=self.cfg.cell_size, 
                                                                        sinr_range=self.norm_ranges["sinr"],
                                                                        fig=self.primary_figs[id],
                                                                        ax=self.primary_axes[id], 
                                                                        save_path=self.images_path)


        return
    
