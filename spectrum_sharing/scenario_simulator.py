""" scenario_simulator.py

For iterating through episodes of Sionna simulations. 

"""

import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera
import numpy as np
from time import perf_counter

from spectrum_sharing.channel_simulator import ChannelSimulator
from spectrum_sharing.logger import logger
from spectrum_sharing.plotting import prop_fair_plotter


class FullSimulator:
    """
    NVIDIA Sionna-based 5G-NR simulator for a single band with multiple 
    transmitters sharing the same spectrum.

    Using ray-tracing, deterministic 

    Parameters
    ----------
    cfg : dict
        Top level configuration dictionary.
    
    prefix : str
        An arbitrary unique identifier for the instance for plot saving etc.

    scene_name : path to Mitsuba XML scene file or sionna.rt.scene object.
        Points to the scene to be loaded.

    carrier_frequency : float
        Centre frequency in Hz.

    pmax : int
        The maximum power in dBm, used to determine the grid.

    transmitters : dict
        A dictionary of the transmitters and their parameters, defined in the config yaml.

    num_rx : int
        How many users are in the scene.

    max_depth : int
        How many steps of ray interactions are considered.

    cell_size : int
        Defines the resolution of coverage maps, relative to the scene unit size. Advised to use 1.

    initial_state : tf.Tensor of tf.bool
        Of [len(transmitters)] with boolean values corresponding to whether each access point is ON/OFF when initialised.

    subcarrier_spacing : int
        Subcarrier spacing in Hz. Links to 5G numerologies. E.g. if u=0, subcarrier_spacing=15e3.

    fft_size : 
        Actually the number of subcarriers in the frequency domain. 
        FFT size would be derived from this, as the power of 2 typically > 2X number of subcarriers.

    batch_size : int
        How many times to simulate each link.

    num_time_steps : int
        Number of OFDM symbols per subframe. Defaults to 14.

    Inputs
    ------
    receivers : nested dict
        Dictionary for the users, first level key f"ue{ue_id}", second level
        provides the position, direction, color, buffer attributes for the users.

    state : tf.Tensor of tf.bool
        Of [len(transmitters)] with boolean values corresponding to whether each access point is ON/OFF.

    transmitters : dict
        Updated transmitter dict with modified power levels or position for example.

    timestep :
        Defaults to None for episode initialisation.

    Outputs
    -------
    results: dict 
        Dictionary containing sinrs, blers and rates as tf.Tensor for the different users to each transmitter in the band.
    
    """
    def __init__(self,
                 cfg,
                 prefix,
                 scene_name,
                 carrier_frequency,
                 pmax,
                 transmitters,
                 num_rx,
                 max_depth,
                 cell_size,
                 initial_state,
                 subcarrier_spacing,
                 fft_size,
                 batch_size,
                 num_time_steps=14,
                 ):
        """ 
        Initialisation. 
        """
        
        # Configuration attributes
        self.cfg = cfg
        self.prefix = prefix
        self.scene_name = scene_name
        self.carrier_frequency = carrier_frequency
        self.pmax = pmax
        self.transmitters = transmitters
        self.receivers = None # updated later
        self.num_tx = len(transmitters)
        self.num_rx = num_rx
        self.max_depth = max_depth
        self.cell_size = cell_size
        self.state = initial_state

        # OFDM/ 5G NR Specific Attributes
        self.bandwidth = subcarrier_spacing * fft_size
        self.subcarrier_spacing = subcarrier_spacing
        self.fft_size = fft_size
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size

        # Instantiating the channel simulator:
        self.simulator = ChannelSimulator(self.num_tx, 
                                          self.num_rx, 
                                          self.subcarrier_spacing,
                                          self.fft_size)
        
        # Calculates instantaneous max data rate, accounting for the numerology - same for all users as not changing MCS and therefore not changing TB size
        self.max_data_rate = (self.simulator.pusch_config.tb_size * self.simulator.pusch_config.carrier.num_slots_per_frame) / self.simulator.pusch_config.carrier.frame_duration
        logger.info("Displaying the PUSCH configuration in the terminal.")
        self.simulator.pusch_config.show()

        # Creating the scene
        self.scene= None
        self.cm = None
        self.sinr = None
        self.grid = None
        self._scene_init()

    def _scene_init(self):
        """ Initialising the scene. """
        self.scene = self._create_scene()
        self.center_transform = np.round(np.array([self.scene.center.numpy()[0], self.scene.center.numpy()[1], 0]))
        self.global_size = self.scene.size.numpy()
        self.cm, area = self._coverage_map(init=True) # used to determine all possible valid areas
        self.cm, self.sinr = self._coverage_map(init=False) # correcting power levels
        self.grid = self._validity_matrix(self.cm.num_cells_x, self.cm.num_cells_y, area)
        self.avg_throughput = np.zeros(self.num_rx) # used for scheduling

        return

    
    def _create_scene(self):
        """ Creating the simulation scene, only called once during intialisation."""
        sn = load_scene(self.scene_name)
        sn.add(Camera("cam1", position=[300, -320, 230], look_at=sn.center))
        sn.add(Camera("cam2", position=[sn.center.numpy()[0],sn.center.numpy()[1],500], orientation=[np.pi/2, np.pi/2,-np.pi/2]))

        sn.synthetic_array=True # False = ray tracing done per antenna element
        sn.frequency = self.carrier_frequency
        sn.bandwidth = self.bandwidth

        # Defining the radio units for the scene
        sn.tx_array = PlanarArray(num_rows=4,
                                 num_cols=4,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901", # classic omnidirectional 5G antenna
                                 polarization="VH")
        sn.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5, # relative to wavelength
                                 horizontal_spacing=0.5,
                                 pattern="iso", # random orientation of UE
                                 polarization="cross")
    
        return sn
    
    def _coverage_map(self, init=False):
        """ Update or generate coverage map. """
        # Adding the transmitters and updating scene - Sionna only holds one scene in GPU memory at a time
        self.scene.frequency = self.carrier_frequency
        self.scene.bandwidth = self.bandwidth

        for tx, state in zip(self.transmitters.keys(), self.state):
            if bool(state) or init: # all transmitters are added for initialisation to map area             
                if tx not in self.scene.transmitters:
                    # Adding new transmitters
                    transmitter = Transmitter(name=self.transmitters[tx]["name"],
                                              position=np.array(self.transmitters[tx]["position"]) + self.center_transform, # add center position to correct for non origin scene centres
                                              orientation=self.transmitters[tx]["orientation"], # angles [alpha,beta,gamma] corrsponding to rotation around [z,y,x] axes (yaw, pitch, roll)
                                              color=self.transmitters[tx]["color"],
                                              power_dbm=self.pmax)
                    self.scene.add(transmitter)
                    self.scene.transmitters[tx].look_at(self.transmitters[tx]["look_at"])

                if not init:
                    self.scene.transmitters[tx].power_dbm = self.transmitters[tx][f"{self.prefix}_power"]
                    self.scene.transmitters[tx].look_at(self.transmitters[tx]["look_at"])

            else:
                # Removing transmitter
                self.scene.remove(tx)

        if self.scene.transmitters == {}:
            return self.cm, tf.zeros(shape=(self.num_tx, self.cm.num_cells_y, self.cm.num_cells_x), dtype=tf.float32)

        # Generating a coverage map for max power to establish valid UE regions
        cm = self.scene.coverage_map(max_depth=self.max_depth,           # Maximum number of ray scene interactions
                                     diffraction=True,
                                     scattering=True, 
                                     cm_cell_size=(self.cell_size, self.cell_size),   # Resolution of the coverage map, smaller is better
                                     )
        
        
        return cm, cm.sinr # [num_tx, num_cells_y, num_cells_x], tf.float
    
    def _validity_matrix(self, x_max, y_max, valid_area):
        """ Calculating the valid user area"""

        grid = tf.zeros((y_max, x_max), dtype=tf.bool)
        validity_matrix = tf.reduce_sum(valid_area, axis=0) # [num_tx, num_cells_y, num_cells_x] -> [num_cells_y, num_cells_x], tf.float
        for y in range(y_max):
            for x in range(x_max):
                if validity_matrix[y,x] != 0:
                    grid = tf.tensor_scatter_nd_update(grid, [[y, x]], [True])
        
        return grid

    def __call__(self, receivers, state, transmitters=None, timestep=None):
        """ Running an episode with proportional fair scheduling. """
        # NB: SINR in dB here, different to coverage maps
        blers = [] # used to estimate throughput
        sinrs = []
        rates = []
        count = 0
        # Updating transmitters
        self.transmitters = transmitters

        # Apply the state and update the coverage map to obtain new 
        self.state = state
        self.cm, self.sinr = self._coverage_map() 

        if np.all(self.state.numpy() == False):
            for state in self.state:
                logger.warning("Skipping calculation as all states are False.")
                blers.append(tf.ones(self.num_rx, dtype=tf.float64))
                sinrs.append(tf.constant(self.cfg.min_sinr, shape=(self.num_rx), dtype=tf.float32))
                rates.append(tf.zeros(shape=(self.num_rx), dtype=tf.float32))
            results = {"bler": tf.stack(blers), "sinr": tf.stack(sinrs), "rate": tf.stack(rates)}
        
            return results

        # Updating the receivers
        per_rx_sinr = self.update_receivers(receivers)
        paths = self.scene.compute_paths(max_depth=self.max_depth, diffraction=True)
        paths.normalize_delays = False

        paths.apply_doppler(sampling_frequency=self.subcarrier_spacing,
                            num_time_steps=self.num_time_steps, # Number of OFDM symbols
                            #tx_velocities=[self.cell_size * tf.convert_to_tensor(transmitter["direction"], dtype=tf.int64) for transmitter in self.transmitters.values()], # [batch_size, num_tx, 3] shape
                            rx_velocities=[self.cell_size * receiver["direction"] for receiver in receivers.values()]) # [batch_size, num_rx, 3] shape
       
        a, tau = paths.cir(los=True) # generating the channel impulse response

        num_active_tx = int(tf.reduce_sum(tf.cast(self.state, tf.int32)))
        self.simulator.update_channel(num_active_tx, a, tau)
        self.simulator.update_sinr(per_rx_sinr)

        # Bit level simulation
        start = perf_counter()
        bler, sinr = self.simulator(block_size=self.batch_size)
        end = perf_counter()
        total = round(end - start, 3)
        logger.info(f"Time taken for bit level simulation: {total}")

        # Handling dynamic size of state:
        for state in self.state:
            if bool(state) is True:
                blers.append(bler[count])
                sinrs.append(tf.clip_by_value(sinr[count], self.cfg.min_sinr, self.cfg.max_sinr))
                count += 1
            else:
                blers.append(tf.ones(self.num_rx, dtype=tf.float64))
                sinrs.append(tf.constant(self.cfg.min_sinr, shape=(self.num_rx), dtype=tf.float32)) # SINR in dB so need to force close to -inf


        # Calculate the users achieving < 1 BLER to schedule  
        num_scheduled = tf.math.count_nonzero(tf.less(blers, 1)) # number that have a connection at all
        if num_scheduled.numpy() == 0:
            logger.warning("Transmitters on but all BLER = 1.")
            max_data_sent_per_ue = 0 # none could be scheduled 
            rates = [(1 - bler) * max_data_sent_per_ue for bler in blers]
        else:
            # Transport block here tells us the number of bits per slot, for 1 PRB/RB over 14 OFDM symbols
            # 14 OFDM symbols per slot typically
            # There is a variable number of slots per subframe, 1 at 15kHz, 2 at 30kHz, etc. 2^u where u is the numerology 
            # A subframe is always 1ms annd there are 10ms in a frame.
            # 12 subcarriers per PRB, TDD is applied within a slot at the OFDM level
            # n_size_grid gives number of resource blocks in the carrier resource grid, each 1 OFDM symbol wide
            # target_coderate gives us the number of information bits/total 
            # num_resource_blocks is the number of blocks allocated for the PUSCH transmissions, we can only schedule 
            # num_res_per_prb gives the number of resource elements per PRB available for data.

            # Proportional fair scheduler approximation, assuming single user MIMO and 1000 slots as we are stepping in 1s increments of 1s
            num_slots = self.simulator.pusch_config.carrier.num_slots_per_frame / self.simulator.pusch_config.carrier.frame_duration # 1000 for u=0, 2000 for u=1, etc. 
            number_rbs = self.simulator.pusch_config.num_resource_blocks # number of resource blocks in the carrier resource grid
            max_data_sent_per_rb = self.max_data_rate / (num_slots * number_rbs) # bits per RB over 14 OFDM symbols

            # Run the PF scheduler
            start = perf_counter()
            for tx in range(len(blers)):
                grid_alloc, rate = self.proportional_fair_scheduler(
                    blers[tx], max_data_sent_per_rb, num_slots, number_rbs, alpha=0.1
                )

                if (timestep is not None) and (self.prefix == "sharing") and self.state[tx]:
                    if timestep < 10:
                        # don't plot if not active and only plot for first timesteps
                        prop_fair_plotter(timestep, 
                                        tx,
                                        grid_alloc, 
                                        self.num_rx,
                                        rate, 
                                        max_data_sent_per_rb,
                                        save_path=self.cfg.images_path)
                rates.append(tf.convert_to_tensor(rate))
            end = perf_counter()
            total = round(end-start, 3)
            logger.info(f"Scheduling took {total}s")

        results = {"bler": tf.stack(blers), "sinr": tf.stack(sinrs), "rate": tf.stack(rates)}                               

        return results


    def update_receivers(self, receivers):
        """ 
        Changing the receivers within the scene. Note: coordinate handling is delicate between Sionna scenes, coverage maps and utility functions. 
        """
        per_rx_sinr_db = []
        max_x = float(self.scene.size[0].numpy())
        max_y = float(self.scene.size[1].numpy())
        if len(self.scene.receivers) == 0:
            # Adding receivers for the first time. 
            for rx_id, rx in enumerate(receivers.values()):
                reversed_coords = np.array([rx["position"][1], rx["position"][0], rx["position"][2]])
                pos = (reversed_coords * self.cell_size) - np.array([round(max_x/2), round(max_y/2), 0]) + self.center_transform
                self.scene.add(Receiver(name=f"rx{rx_id}",
                                        position=pos, 
                                        color=rx["color"],)) 
                sinr_db = 10 * tf.math.log(self.sinr[:,rx["position"][0], rx["position"][1]]) / tf.math.log(10.0) # log of zero will occur here, causing inf, clipped later
                per_rx_sinr_db.append(sinr_db)                        

        else:
            for rx_id, rx in enumerate(receivers.values()):
                reversed_coords = np.array([rx["position"][1], rx["position"][0], rx["position"][2]])
                pos = (reversed_coords * self.cell_size) - np.array([round(max_x/2), round(max_y/2), 0]) + self.center_transform
                self.scene.receivers[f"rx{rx_id}"].position = pos
                sinr_db = 10 * tf.math.log(self.sinr[:,rx["position"][0], rx["position"][1]]) / tf.math.log(10.0)
                per_rx_sinr_db.append(sinr_db)    
        self.receivers = receivers

        return tf.reshape(tf.transpose(tf.stack(per_rx_sinr_db)) , [-1])
    
    def proportional_fair_scheduler(self, blers, max_data_sent_per_rb, num_slots, number_rbs, alpha=0.2, disconnect=0.95):
        """
        Schedule RBs to users using proportional fair scheduling, 
        while excluding users with no connection (BLER>disconnect).
        
        Parameters
        ----------
        blers: ID tensor of blers for each user.
        max_data_sent_per_rb: maximum bits per RB (given the numerology etc.).
        num_slots: total number of time slots in 1 second.
        number_rbs: number of resource blocks per slot.
        alpha: EWMA weight for throughput update, a smaller alpha remembers the past more
        
        Returns
        -------
        grid_alloc: np.ndarray
            [time slots, RBs] with allocated user IDs.
        avg_throughput: np.ndarray
            [Users], final average throughput for each user.
        """
        blers = blers.numpy()
        num_users = len(blers)
        
        # Compute the instantaneous rate per RB for each user.
        # If BLER > 0.9 (no connection), instantaneous rate is 0.
        instantaneous_rates = np.array([
            (1 - bler) * max_data_sent_per_rb if bler < disconnect else 0.0
            for bler in blers
        ])
        
        # Create a grid to hold the scheduling decision for each RB.
        grid_alloc = np.zeros((int(num_slots), int(number_rbs)), dtype=int)
        
        # To accumulate the bits sent to each user.
        bits_sent_per_user = np.zeros(num_users)
        
        # Simulate scheduling over each resource block in the grid.
        for t in range(int(num_slots)):
            for rb in range(int(number_rbs)):
                # Compute PF metrics; if a user's instantaneous rate is zero (no connection),
                # assign a very low metric so it is never scheduled.
                pf_metrics = np.where(instantaneous_rates > 0,
                                    instantaneous_rates / (self.avg_throughput + 1e-6),
                                    0)

                # Select the user(s) with the highest PF metric.
                max_indices = np.flatnonzero(pf_metrics == np.max(pf_metrics))

                # Randomly select one of the max indices.
                scheduled_user = int(np.random.choice(max_indices))

                # Record the allocation.
                grid_alloc[t, rb] = scheduled_user
                
                # Add the bits transmitted in this RB for the scheduled user.
                bits_sent_per_user[scheduled_user] += instantaneous_rates[scheduled_user]
                
                # Update the scheduled user's average throughput with an EWMA.
                self.avg_throughput[scheduled_user] = ((1 - alpha) * self.avg_throughput[scheduled_user] 
                                                + alpha * instantaneous_rates[scheduled_user])

        
        # Since the simulation covers 1 second, the total bits sent equals bps.
        user_rates = bits_sent_per_user  # bits per second

        return grid_alloc, user_rates
    
    def reset(self):
        """ Resetting the simulator."""
        return self._scene_init()