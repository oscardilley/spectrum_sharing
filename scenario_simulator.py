""" scenario_simulator.py

For iterating through episodes of Sionna simulations. 

"""

import tensorflow as tf
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera
import numpy as np

from channel_simulator import ChannelSimulator
from logger import logger


class FullSimulator:
    """
    Full simulator

    Docs coming soon ...
    """
    def __init__(self,
                 prefix,
                 scene_name,
                 carrier_frequency,
                 bandwidth,
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
        
        # Configuration attributes
        self.prefix = prefix
        self.scene_name = scene_name
        self.carrier_frequency = carrier_frequency
        self.bandwidth = bandwidth
        self.pmax = pmax
        self.transmitters = transmitters
        self.receivers = None # updated later
        self.num_tx = len(transmitters)
        self.num_rx = num_rx
        self.max_depth = max_depth
        self.cell_size = cell_size
        self.state = initial_state

        # OFDM/ 5G NR Specific Attributes
        self.subcarrier_spacing = subcarrier_spacing
        self.fft_size = fft_size
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size

        # Instantiating the channel simulator:
        self.simulator = ChannelSimulator(self.num_tx, 
                                          self.num_rx, 
                                          self.subcarrier_spacing,
                                          self.fft_size)

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
        self.cm, area = self._coverage_map(init=True) # used to determine all possible valid areas
        self.cm, self.sinr = self._coverage_map(init=False) # correcting power levels
        self.grid = self._validity_matrix(self.cm.num_cells_x, self.cm.num_cells_y, area)
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

    def __call__(self, receivers, state, transmitters=None, timestep=None, path=None):
        """ Running an episode. """
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
                sinrs.append(tf.constant(-1000, shape=(self.num_rx), dtype=tf.float32))
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
       
        a, tau = paths.cir(los=True)
        num_active_tx = int(tf.reduce_sum(tf.cast(self.state, tf.int32)))
        self.simulator.update_channel(num_active_tx, a, tau)
        self.simulator.update_sinr(per_rx_sinr)

        # Bit level simulation
        bler, sinr = self.simulator(block_size=self.batch_size)

        # Handling dynamic size of state:
        for state in self.state:
            if bool(state) is True:
                blers.append(bler[count])
                sinrs.append(tf.clip_by_value(sinr[count], -1000, 1000))
                count += 1
            else:
                blers.append(tf.ones(self.num_rx, dtype=tf.float64))
                sinrs.append(tf.constant(-1000, shape=(self.num_rx), dtype=tf.float32)) # SINR in dB so need to force to -inf

        # Calculate the users achieving < 1 BLER to schedule  
        num_scheduled = tf.math.count_nonzero(tf.less(blers, 1)) 
        if num_scheduled.numpy() == 0:
            logger.warning("Transmitters on but all BLER = 1.")
            max_data_sent_per_ue = 0 
        else:
            max_data_sent_per_ue = (self.simulator.pusch_config.tb_size * self.batch_size) / num_scheduled # bits
        time_step = (self.batch_size / self.simulator.pusch_config.carrier.num_slots_per_frame) * self.simulator.pusch_config.carrier.frame_duration
        max_rate_per_ue = max_data_sent_per_ue / time_step # in order to account for different numerologies
        rates = [(1 - bler) * max_rate_per_ue for bler in blers]

        results = {"bler": tf.stack(blers), "sinr": tf.stack(sinrs), "rate": tf.stack(rates)}

        # Only saving if timestep is provided (sharing band only):
        if timestep is not None:
            self.scene.render_to_file(camera="cam1",
                                    filename=str(path)+f"Camera 1 Step {timestep}.png",
                                    paths=paths,
                                    show_paths=True,
                                    show_devices=True,
                                    #coverage_map=self.cm,
                                    cm_db_scale=True,
                                    cm_vmin=-100,
                                    cm_vmax=100,
                                    resolution=[1920,1080],
                                    fov=55)
            self.scene.render_to_file(camera="cam2",
                                    filename=str(path)+f"Camera 2 Step {timestep}.png",
                                    paths=paths,
                                    show_paths=True,
                                    show_devices=True,
                                    #coverage_map=self.cm,
                                    cm_db_scale=True,
                                    cm_vmin=-100,
                                    cm_vmax=100,
                                    resolution=[1920,1080],
                                    fov=55)
                                    

        return results


    def update_receivers(self, receivers):
        """ Changing the receivers within the scene. 
        
        Note that coordinate handling is delicate between Sionna scenes, coverage maps and utility functions. 
        
        """
        per_rx_sinr_db = []
        max_x = float(self.scene.size[0].numpy())
        max_y = float(self.scene.size[1].numpy())
        if len(self.scene.receivers) == 0:
            # Adding receivers for the first time. 
            for rx_id, rx in enumerate(receivers.values()):
                reversed_coords = np.array([rx["position"].numpy()[1],rx["position"].numpy()[0], rx["position"].numpy()[2]])
                pos = (reversed_coords * self.cell_size) - np.array([round(max_x/2), round(max_y/2), 0]) + self.center_transform
                self.scene.add(Receiver(name=f"rx{rx_id}",
                                        position=pos, 
                                        color=rx["color"],)) 
                sinr_db = 10 * tf.math.log(self.sinr[:,rx["position"][0], rx["position"][1]]) / tf.math.log(10.0)
                per_rx_sinr_db.append(sinr_db)                        

        else:
            for rx_id, rx in enumerate(receivers.values()):
                reversed_coords = np.array([rx["position"].numpy()[1],rx["position"].numpy()[0], rx["position"].numpy()[2]])
                pos = (reversed_coords * self.cell_size) - np.array([round(max_x/2), round(max_y/2), 0]) + self.center_transform
                self.scene.receivers[f"rx{rx_id}"].position = pos
                sinr_db = 10 * tf.math.log(self.sinr[:,rx["position"][0], rx["position"][1]]) / tf.math.log(10.0)
                per_rx_sinr_db.append(sinr_db)    
        self.receivers = receivers

        return tf.concat(per_rx_sinr_db, axis=0) 
    
    def reset(self):
        """ Resetting the simulator."""

        return self._scene_init()