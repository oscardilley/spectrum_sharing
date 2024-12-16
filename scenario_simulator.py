""" scenario_simulator.py

For iterating through episodes of Sionna simulations. 

"""

import tensorflow as tf
# import numpy as np
# import sionna
# from time import perf_counter
# from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, ApplyOFDMChannel #, CIRDataset, OFDMChannel
# from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
# from sionna.utils import compute_bler, compute_ber, ebnodb2no
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
# from hydra import compose, initialize 
# from omegaconf import OmegaConf
# import matplotlib.pyplot as plt
# import matplotlib as mpl

from channel_simulator import ChannelSimulator


class FullSimulator:
    """
    Full simulator

    Docs coming soon ...
    """
    def __init__(self,
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
                 num_time_steps = 14,
                 ):
        
        # Configuration attributes
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
        self.block_size = 1352

        # Instantiating the channel simulator:
        self.simulator = ChannelSimulator(self.num_tx, 
                                          self.num_rx, 
                                          self.subcarrier_spacing,
                                          self.fft_size)

        # Creating the scene
        self.scene = self._create_scene()
        self.cm, self.sinr = self._coverage_map()
        self.grid = self._validity_matrix()


    def _create_scene(self):
        """ Creating the simulation scene, only called once during intialisation."""
        sn = load_scene(self.scene_name)
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
    
    def _coverage_map(self):
        """ Update or generate coverage map. """
        # Adding the transmitters
        for tx in self.transmitters.keys():
            transmitter = Transmitter(name=self.transmitters[tx]["name"],
                                            position=self.transmitters[tx]["position"], # top of red building on simple street canyon
                                            orientation=self.transmitters[tx]["orientation"], # angles [alpha,beta,gamma] corrsponding to rotation around [z,y,x] axes (yaw, pitch, roll)
                                            color=self.transmitters[tx]["color"],
                                            power_dbm=self.pmax)
            self.scene.add(transmitter)
            transmitter.look_at(self.transmitters[tx]["look_at"])

            # Need to use self.state and detect init




        # Generating a coverage map for max power to establish valid UE regions
        cm = self.scene.coverage_map(max_depth=self.max_depth,           # Maximum number of ray scene interactions
                             diffraction=True,
                             scattering=True, 
                             cm_cell_size=(self.cell_size, self.cell_size),   # Resolution of the coverage map, smaller is better
                            )
        
        return cm, cm.sinr # [num_tx, num_cells_y, num_cells_x], tf.float
    
    def _validity_matrix(self):
        """ Calculating the valid user area"""
        x_max = self.cm.num_cells_x
        y_max = self.cm.num_cells_y

        grid = tf.zeros((y_max, x_max), dtype=tf.bool)
        validity_matrix = tf.reduce_sum(self.cm.sinr, axis=0) # [num_tx, num_cells_y, num_cells_x] -> [num_cells_y, num_cells_x], tf.float
        for y in range(y_max):
            for x in range(x_max):
                if validity_matrix[y,x] != 0:
                    grid = tf.tensor_scatter_nd_update(grid, [[y, x]], [True])
        
        return grid

    def __call__(self, receivers, state, transmitters=None):
        """ Running an episode. """
        # Updating transmitters if required
        if transmitters is not None:
            # Only updating transmitters if parameters have changed
            self.transmitters = transmitters

        # Apply the state and update the coverage map to obtain new 
        if tf.reduce_all(tf.equal(state, self.state)) is False or transmitters is not None:
            self.state = state
            self.cm, self.sinr = self._coverage_map() 

        # Updating the receivers
        per_rx_sinr = self.update_receivers(receivers)

        paths = self.scene.compute_paths(max_depth=self.max_depth, diffraction=True)
        paths.normalize_delays = False
        
        paths.apply_doppler(sampling_frequency=self.subcarrier_spacing, # Set to 15e3 Hz
                            num_time_steps=self.num_time_steps, # Number of OFDM symbols
                            tx_velocities=[self.cell_size * tf.convert_to_tensor(transmitter["direction"], dtype=tf.int64) for transmitter in self.transmitters.values()], # [batch_size, num_tx, 3] shape
                            rx_velocities=[self.cell_size * receiver["direction"] for receiver in receivers.values()]) # [batch_size, num_rx, 3] shape
        a, tau = paths.cir(los=True)
        self.simulator.update_channel(a, tau)
        self.simulator.update_sinr(per_rx_sinr)

        # Bit level simulation
        ber, sinr = self.simulator(block_size=self.block_size)

        # Output processing
            # May Need to handle inf etc.

        rewards = {"ber": ber, "sinr": sinr}

        return rewards


    def update_receivers(self, receivers):
        """ Changing the receivers within the scene. """
        per_rx_sinr_db = []
        if len(self.scene.receivers) == 0:
            # Adding receivers for the first time.
            for rx_id, rx in enumerate(receivers.values()):
                self.scene.add(Receiver(name=f"rx{rx_id}",
                                        position=rx["position"], 
                                        color=rx["color"],)) 
                sinr_db = 10 * tf.math.log(self.sinr[:,rx["position"][0], rx["position"][1]]) / tf.math.log(10.0)
                per_rx_sinr_db.append(sinr_db)                        

        else:
            for rx_id, rx in enumerate(receivers.values()):
                self.scene.receivers[f"rx{rx_id}"].position = rx["position"]
                sinr_db = 10 * tf.math.log(self.sinr[:,rx["position"][0], rx["position"][1]]) / tf.math.log(10.0)
                per_rx_sinr_db.append(sinr_db)    
        self.receivers = receivers

        return tf.concat(per_rx_sinr_db, axis=0) 
