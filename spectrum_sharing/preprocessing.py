""" preprocessing.py

Pre-compute and save SINR, BER and BLER maps. Must be run as a module: python3 -m spectrum_sharing.preprocessing"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING not displayed
import tensorflow as tf
import sionna
from hydra import compose, initialize 
import numpy as np
import matplotlib.pyplot as plt 
from itertools import product
import gc
import subprocess
from time import perf_counter

from spectrum_sharing.logger import logger
from spectrum_sharing.scenario_simulator import FullSimulator
from spectrum_sharing.channel_simulator import ChannelSimulator

CONFIG_NAME = "preprocessing" # the only config selection in the script

def main(cfg):
    """
    Saving maps of the SINR and BLER for the primary and sharing bands.
    """
    # Setting up constants and data structures
    transmitters = dict(cfg.transmitters)
    grid_transmitters = dict(cfg.grid_transmitters)
    num_tx = len(transmitters)
    num_tx_grid = len(grid_transmitters) # use to generate a better grid
    primary_bandwidth = cfg.primary_fft_size * cfg.primary_subcarrier_spacing
    sharing_bandwidth = cfg.primary_fft_size * cfg.primary_subcarrier_spacing
    users = {}
    primary_maps_filename = "primary_maps.npy"
    sharing_maps_filename = "sharing_maps.npy"
    grid_filename = "grid.npy"

    # Generating the possible states
    logger.info("Generating the possible states.")
    powers = [[0] for _ in range(num_tx)]
    for id, tx in enumerate(transmitters.values()):
        powers[id] = powers[id] + list(range(int(tx["min_power"]), int(tx["max_power"]) + 1))
    powers = list(product(*[powers[id] for id in range(num_tx)])) # all possible combinations of powers

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Primary bands processing
    for id, tx in enumerate(transmitters.values()):
        initial_state = tf.cast(tf.one_hot(id, num_tx, dtype=tf.int16), dtype=tf.bool)

        if id == 0:
            logger.warning("Generating grid and users.")
            primaryBand = FullSimulator(cfg=cfg,
                                        prefix="primary",
                                        scene_name= cfg.scene_path + "simple_OSM_scene.xml", #sionna.rt.scene.simple_street_canyon,
                                        carrier_frequency=tx["primary_carrier_freq"],
                                        pmax=100, # maximum power
                                        transmitters=grid_transmitters, # use for creating a grid
                                        # transmitters=transmitters,
                                        num_rx = 1,
                                        max_depth= cfg.max_depth,
                                        cell_size= cfg.cell_size,
                                        initial_state = tf.cast(tf.one_hot(id, num_tx_grid, dtype=tf.int16), dtype=tf.bool),
                                        subcarrier_spacing = cfg.primary_subcarrier_spacing,
                                        fft_size = cfg.primary_fft_size,
                                        batch_size=cfg.batch_size,
                                        )
            logger.info(f"Scene size: {primaryBand.scene.size.numpy()}")
            logger.info(f"Scene centre: {primaryBand.scene.center.numpy()}")
            

            grid = primaryBand.grid # use the same grid throughout to prevent minor discrepancies
            # plot_coverage_map(grid, cfg.images_path, title=f"Grid", plot_min=0, plot_max=1)
            num_tx = len(transmitters)
            np.save(cfg.assets_path + grid_filename, grid)
            if users == {}:
                users = generate_users(grid, users) # Generate the user positions from the validity matrix
                sinrs = np.full((num_tx, primaryBand.grid.shape[0], primaryBand.grid.shape[1]), cfg.min_sinr, dtype=np.float32) # initialise to the minimum SINR dB value
                blers = np.ones((num_tx, primaryBand.grid.shape[0], primaryBand.grid.shape[1]), dtype=np.float32) # initialise to the minimum SINR dB value
                bers = np.ones((num_tx, primaryBand.grid.shape[0], primaryBand.grid.shape[1]), dtype=np.float32) # initialise to the minimum SINR dB value

            # Slicing up user groups to avoid exceeding memory
            num_tx = len(transmitters)
            users_list = list(users.items())
            batch_size = len(users_list) // cfg.num_slices

            slices = [users_list[i * batch_size: (i + 1) * batch_size] for i in range(cfg.num_slices - 1)]
            slices.append(users_list[(cfg.num_slices - 1) * batch_size:])  # Last slice with remaining users

        if not os.path.exists(cfg.assets_path + primary_maps_filename):
            logger.info("Processing primary maps.")

            for slice in slices: 
                start = perf_counter()
                if len(slice) == 0:
                    break
                # print_gpu_memory_stats()
                if len(slice) != primaryBand.num_rx: # avoiding reinstantiating to hopefully maintain the jit compilation
                    tf.keras.backend.clear_session()
                    if hasattr(primaryBand, 'simulator') and hasattr(primaryBand.simulator, 'h_freq'):
                        primaryBand.simulator.h_freq = None
                    del primaryBand
                    gc.collect()
                    primaryBand = FullSimulator(cfg=cfg,
                                                prefix="primary",
                                                scene_name= cfg.scene_path + "simple_OSM_scene.xml", #sionna.rt.scene.simple_street_canyon,
                                                carrier_frequency=tx["primary_carrier_freq"],
                                                pmax=100, # maximum power
                                                transmitters=transmitters, 
                                                num_rx = len(slice),
                                                max_depth= cfg.max_depth,
                                                cell_size= cfg.cell_size,
                                                initial_state = initial_state,
                                                subcarrier_spacing = cfg.primary_subcarrier_spacing,
                                                fft_size = cfg.primary_fft_size,
                                                batch_size=cfg.batch_size,
                                                )
                    primaryBand.grid = grid # giving it the primary grid for consistency - unsure if necessary
                primaryBand.receivers = None # clearing the previous receivers forces an update

                users_slice = dict(slice)
                # MCS can be changed here - tuple in (table, index)
                output = primaryBand(users_slice, initial_state, transmitters, mcs=None) # not using rate, calculate in the loop
                logger.info("Completed slice.")

                for user, sinr, bler, ber in zip(users_slice.values(), output["sinr"][id].numpy(), output["bler"][id].numpy(), output["ber"][id].numpy()):
                    y,x = user["position"][0], user["position"][1]
                    sinrs[id,y,x] = sinr
                    blers[id,y,x] = bler
                    bers[id,y,x] = ber

                plot_coverage_map(sinrs[id,:,:], cfg.assets_path + "Maps/", title=f"SINRs map for Transmitter {id}")
                plot_coverage_map(blers[id,:,:], cfg.assets_path + "Maps/", title=f"BLERs map for Transmitter {id}", plot_min=0, plot_max=1)
                plot_coverage_map(bers[id,:,:], cfg.assets_path + "Maps/", title=f"BERs map for Transmitter {id}", plot_min=0, plot_max=1)

                end = perf_counter()
                time = end - start
                logger.info(f"Slice took : {time}")

            if id == num_tx - 1:
                # Save on final loop
                logger.warning(f"Saving primary maps at {cfg.assets_path + primary_maps_filename}.")
                primary_maps = np.stack([sinrs, blers, bers], axis=0).astype(np.float16) # save as float 16 ()
                np.save(cfg.assets_path + primary_maps_filename, primary_maps)

        else:
            logger.warning("Skipping primary maps computation. File already exists.")

    tf.keras.backend.clear_session()
    if hasattr(primaryBand, 'simulator') and hasattr(primaryBand.simulator, 'h_freq'):
        primaryBand.simulator.h_freq = None
    del primaryBand
    gc.collect()

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Processing the sharing bands
    if not os.path.exists(cfg.assets_path + sharing_maps_filename):

        logger.info("Processing sharing bands.")
        sinrs = np.full((len(powers), num_tx, grid.shape[0], grid.shape[1]), -1000, dtype=np.float32) # initialise to the minimum SINR dB value
        blers = np.ones((len(powers), num_tx, grid.shape[0], grid.shape[1]), dtype=np.float32) # initialise to the minimum SINR dB value
        bers = np.ones((len(powers), num_tx, grid.shape[0], grid.shape[1]), dtype=np.float32) # initialise to the minimum SINR dB value
                
        for id, tx_powers in enumerate(powers):
            if (id < 0):
                # Skipping ahead
                logger.info(f"Skipping {id}.")
                continue
            logger.info(f"Starting on state {id}.")
            state = [True, True]
            for tx_id, tx_power in enumerate(tx_powers):
                if tx_power == 0:
                    state[tx_id] = False
                transmitters[f"tx{tx_id}"]["sharing_power"] = tx_power
            
            state = tf.cast(tf.convert_to_tensor(state), dtype=tf.bool)
            logger.warning(f"State: {state}")
            
            sharingBand = FullSimulator(cfg=cfg,
                                        prefix="sharing",
                                        scene_name=cfg.scene_path + "simple_OSM_scene.xml",
                                        carrier_frequency=cfg.sharing_carrier_freq,
                                        pmax=50, # maximum power
                                        transmitters=transmitters,
                                        num_rx = len(slices[0]), # changed further on
                                        max_depth=cfg.max_depth,
                                        cell_size=cfg.cell_size,
                                        # initial_state = tf.convert_to_tensor([True for _ in range(num_tx)], dtype=tf.bool), 
                                        initial_state = tf.convert_to_tensor(state, dtype=tf.bool),
                                        subcarrier_spacing = cfg.sharing_subcarrier_spacing,
                                        fft_size = cfg.sharing_fft_size,
                                        batch_size= cfg.batch_size,
                                        )
            sharingBand.grid = grid # giving it the primary grid for consistency - unsure if necessary

            for slice_id, slice in enumerate(slices): # only using the first slice for now
                logger.info(f"Slice {slice_id} in set {id}.")
                start = perf_counter()
                if len(slice) == 0:
                    break
                # print_gpu_memory_stats()
                if len(slice) != sharingBand.num_rx: # avoiding reinstantiating to hopefully maintain the jit compilation
                    del sharingBand
                    gc.collect()
                    print("Changing num_rx.")
                    sharingBand = FullSimulator(cfg=cfg,
                                                prefix="sharing",
                                                scene_name=cfg.scene_path + "simple_OSM_scene.xml",
                                                carrier_frequency=cfg.sharing_carrier_freq,
                                                pmax=50, # maximum power
                                                transmitters=transmitters,
                                                num_rx = len(slice), # changed further on
                                                max_depth=cfg.max_depth,
                                                cell_size=cfg.cell_size,
                                                initial_state = tf.convert_to_tensor([True for _ in range(num_tx)], dtype=tf.bool),
                                                subcarrier_spacing = cfg.sharing_subcarrier_spacing,
                                                fft_size = cfg.sharing_fft_size,
                                                batch_size= cfg.batch_size,
                                                )
                    sharingBand.grid = grid # giving it the primary grid for consistency - unsure if necessary
                sharingBand.receivers = None # clearing the previous receivers forces an update

                users_slice = dict(slice)
                output = sharingBand(users_slice, state, transmitters, mcs=None) # not using rate, calculate in the loop
                logger.info("Completed slice.")

                for tx_id in range(num_tx):
                    for user, sinr, bler in zip(users_slice.values(), output["sinr"][tx_id].numpy(), output["bler"][tx_id].numpy()):
                        y,x = user["position"][0], user["position"][1]
                        sinrs[id,tx_id,y,x] = sinr
                        blers[id,tx_id,y,x] = bler
                        bers[id,tx_id,y,x] = ber

                # counter = 0
                # for tx_id in range(num_tx):
                #     if bool(state[tx_id].numpy()) is True:
                #         print("Working")
                #         sinrs[id,tx_id,:,:] = tf.clip_by_value(10 * tf.math.log(sharingBand.sinr[counter]) / tf.math.log(10.0), -1000, 1000).numpy()
                #         counter += 1
                #     plot_coverage_map(sinrs[id,tx_id,:,:], cfg.images_path, title=f"Sinrs map for Transmitter {tx_id}, State {id}")
                    # cmap = tf.clip_by_value(10 * tf.math.log(sharingBand.sinr[tx_id]) / tf.math.log(10.0), -1000, 1000).numpy()
                    # plot_coverage_map(cmap, cfg.images_path, title=f"Cmap for Transmitter {tx_id}, State {id}") # will get error with tx_id index above

                # for tx_id in range(num_tx):
                #     # cmap = tf.clip_by_value(10 * tf.math.log(sharingBand.sinr[tx_id]) / tf.math.log(10.0), -1000, 1000).numpy()
                #     # plot_coverage_map(cmap, cfg.images_path, title=f"Cmap for Transmitter {tx_id}, State {id}") # will get error with tx_id index above

                end = perf_counter()
                time = end - start
                logger.info(f"Slice took : {time}")

            sharing_maps_temp = np.stack([sinrs, blers, bers], axis=0).astype(np.float16) # save as float 16 ()
            np.save(cfg.maps_path + f"temp {id}", sharing_maps_temp)

            tf.keras.backend.clear_session()
            if hasattr(sharingBand, 'simulator') and hasattr(sharingBand.simulator, 'h_freq'):
                sharingBand.simulator.h_freq = None
            del sharingBand
            gc.collect()

        logger.info(f"Saving arrays of shapes:\n\tSINR: {sinrs.shape}\n\tBLER: {blers.shape}\n\tBER: {bers.shape}")  
        logger.warning(f"Saving primary maps at {cfg.maps_path + sharing_maps_filename}.")
        sharing_maps = np.stack([sinrs, blers, bers], axis=0).astype(np.float16) # save as float 16 ()
        np.save(cfg.maps_path + sharing_maps_filename, sharing_maps)
    
    else:
        logger.warning("Skipping sharing maps computation. File already exists.")

    return


def print_gpu_memory_stats():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'])
    memory_used, memory_free = map(int, result.decode('utf-8').strip().split(','))
    logger.info(f"GPU Memory: {memory_used}MB used, {memory_free}MB free")


def generate_users(grid, users):
    """ Return a user for every valid position in the grid. """
    if users is None:
        pass

    y_max = grid.shape[0]
    x_max = grid.shape[1]

    if users == {}:
        for y in range(y_max):
            for x in range(x_max):
                if grid[y, x]:
                    users[f"({y},{x})"] = {
                                            "position": tf.concat([tf.convert_to_tensor([y,x]), tf.constant([1])], axis=0),
                                            "color": [1,0,1],
                                            "direction": tf.convert_to_tensor([0,0,0]),
                                        }
    else:
        raise Exception("Function not designed to be called repeatedly.")

    return users


def plot_coverage_map(data, save_path, title="Coverage Map", plot_min=-100, plot_max=100, cmap="jet"):
    """
    Saves a coverage map with a dB color scale from -100 to 100 to the Logging directory.
    
    Parameters:
        data (numpy.ndarray): 2D array representing the coverage map.
        title (str): Title of the plot.
    """
    # Mask invalid values (-1000)
    masked_data = np.ma.masked_equal(data, -1000)
    
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap(cmap)
    
    im = plt.imshow(masked_data, cmap=cmap, vmin=plot_min, vmax=plot_max, origin='lower')
    if plot_max > 1:
        plt.colorbar(im, label='dB')
    else:
         plt.colorbar(im)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    # Save the figure
    save_path = save_path + f"{title}.png"
    plt.savefig(save_path, dpi=400)
    plt.close()



if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f'Number of GPUs available : {len(gpus)}')
    if gpus:
        gpu_num = 0 # Index of the GPU to be used
        try:
            tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
            logger.warning(f'Only GPU number {gpu_num} used.')
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True) # manages memory growth
        except RuntimeError as e:
            logger.critical(e)
    with initialize(version_base=None, config_path="Config", job_name=CONFIG_NAME):
        config = compose(config_name=CONFIG_NAME)
        logger.info(f"Config:\n{config}\n")

    sionna.config.xla_compat=True # crucial to get the JIT compilation to work.

    main(config)