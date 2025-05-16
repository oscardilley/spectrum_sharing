"""" validate_preprocess.py 

Testing a coverage map npy file by plotting all of the layers with matplotlib. Must be run as a module. """

from hydra import compose, initialize 
import numpy as np
import matplotlib.pyplot as plt 

CONFIG_NAME = "preprocessing" # the only config selection in the script

def main(cfg):
    """
    Saving maps of the SINR and BLER for the primary and sharing bands.
    """
    # Configuration to change
    TEST_FILE = cfg.assets_path + f"primary_maps.npy" # edit this
    primary=True # True for shape [SINR/BLER/BER, tx_id, y, x] and False for shape [SINR/BLER/BER, i, tx_id, y, x]

    map = np.load(TEST_FILE)
    if primary:
        for tx_id in range(len(cfg.transmitters)):
            plot_coverage_map(np.clip(map[0,tx_id,:,:],-100,100), cfg.assets_path + "Maps/" + f"Primary SINR test tx {tx_id}") # sinr
            plot_coverage_map(map[1,tx_id,:,:], cfg.assets_path + "Maps/" + f"Primary BLER test tx {tx_id}",plot_min=0, plot_max=1) # bler
            plot_coverage_map(map[2,tx_id,:,:], cfg.assets_path + "Maps/" + f"Primary BER test tx {tx_id}",plot_min=0, plot_max=1) # ber
        return
    else:
        for i in range(map.shape[1]):
            for tx_id in range(len(cfg.transmitters)):
                plot_coverage_map(np.clip(map[0,i,tx_id,:,:],-100,100), cfg.assets_path + "Maps/" + f"SINR test {i} tx {tx_id}") # sinr
                plot_coverage_map(map[1,i,tx_id,:,:], cfg.assets_path + "Maps/" + f"BLER test {i} tx {tx_id}",plot_min=0, plot_max=1) # bler
                plot_coverage_map(map[2,i,tx_id,:,:], cfg.assets_path + "Maps/" + f"BER test {i} tx {tx_id}",plot_min=0, plot_max=1) # ber
        return


def plot_coverage_map(data, save_path, title="Coverage Map", plot_min=-100, plot_max=100):
    """
    Saves a coverage map with a dB color scale from -100 to 100 to the Logging directory.
    
    Parameters:
        data (numpy.ndarray): 2D array representing the coverage map.
        title (str): Title of the plot.
    """
    # Mask invalid values (-1000)
    masked_data = np.ma.masked_equal(data, -1000)
    
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("jet")
    
    im = plt.imshow(masked_data, cmap=cmap, vmin=plot_min, vmax=plot_max, origin='lower')
    plt.colorbar(im, label='dB')
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    # Save the figure
    save_path = save_path + f"{title}.png"
    plt.savefig(save_path, dpi=400)
    plt.close()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="Config", job_name=CONFIG_NAME):
        config = compose(config_name=CONFIG_NAME)
    main(config)