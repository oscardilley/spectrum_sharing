""" utils.py

Key utility functions for running simuations. 

"""

import tensorflow as tf
import numpy as np

def update_users(grid, num_users, users, max_move=2):
    """
    Update the positions and directions of users on a grid.

    This function either initializes users with random valid positions on the grid
    or updates their positions based on their current direction. It also generates
    new valid movement directions for each user.

    Parameters
    ----------
    grid : tf.Tensor
        A 2D TensorFlow tensor representing the grid where users move.

    num_users : int
        The number of users to update or initialize.

    users : dict
        A dictionary containing user attributes. If empty, users are initialized.

    max_move : int, optional
        The maximum movement distance in any direction. Defaults to 2.

    Returns
    -------
    users : dict
        Updated dictionary of users with new positions and directions.
    """
    y_max = grid.shape[0]
    x_max = grid.shape[1]
    valid_indices = tf.where(grid)
    
    if users == {}:
        # Initialising users
        random_ids = tf.random.uniform(shape=(num_users,), maxval=tf.shape(valid_indices)[0], dtype=tf.int32)
        positions = tf.gather(valid_indices, random_ids) # random starting positions    
        for ue in range(num_users):
            attributes = {"color": [1,0,1], 
                          "position": tf.concat([positions[ue], tf.constant([1], dtype=tf.int64)], axis=0), 
                          "direction": None, 
                          "buffer": 100,}
            users[f"ue{ue}"] = attributes
    
    else:
        # Adding the previously calculated and verified distance vectors to move existing users
        for ue in range(num_users):
            users[f"ue{ue}"]["position"] +=  users[f"ue{ue}"]["direction"] 
        
    # Generating new valid direction values based on position
    for ue in range(num_users):
        start = users[f"ue{ue}"]["position"]
        valid_move = False
        while not valid_move:
            move = tf.random.uniform(shape=(2,), minval=-1*max_move, maxval=max_move+1, dtype=tf.int64)
            pos = start[0:2] + move
            if pos[1] >= x_max or pos[1] < 0:
                continue
            elif pos[0] >= y_max or pos[0] < 0:
                continue
            elif not bool(tf.gather_nd(grid, pos)):
                continue
            else:
                valid_move=True
        users[f"ue{ue}"]["direction"] = tf.concat([move, tf.constant([0], dtype=tf.int64)], axis=0) 

    return users


def levy_step():
    """
    Generate a step size following a Levy distribution, scaled for grid movement.

    This function generates a random step size based on the Levy distribution,
    which is then scaled to represent movement on a grid.

    Returns
    -------
    step : tf.Tensor
        A scalar TensorFlow tensor representing the step size, clipped between 1.0 and 7.0.
    """
    u = tf.random.normal(shape=(), mean=0, stddev=1)
    v = tf.random.normal(shape=(), mean=0, stddev=1)
    step = u / tf.abs(v)**(1/2)
    # Scale to get movements between 1-5 grid spaces
    step = 4.0 + (step * 2.0)  # Center around 3 with ±2 variation
    return tf.clip_by_value(step, 1.0, 7.0)

def find_valid_position(grid, base_pos, max_radius=None):
    """
    Find the nearest valid position to the given base position on the grid.

    This function searches for a valid position on the grid starting from the
    given base position. If no valid position is found within the specified
    radius, a random valid position is returned.

    Parameters
    ----------
    grid : tf.Tensor
        A 2D TensorFlow tensor representing the grid where positions are checked.

    base_pos : tf.Tensor
        A 1D TensorFlow tensor of shape (2,) representing the base position.

    max_radius : int, optional
        The maximum radius to search for a valid position. Defaults to the entire grid if None.

    Returns
    -------
    pos : tf.Tensor
        A 1D TensorFlow tensor of shape (2,) representing the nearest valid position.
    """
    y_max, x_max = grid.shape
    radius = 0
    base_pos = tf.cast(base_pos, tf.int64)
    
    if max_radius is None:
        max_radius = max(y_max, x_max)
    
    # First check if base_pos itself is valid
    if (base_pos[0] >= 0 and base_pos[0] < y_max and 
        base_pos[1] >= 0 and base_pos[1] < x_max and 
        bool(tf.gather_nd(grid, base_pos))):
        return base_pos
    
    # Spiral outward to find valid position
    while radius < max_radius:
        # Check positions in a spiral pattern
        for layer in range(8):  # Check 8 directions
            angle = (layer * np.pi / 4)  # 45-degree increments
            dy = tf.cast(tf.round(tf.sin(angle) * radius), tf.int64)
            dx = tf.cast(tf.round(tf.cos(angle) * radius), tf.int64)
            pos = base_pos + tf.stack([dy, dx])
            
            if (pos[0] >= 0 and pos[0] < y_max and 
                pos[1] >= 0 and pos[1] < x_max and 
                bool(tf.gather_nd(grid, pos))):
                return pos
        radius += 1
    
    # If no valid position found within radius, return a random valid position
    valid_indices = tf.where(grid)
    random_idx = tf.random.uniform(shape=(), maxval=tf.shape(valid_indices)[0], dtype=tf.int32)

    return tf.gather(valid_indices, random_idx)

def get_throughput(rates):
    """
    Calculate average link-level throughput.

    This function computes the total, per-band, and per-user throughput
    from the given data rates.

    Parameters
    ----------
    rates : tf.Tensor
        A 3D TensorFlow tensor representing data rates in bits per second over [Transmitters, Bands, Users]

    Returns
    -------
    total_throughput : tf.float32
        Total aggregated throughput in Mbps.

    per_ue_throughput : tf.Tensor of tf.float32
        Per-UE throughput in Mbps. 1D of size [Users]

    per_ap_per_band_throughput : tf.Tensor of tf.float32
        Per-ap per-band throughput in Mbps. 2D of [Transmitters, Bands]
    """
    rates = rates / 1e6 # convert to Mbps
    return tf.cast(tf.reduce_sum(rates), dtype=tf.float32), tf.cast(tf.reduce_sum(rates, axis=[0,1]), dtype=tf.float32), tf.cast(tf.reduce_sum(rates, axis=2), dtype=tf.float32)

def get_power_efficiency(primary_bw, sharing_bw, sharing_state, primary_power, sharing_power, mu_pa):
    """
    Calculate average power efficiency in W/MHz.

    This function computes the power efficiency for both primary and sharing
    bands, as well as their combined efficiency.

    Parameters
    ----------
    primary_bw : float
        Bandwidth of the primary network in Hz.

    sharing_bw : float
        Bandwidth of the sharing network in Hz.

    sharing_state : tf.Tensor
        A tensor indicating the permitted sharing band state (ON/OFF) for 1D of size [Access points].

    primary_power : tf.Tensor of tf.float32
        1D tensor of the primary transmission powers for the access points in W.

    sharing_power : tf.Tensor of tf.float32
        1D tensor of the sharing transmission powers for the access points in W.

    mu_pa : tf.Tensor of tf.float32
        Power amplifier efficiency, 1D, for each access point.

    Returns
    -------
    total_power_efficiency : tf.float32
        Total power efficiency in W/MHz.

    per_ap_power_efficiency : tf.Tensor of tf.float32
        Per-access point power efficiency in W/MHz over sharing and primary bands.
    """
    primary_pe = (primary_power / mu_pa) / primary_bw
    sharing_pe = (tf.cast(sharing_state, tf.float32) * (sharing_power / mu_pa)) / sharing_bw
    combined_pe = (primary_pe + sharing_pe)

    return tf.cast(tf.reduce_sum(combined_pe), dtype=tf.float32), tf.cast(combined_pe, dtype=tf.float32)

def get_spectral_efficiency(primary_bw, sharing_bw, per_ap_per_band_throughput):
    """
    Calculate average spectral efficiency.

    This function computes the spectral efficiency for both primary and sharing
    networks, as well as their combined efficiency.

    Parameters
    ----------
    primary_bw : float
        Bandwidth of the primary network in Hz.

    sharing_bw : float
        Bandwidth of the sharing network in Hz.

    per_ap_per_band_throughput : tf.Tensor
        A tensor of throughput values per access point and band. 2D of [Transmitters, Bands]

    Returns
    -------
    avg_spectral_efficiency : tf.float32
        Average spectral efficiency.

    per_ap_spectral_efficiency : tf.Tensor of tf.float32
        Per access point spectral efficiency. 1D of [Transmitters]
    """
    # Convert throughput from Mbps to bps
    per_ap_per_band_se = per_ap_per_band_throughput * 1e6  

    # Get number of bands dynamically
    num_bands = tf.shape(per_ap_per_band_se)[1]

    # Compute spectral efficiency using proper tensor operations
    if num_bands > 1:
        primary_se = per_ap_per_band_se[:, :-1] / primary_bw  # Primary bands
        sharing_se = per_ap_per_band_se[:, -1:] / sharing_bw   # Sharing band (keep it 2D)
        per_ap_per_band_se = tf.concat([primary_se, sharing_se], axis=1)
    else:
        per_ap_per_band_se = per_ap_per_band_se / sharing_bw  # If there's only one band, it's a sharing band

    # Sum spectral efficiency over bands
    per_ap_se = tf.reduce_sum(per_ap_per_band_se, axis=1)

    # Return average and per AP spectral efficiency
    return tf.cast(tf.reduce_mean(per_ap_se), dtype=tf.float32), tf.cast(per_ap_se, dtype=tf.float32)


def get_spectrum_utility(primary_bw, sharing_bw, sharing_state, total_throughput):
    """
    Calculate how sensibly the spectrum is used.

    This function computes the spectrum utility, which represents the
    efficiency of spectrum usage by combining primary and sharing networks. It encourages opening 
    spectrum for sharing when it is not needed.

    Parameters
    ----------
    primary_bw : float
        Bandwidth of the primary network in Hz.

    sharing_bw : float
        Bandwidth of the sharing network in Hz.

    sharing_state : tf.Tensor
        A tensor indicating the permitted sharing band state (ON/OFF) for 1D of size [Access points].

    total_throughput : float
        Total throughput in bits per second.

    Returns
    -------
    spectrum_utility : tf.float32
        Spectrum utility as a float.
    """
    num_bs = tf.cast(sharing_state.shape[0], dtype=tf.float32)
    total_primary_spectrum = tf.reduce_sum(num_bs * primary_bw)
    total_sharing_spectrum = tf.reduce_sum(tf.cast(sharing_state, tf.float32) * sharing_bw)

    return  tf.cast(total_throughput * 1e6, dtype=tf.float32) / (total_primary_spectrum + total_sharing_spectrum)


def get_power_efficiency_bounds(primary_bw, sharing_bw, primary_power, mu_pa, min_sharing_power, max_sharing_power):
    """
    Calculate per-access point and total minimum and maximum power efficiencies in W/MHz.

    Parameters
    ----------
    primary_bw : float
        Bandwidth of the primary network in Hz.
    sharing_bw : float
        Bandwidth of the sharing network in Hz.
    primary_power : tf.Tensor
        Tensor of primary power values (fixed) for each access point. Shape: [num_APs].
    mu_pa : tf.Tensor
        Tensor of power amplifier efficiencies for each access point. Shape: [num_APs].
    min_sharing_power : tf.Tensor
        Tensor of minimum sharing power values for each access point. Shape: [num_APs].
    max_sharing_power : tf.Tensor
        Tensor of maximum sharing power values for each access point. Shape: [num_APs].

    Returns
    -------
    total_eff_min : tf.float32
        Total minimum efficiency (sum over access points) in W/MHz.
    total_eff_max : tf.float32
        Total maximum efficiency (sum over access points) in W/MHz.
    per_ap_eff_min : tf.Tensor
        Per-access point minimum efficiency in W/MHz.
    per_ap_eff_max : tf.Tensor
        Per-access point maximum efficiency in W/MHz.
    """
    # Efficiency when using the primary network only (sharing off)
    primary_eff = primary_power / (mu_pa * primary_bw)

    # Sharing network efficiency contribution per access point for min and max cases.
    sharing_eff_min = min_sharing_power / (mu_pa * sharing_bw)
    sharing_eff_max = max_sharing_power / (mu_pa * sharing_bw)

    # When sharing is turned on, total efficiency is the sum of primary and sharing components.
    eff_with_sharing_min = primary_eff + sharing_eff_min
    eff_with_sharing_max = primary_eff + sharing_eff_max

    # For each AP, the minimum possible efficiency is the lower of having sharing off or on (with min sharing power)
    per_ap_eff_min = tf.minimum(primary_eff, eff_with_sharing_min)
    
    # Similarly, the maximum possible efficiency is the higher of having sharing off or on (with max sharing power)
    per_ap_eff_max = tf.maximum(primary_eff, eff_with_sharing_max)

    # Total efficiency is the sum over all access points.
    total_eff_min = tf.reduce_sum(per_ap_eff_min)
    total_eff_max = tf.reduce_sum(per_ap_eff_max)

    return total_eff_min, total_eff_max, per_ap_eff_min, per_ap_eff_max

def get_fairness(per_ue_throughput):
    """
    Calculating the Jain's fairness index across the UE throughput values.

    Parameters
    ----------
    per_ue_throughput : tf.Tensor of tf.float32
        1D tensor of the UE throughput values achieved - this should strive to be equal.

    Returns
    -------
    fairness : float
        JFI value, automatically normalised in the range [0,1].
    """
    n = len(per_ue_throughput) # number of users
    numerator = tf.math.square(tf.reduce_sum(per_ue_throughput))
    denominator = n * tf.reduce_sum(tf.math.square(per_ue_throughput))
    fairness = numerator / denominator

    return fairness