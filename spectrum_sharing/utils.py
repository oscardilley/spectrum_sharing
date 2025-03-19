""" utils.py

Key utility functions for running simuations. 

"""

import tensorflow as tf
import numpy as np

def update_users(grid, num_users, users, max_move=2):
    """ Based on tensorflow validity matrix, either generate random users or update existing"""
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
    """Generate a step size following a Levy distribution, scaled for grid movement"""
    u = tf.random.normal(shape=(), mean=0, stddev=1)
    v = tf.random.normal(shape=(), mean=0, stddev=1)
    step = u / tf.abs(v)**(1/2)
    # Scale to get movements between 1-5 grid spaces
    step = 4.0 + (step * 2.0)  # Center around 3 with ±2 variation
    return tf.clip_by_value(step, 1.0, 7.0)

def find_valid_position(grid, base_pos, max_radius=None):
    """Find the nearest valid position to the given base position"""
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
    """ Calculate average link level throughput. """
    rates = rates / 1e6 # convert to Mbps
    return tf.cast(tf.reduce_sum(rates), dtype=tf.float32), tf.cast(tf.reduce_sum(rates, axis=[0,1]), dtype=tf.float32), tf.cast(tf.reduce_sum(rates, axis=2), dtype=tf.float32)

def get_power_efficiency(primary_bw, sharing_bw, sharing_state, primary_power, sharing_power, mu_pa):
    """ Calculate average power efficiency in W/MHz which is later abstracted to energy efficiency. 
    Bandwidths provided in Hz, powers in W. Aiming to minimise this value."""
    primary_pe = (primary_power / mu_pa) / primary_bw
    sharing_pe = (tf.cast(sharing_state, tf.float32) * (sharing_power / mu_pa)) / sharing_bw
    combined_pe = (primary_pe + sharing_pe)

    return tf.cast(tf.reduce_sum(combined_pe), dtype=tf.float32), tf.cast(combined_pe, dtype=tf.float32)

def get_spectral_efficiency(primary_bw, sharing_bw, per_ap_per_band_throughput):
    """ Calculate average spectral efficiency. """
    per_ap_per_band_throughput = per_ap_per_band_throughput * 1e6 # convert back to bps from Mbps
    primary_se = tf.reduce_sum(tf.stack([per_ap_per_band_throughput[bs,:] / primary_bw for bs in range(int(per_ap_per_band_throughput.shape[1]))]), axis=0) # for separated primary bands
    sharing_se = per_ap_per_band_throughput[-1,:] / sharing_bw # single sharing band - easier calculation
    combined = tf.stack([primary_se, sharing_se])

    return tf.cast(tf.reduce_mean(combined), dtype=tf.float32), tf.cast(combined, dtype=tf.float32)

def get_spectrum_utility(primary_bw, sharing_bw, sharing_state, total_throughput):
    """ Calculate how much of the spectrum is used. """
    num_bs = tf.cast(sharing_state.shape[0], dtype=tf.float32)
    total_primary_spectrum = tf.reduce_sum(num_bs * primary_bw)
    total_sharing_spectrum = tf.reduce_sum(tf.cast(sharing_state, tf.float32) * sharing_bw)

    return  tf.cast(total_throughput * 1e6, dtype=tf.float32) / (total_primary_spectrum + total_sharing_spectrum)
