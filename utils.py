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
            move = tf.random.uniform(shape=(2,), minval=-1*max_move, maxval=max_move, dtype=tf.int64)
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

def get_throughput(rates):
    """ Calculate average link level throughput. """

    return tf.reduce_sum(rates), tf.reduce_sum(rates, axis=[0,1]), tf.reduce_sum(rates, axis=2)

def get_power_efficiency(primary_bw, sharing_bw, sharing_state, primary_power, sharing_power, mu_pa):
    """ Calculate average power efficiency in W/Hz which is later abstracted to energy efficiency. """
    primary_pe = (primary_power / mu_pa) / primary_bw
    sharing_pe = (tf.cast(sharing_state, tf.float32) * (sharing_power / mu_pa)) / sharing_bw
    combined_pe = primary_pe + sharing_pe

    return tf.reduce_sum(combined_pe), combined_pe

def get_spectral_efficiency(primary_bw, sharing_bw, per_ap_per_band_throughput):
    """ Calculate average spectral efficiency. """
    primary_se = tf.reduce_sum(tf.stack([per_ap_per_band_throughput[bs,:] / primary_bw for bs in range(int(per_ap_per_band_throughput.shape[1]))]) ,axis=0) # for separated primary bands
    sharing_se = per_ap_per_band_throughput[2,:] / sharing_bw # single sharing band - easier calculation
    combined = tf.stack([primary_se, sharing_se])

    return tf.reduce_mean(combined), combined

def get_spectrum_utility(primary_bw, sharing_bw, sharing_state, total_throughput):
    """ Calculate how much of the spectrum is used. """
    num_bs = tf.cast(sharing_state.shape[0], dtype=tf.float32)
    total_primary_spectrum = tf.reduce_sum(num_bs * primary_bw)
    total_sharing_spectrum = tf.reduce_sum(tf.cast(sharing_state, tf.float32) * sharing_bw)

    return  tf.cast(total_throughput, dtype=tf.float32) /(total_primary_spectrum + total_sharing_spectrum)
