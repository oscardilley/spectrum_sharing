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

def get_throughput():
    """ Calculate average link level throughput. """

    return

def get_spectral_efficiency():
    """ Calculate average spectral efficiency. """

    return

def get_energy_efficiency():
    """ Calculate average energy efficiency. """

    return