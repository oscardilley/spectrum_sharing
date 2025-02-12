""" utils.py

Key utility functions for running simuations. 

"""

import tensorflow as tf
import numpy as np

# def levy_step():
#     """Generate a step size following a Levy distribution, scaled for grid movement"""
#     u = np.random.normal(0, 1)
#     v = np.random.normal(0, 1)
#     step = u / np.abs(v)**(1/2)
#     step = 4.0 + (step * 2.0)  # Center around 3 with ±2 variation
#     return np.clip(step, 1.0, 7.0)

# def find_valid_position(grid, base_pos, max_radius=None):
#     """Find the nearest valid position to the given base position"""
#     y_max, x_max = grid.shape
#     base_pos = np.array(base_pos, dtype=np.int64)
    
#     if max_radius is None:
#         max_radius = max(y_max, x_max)
    
#     if (0 <= base_pos[0] < y_max and 0 <= base_pos[1] < x_max and grid[tuple(base_pos)]):
#         return base_pos
    
#     radius = 0
#     while radius < max_radius:
#         for layer in range(8):  # Check 8 directions
#             angle = layer * np.pi / 4  # 45-degree increments
#             dy = int(round(np.sin(angle) * radius))
#             dx = int(round(np.cos(angle) * radius))
#             pos = base_pos + np.array([dy, dx])
            
#             if (0 <= pos[0] < y_max and 0 <= pos[1] < x_max and grid[tuple(pos)]):
#                 return pos
#         radius += 1
    
#     valid_indices = np.argwhere(grid)
#     random_idx = np.random.randint(0, len(valid_indices))
#     return valid_indices[random_idx]

# def update_users(grid, num_users, users, max_move=7):
#     """
#     Update user positions with deterministic motion based on NumPy randomness.
#     """
#     y_max, x_max = grid.shape
#     valid_indices = np.argwhere(grid)
    
#     if not users:
#         if len(valid_indices) < num_users:
#             raise ValueError(f"Not enough valid positions ({len(valid_indices)}) for requested users ({num_users})")
        
#         num_clusters = max(1, num_users // 4)
#         cluster_centers = valid_indices[np.random.choice(len(valid_indices), num_clusters, replace=False)]
        
#         positions = []
#         for _ in range(num_users):
#             cluster_idx = np.random.randint(0, num_clusters)
#             base_pos = cluster_centers[cluster_idx].astype(float)
#             grid_scale = min(y_max, x_max) / 50
#             noise = np.random.normal(0, grid_scale, size=2)
#             pos = base_pos + noise
#             pos = np.clip(pos, [0, 0], [y_max-1, x_max-1]).astype(np.int64)
#             positions.append(pos)
        
#         for ue in range(num_users):
#             pos = find_valid_position(grid, positions[ue])
#             initial_direction = np.random.randint(-max_move, max_move+1, size=2)
#             users[f"ue{ue}"] = {
#                 "color": [1, 0, 1],
#                 "position": np.append(pos, 1),
#                 "direction": np.append(initial_direction, 0),
#                 "buffer": 100,
#                 "momentum": initial_direction.astype(float),
#                 "steps_in_direction": 0
#             }
#     else:
#         for ue in range(num_users):
#             users[f"ue{ue}"]["position"] += users[f"ue{ue}"]["direction"]
#             momentum = users[f"ue{ue}"]["momentum"]
#             direction = users[f"ue{ue}"]["direction"][:2].astype(float)
#             users[f"ue{ue}"]["momentum"] = 0.8 * momentum + 0.2 * direction
    
#     for ue in range(num_users):
#         start = users[f"ue{ue}"]["position"]
#         momentum = users[f"ue{ue}"]["momentum"]
#         steps_in_direction = users[f"ue{ue}"]["steps_in_direction"]
        
#         if steps_in_direction > 3 or np.random.rand() < 0.15:
#             step_size = levy_step()
#             angle = np.random.uniform(-np.pi, np.pi)
#             new_direction = np.array([np.cos(angle) * step_size, np.sin(angle) * step_size])
#             new_direction = 0.7 * new_direction + 0.3 * momentum
#             users[f"ue{ue}"]["steps_in_direction"] = 0
#         else:
#             noise = np.random.normal(0, 0.3, size=2)
#             new_direction = momentum + noise
#             users[f"ue{ue}"]["steps_in_direction"] += 1
        
#         move = np.clip(new_direction, -max_move, max_move).astype(np.int64)
#         valid_move = False
#         attempts = 0
#         original_move = move
        
#         while not valid_move and attempts < 20:
#             pos = start[:2] + move
#             at_edge = (pos[1] >= x_max-1 or pos[1] <= 0 or pos[0] >= y_max-1 or pos[0] <= 0)
            
#             if at_edge or not grid[tuple(pos)]:
#                 if at_edge:
#                     center = np.array([y_max/2, x_max/2])
#                     current = start[:2].astype(float)
#                     to_center = center - current
#                     to_center /= max(np.linalg.norm(to_center), 1e-6)
#                     random_angle = np.random.uniform(-np.pi/4, np.pi/4)
#                     rotation = np.array([[np.cos(random_angle), -np.sin(random_angle)],
#                                          [np.sin(random_angle), np.cos(random_angle)]])
#                     new_dir = np.dot(to_center, rotation) * max_move
#                     move = new_dir.astype(np.int64)
#                     original_move = move
#                 else:
#                     scale = 0.8 - (attempts * 0.02)
#                     move = (original_move.astype(float) * scale).astype(np.int64)
#                 attempts += 1
#             else:
#                 valid_move = True
        
#         if not valid_move:
#             new_pos = find_valid_position(grid, start[:2])
#             move = new_pos - start[:2]
        
#         users[f"ue{ue}"]["direction"] = np.append(move, 0)
    
#     return users


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

# def update_users(grid, num_users, users, max_move=7):
#     """
#     Update user positions with more pronounced grid movements.
#     Ensures valid spawning and maintains consistent movement.
#     """
#     y_max = grid.shape[0]
#     x_max = grid.shape[1]
#     valid_indices = tf.where(grid)
    
#     if users == {}:
#         # Initialize users with clustering
#         valid_count = tf.shape(valid_indices)[0]
#         if valid_count < num_users:
#             raise ValueError(f"Not enough valid positions ({valid_count}) for requested users ({num_users})")
        
#         # Create clusters
#         num_clusters = max(1, num_users // 4)  
#         random_cluster_ids = tf.random.uniform(shape=(num_clusters,), maxval=valid_count, dtype=tf.int32)
#         cluster_centers = tf.gather(valid_indices, random_cluster_ids)
        
#         # Initialize positions around cluster centers
#         positions = []
#         for ue in range(num_users):
#             # Assign to random cluster
#             cluster_idx = tf.random.uniform(shape=(), maxval=num_clusters, dtype=tf.int32)
#             base_pos = tf.cast(cluster_centers[cluster_idx], tf.float32)
            
#             # Add noise scaled by grid size for better spread
#             grid_scale = tf.cast(tf.minimum(y_max, x_max) / 50, tf.float32)  # Scale noise by grid size
#             noise = tf.random.normal(shape=(2,), mean=0.0, stddev=grid_scale)
#             pos = base_pos + noise
            
#             # Ensure within grid bounds
#             pos = tf.clip_by_value(pos, [0, 0], [y_max-1, x_max-1])
#             positions.append(tf.cast(pos, tf.int64))
        
#         for ue in range(num_users):
#             # Ensure position is valid
#             pos = find_valid_position(grid, positions[ue])
            
#             # Initialize with a random direction and momentum
#             initial_direction = tf.random.uniform(shape=(2,), minval=-1*max_move, maxval=max_move+1, dtype=tf.int64)
            
#             attributes = {
#                 "color": [1,0,1],
#                 "position": tf.concat([pos, tf.constant([1], dtype=tf.int64)], axis=0),
#                 "direction": tf.concat([initial_direction, tf.constant([0], dtype=tf.int64)], axis=0),
#                 "buffer": 100,
#                 "momentum": tf.cast(initial_direction, tf.float32),
#                 "steps_in_direction": 0
#             }
#             users[f"ue{ue}"] = attributes
    
#     else:
#         # Update existing users with momentum-based movement
#         for ue in range(num_users):
#             users[f"ue{ue}"]["position"] += users[f"ue{ue}"]["direction"]
            
#             # Update momentum with stronger persistence
#             momentum = users[f"ue{ue}"]["momentum"]
#             direction = tf.cast(users[f"ue{ue}"]["direction"][:2], tf.float32)
#             users[f"ue{ue}"]["momentum"] = 0.8 * momentum + 0.2 * direction
    
#     # Generate new directions with stronger movements
#     for ue in range(num_users):
#         start = users[f"ue{ue}"]["position"]
#         momentum = users[f"ue{ue}"]["momentum"]
#         steps_in_direction = users[f"ue{ue}"]["steps_in_direction"]
        
#         # Decide if we should change direction
#         if steps_in_direction > 3 or tf.random.uniform(()) < 0.15:  # 15% chance to change direction
#             # Use Levy flight for step size
#             step_size = levy_step()
            
#             # Generate new direction with momentum influence
#             angle = tf.random.uniform(shape=(), minval=-np.pi, maxval=np.pi)
#             new_direction = tf.stack([
#                 tf.cos(angle) * step_size,
#                 tf.sin(angle) * step_size
#             ])
            
#             # Blend with momentum for smoother transitions
#             new_direction = 0.7 * new_direction + 0.3 * momentum
#             users[f"ue{ue}"]["steps_in_direction"] = 0
#         else:
#             # Continue in current direction with slight variation
#             noise = tf.random.normal(shape=(2,), stddev=0.3)
#             new_direction = momentum + noise
#             users[f"ue{ue}"]["steps_in_direction"] += 1
        
#         # Scale the movement to ensure it's noticeable
#         move = tf.cast(tf.clip_by_value(new_direction, -max_move, max_move), tf.int64)
        
#         # Validate the move
#         valid_move = False
#         attempts = 0
#         original_move = move
#         while not valid_move and attempts < 20:
#             pos = start[0:2] + move
#             at_edge = (pos[1] >= x_max-1 or pos[1] <= 0 or 
#                       pos[0] >= y_max-1 or pos[0] <= 0)
            
#             if at_edge or not bool(tf.gather_nd(grid, pos)):
#                 if at_edge:
#                     # If at edge, generate new direction pointing inward
#                     center = tf.constant([y_max/2, x_max/2], dtype=tf.float32)
#                     current = tf.cast(start[0:2], tf.float32)
#                     to_center = center - current
#                     to_center = to_center / tf.maximum(tf.norm(to_center), 1e-6)
                    
#                     # Add some randomness to avoid straight lines to center
#                     random_angle = tf.random.uniform(shape=(), minval=-np.pi/4, maxval=np.pi/4)
#                     cos_theta = tf.cos(random_angle)
#                     sin_theta = tf.sin(random_angle)
#                     rotation = tf.stack([[cos_theta, -sin_theta], 
#                                       [sin_theta, cos_theta]])
                    
#                     new_dir = tf.matmul(tf.reshape(to_center, [1, 2]), rotation)
#                     new_dir = tf.squeeze(new_dir) * tf.cast(max_move, tf.float32)
#                     move = tf.cast(new_dir, tf.int64)
#                     original_move = move  # Update original move for scaling
#                 else:
#                     # If invalid but not at edge, try scaled version
#                     scale = 0.8 - (attempts * 0.02)
#                     move = tf.cast(tf.cast(original_move, tf.float32) * scale, tf.int64)
#                 attempts += 1
#             else:
#                 valid_move = True
        
#         if not valid_move:
#             # If all attempts failed, find a valid nearby position
#             current_pos = start[0:2]
#             new_pos = find_valid_position(grid, current_pos)
#             move = new_pos - current_pos
        
#         users[f"ue{ue}"]["direction"] = tf.concat([move, tf.constant([0], dtype=tf.int64)], axis=0)
    
#     return users


def get_throughput(rates):
    """ Calculate average link level throughput. """
    rates = rates / 1e6 # convert to MHz
    return tf.cast(tf.reduce_sum(rates), dtype=tf.float32), tf.cast(tf.reduce_sum(rates, axis=[0,1]), dtype=tf.float32), tf.cast(tf.reduce_sum(rates, axis=2), dtype=tf.float32)

def get_power_efficiency(primary_bw, sharing_bw, sharing_state, primary_power, sharing_power, mu_pa):
    """ Calculate average power efficiency in W/MHz which is later abstracted to energy efficiency. """
    primary_pe = (primary_power / mu_pa) / primary_bw
    sharing_pe = (tf.cast(sharing_state, tf.float32) * (sharing_power / mu_pa)) / sharing_bw
    combined_pe = (primary_pe + sharing_pe) * 1e6

    return tf.cast(tf.reduce_sum(combined_pe), dtype=tf.float32), tf.cast(combined_pe, dtype=tf.float32)

def get_spectral_efficiency(primary_bw, sharing_bw, per_ap_per_band_throughput):
    """ Calculate average spectral efficiency. """
    per_ap_per_band_throughput = per_ap_per_band_throughput * 1e6 # convert back to Hz from MHz
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
