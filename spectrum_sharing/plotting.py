""" plotting.py 

Key plotting functions for the simulator.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np

# def prop_fair_plotter(timestep, 
#                       tx,
#                       grid_alloc, 
#                       num_users,
#                       save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
#     """
#     Plot the resource allocation grid with a color for each user.
    
#     Parameters:
#       grid_alloc: 2D numpy array with user IDs allocated for each RB.
#       num_users: total number of users (used for the color mapping).
#     """
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(18, 9))
    
#     # Use a colormap with at least 20 distinct colours; 'tab20' supports 20 colours.
#     cmap = plt.get_cmap('tab20', num_users)
#     cax = ax.imshow(grid_alloc, aspect='auto', cmap=cmap)
    
#     # Add a colorbar and label the ticks with user IDs.
#     cbar = fig.colorbar(cax, ticks=range(num_users))
#     cbar.ax.set_yticklabels([f"User {i}" for i in range(num_users)])
    
#     ax.set_xlabel('Resource Block Index')
#     ax.set_ylabel('Time Slot Index')
#     ax.set_title(f'Resource Block Allocation over 1 Second, for TX {tx}, time {timestep}')
#     fig.savefig(save_path + f"Scheduler for TX {tx}, time {timestep}.png", dpi=600)
#     plt.close()

def prop_fair_plotter(timestep, tx, grid_alloc, num_users, user_rates, max_data_sent_per_rb, 
                        save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """
    Plot the resource allocation grid, a dual-axis bar chart (RB allocation and throughput),
    and the RB allocation time series.
    
    Parameters:
      grid_alloc: 2D numpy array with user IDs allocated for each RB.
      num_users: total number of users.
      user_rates: 1D np.ndarray with achieved throughput (bps) for each user.
      max_data_sent_per_rb: Maximum bits per RB if BLER were zero.
    """
    # Compute per-user statistics
    time_slots = grid_alloc.shape[0]
    rb_per_user_per_timestep = np.array([(grid_alloc == i).sum(axis=1) for i in range(num_users)])
    total_rbs_per_user = rb_per_user_per_timestep.sum(axis=1)
    
    # Create figure with GridSpec:
    # Top row: two columns; Bottom row: single subplot spanning both columns.
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    cmap = plt.get_cmap('tab20', num_users)
    
    # ---------------------------
    # Top-Left: Resource Allocation Grid
    # ---------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    cax = ax1.imshow(grid_alloc.T, aspect='auto', cmap=cmap, origin='lower')
    cbar = fig.colorbar(cax, ax=ax1, ticks=range(num_users))
    cbar.ax.set_yticklabels([f"User {i}" for i in range(num_users)])
    ax1.set_xlabel('Time Slot Index', fontsize=12)
    ax1.set_ylabel('Resource Block Index', fontsize=12)
    ax1.set_title(f'Resource Block Allocation Grid (TX {tx}, Time {timestep})', fontsize=14)
    plt.setp(ax1.get_xticklabels(), fontsize=10)
    
    # ---------------------------
    # Top-Right: Dual-Axis Bar Chart for Total RB Allocation & Throughput
    # ---------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.45
    x = np.arange(num_users)
    
    # Left axis: Total RB allocation bars
    bars_rb = ax2.bar(x - width/2, total_rbs_per_user, width=width, 
                      label='Total RBs Allocated',
                      color=[cmap(i) for i in range(num_users)])
    ax2.set_xlabel('User ID', fontsize=12)
    ax2.set_ylabel('Total RBs Allocated', color='black', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'User {i}' for i in range(num_users)], fontsize=8)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Right axis: Throughput bars in Mbps (actual values, not normalized)
    ax2_twin = ax2.twinx()
    # Convert throughput from bps to Mbps
    throughput_mbps = user_rates / 1e6
    bars_tp = ax2_twin.bar(x + width/2, throughput_mbps, width=width, 
                           label='Throughput (Mbps)',
                           color=[cmap(i) for i in range(num_users)], hatch='//', alpha=0.7)
    ax2_twin.set_ylabel('Throughput (Mbps)', color='black', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='black')
    
    # Manually create a combined legend
    # Use the first bar from each set to represent that series
    ax2.legend([bars_rb[0], bars_tp[0]], ['Total RBs Allocated', 'Throughput (Mbps)'], loc='upper right', fontsize=10) # Change to fix legend
    ax2.set_title('Total RB Allocation & Throughput per User', fontsize=14)
    plt.setp(ax2.get_xticklabels(), fontsize=10)
    
    # ---------------------------
    # Bottom: RB Allocation Time Series (spanning full width)
    # ---------------------------
    ax3 = fig.add_subplot(gs[1, :])
    for i in range(num_users):
        ax3.plot(range(time_slots), rb_per_user_per_timestep[i], label=f'User {i}', color=cmap(i))
    ax3.set_xlabel('Time Slot Index', fontsize=12)
    ax3.set_ylabel('RBs Allocated', fontsize=12)
    ax3.set_title('RB Allocation per User Over Time', fontsize=14)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylim(0, max(rb_per_user_per_timestep.max(), 1))
    ax3.set_xlim(0, time_slots)  # x-axis from 0 to number of time slots
    plt.setp(ax3.get_xticklabels(), fontsize=10)
    
    # Adjust layout so all subplots have equal spacing
    plt.tight_layout()
    fig.savefig(save_path + f"Scheduler_TX_{tx}_Time_{timestep}.png", dpi=600)
    plt.close()


def plot_total_rewards(episode,
                       reward,
                       reward_min,
                       reward_max,
                       throughput,
                       se,
                       pe,
                       su,
                       save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """ Plot reward functions over time."""
    # Axis initialisation
    labels = ["Normalised Average Reward", "Normalised Reward Min", "Normalised Reward Max", "Normalised Total Throughput", "Normalised Spectral Efficiency", "Normalised Power Efficiency", "Normalised Spectrum Utility"]
    fig, ax = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    cmap = plt.get_cmap("tab10", len(labels) - 1) # plot the metrics in consistent colours and the total in black

    upper_x = episode + 1
    x = np.linspace(0, episode, upper_x)

    # Plotting reward min/max
    ax.plot(x, reward[0:episode+1], linewidth=2, linestyle="solid", color="black", alpha=0.95)
    ax.plot(x, reward_min[0:episode+1], linewidth=2, linestyle="dashed", color="black", alpha=0.5)
    ax.plot(x, reward_max[0:episode+1], linewidth=2, linestyle="dashed", color="black", alpha=0.5)

    # Plotting specific normalised components
    ax.plot(x, throughput[:episode+1], linewidth=2, linestyle="solid", color=cmap(0), alpha=0.8)
    ax.plot(x, se[:episode+1], linewidth=2, linestyle="solid", color=cmap(1), alpha=0.8)
    ax.plot(x, pe[:episode+1], linewidth=2, linestyle="solid", color=cmap(2), alpha=0.8)
    ax.plot(x, su[:episode+1], linewidth=2, linestyle="solid", color=cmap(3), alpha=0.8)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Normalised Value", fontsize=12)
    ax.set_xlim([0, episode])
    ax.set_title("Average Performance between Episodes", fontsize=20)
    ax.legend(labels=labels, fontsize=8)

    fig.savefig(save_path + f"Rewards Tracker.png", dpi=400)#, bbox_inches="tight")
    plt.close()

    return 

def plot_rewards(episode,
                 step,
                 rewards,
                 save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """ Plot reward functions over time."""
    # Axis initialisation
    reward_labels = ["Total Throughput [MHz]", "Spectral Efficiency [bits/s/Hz]", "Power Efficiency [W/Hz]", "Spectrum Utility [bits/s/Hz]"]
    reward_titles = ["Total Throughput", "Spectral Efficiency", "Power Efficiency", "Spectrum Utility"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    cmap = plt.get_cmap("tab10", rewards.shape[1])

    upper_x = step + 1
    x = np.linspace(0, step, upper_x)

    for i, ax in enumerate(fig.axes):
        ax.plot(x, rewards[:upper_x,i], linewidth=2, linestyle="solid", color=cmap(i), alpha=0.8)
        ax.set_xlim([0, step])
        ax.set_title(reward_titles[i], fontsize=16)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(reward_labels[i], fontsize=12)

    fig.savefig(save_path + f"Rewards Ep{episode}.png", dpi=400)#, bbox_inches="tight")
    # fig.savefig(save_path + f"Rewards Ep{episode} Step{step}.png", dpi=400)#, bbox_inches="tight")
    plt.close()

    return 

def plot_motion(step, 
                id, 
                grid, 
                cm,
                color, 
                users, 
                transmitters, 
                cell_size, 
                sinr_range=[-100,100],
                fig=None, 
                ax=None, 
                save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """ Visualising user motion. """
    y_max = grid.shape[0]
    x_max = grid.shape[1]
    cmap = plt.get_cmap("tab20b", len(users))

    # Users
    x_positions = [ue["position"][1] for ue in users.values()]
    y_positions = [ue["position"][0] for ue in users.values()]
    dx = [ue["direction"][1] for ue in users.values()]
    dy = [ue["direction"][0] for ue in users.values()]

    # Transmitters - positions adjusted to cm coordinate system
    tx_x_positions = [transmitter["position"][0]/cell_size + (x_max/cell_size) for transmitter in transmitters.values()]
    tx_y_positions = [transmitter["position"][1]/cell_size + (y_max/cell_size) for transmitter in transmitters.values()]

    # Axis initialisation
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(int(x_max/4), int(y_max/4)), constrained_layout=True)
    
        # Add gridlines and lims
        ax.set_xticks(np.arange(x_max), minor=True)
        ax.set_yticks(np.arange(y_max), minor=True)
        ax.tick_params(which='major', size=25, labelsize=25)
        ax.tick_params(which='minor', size=0)  # Remove tick markers for minor grid lines
        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])

        ax.scatter(
                    tx_x_positions, tx_y_positions, s=1000, c="r", marker="*"
                  )

    else:
        [arrow.remove() for arrow in ax.findobj(match=mpl.quiver.Quiver)]
        [img.remove() for img in ax.images]
        if step > 5:
            [line.remove() for line in ax.lines[:min(10,len(ax.lines))]]

    # Plotting the coverage map
    cm_db = 10 * tf.math.log(cm) / tf.math.log(10.0)
    map = ax.imshow(cm_db, cmap=color, origin='lower', alpha=0.7, extent=[0, x_max, 0, y_max], vmin=sinr_range[0], vmax=sinr_range[1])
    if step == 0:
        cbar = fig.colorbar(map, ax=ax, shrink=0.8)
        cbar.set_label("SINR [dB]", fontsize=40) 
        cbar.ax.tick_params(labelsize=30)
    grid = grid.numpy()
    ax.imshow(np.ma.masked_where(grid > 0, grid), cmap='Set2', origin='lower', alpha=0.6, extent=[0, x_max, 0, y_max])

    # Plot motion vectors
    ax.quiver(
        x_positions, y_positions,  # Start positions of the arrows
        dx, dy,  # Vector components
        angles='xy', scale_units='xy', scale=2.5, label='Direction'
    )

    # Labels and title
    ax.set_title(f"\n{id} Step {step}\n", fontsize=70)
    ax.set_xlabel("X [Cells]", fontsize=40)
    ax.set_ylabel("Y [Cells]", fontsize=40)
    ax.legend(fontsize=30)
    
    # fig.savefig(save_path + f"Scene {id} Step {step}.png")
    if id == "Sharing Band, Max SINR":
        fig.savefig(save_path + f"Scene {id} Step {step}.png")
    else:
        fig.savefig(save_path + f"Scene {id}.png")#, bbox_inches="tight")

    for i in range(len(users)):
        ax.plot([x_positions[i], x_positions[i] + dx[i]], [y_positions[i], y_positions[i] + dy[i]], color=cmap(i), linewidth=4)

    return fig, ax


def plot_performance(step,
                     users, 
                     performance,
                     save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """ Plotting the BLER and SNR for the users. """
    length = len(performance)
    x = np.linspace(step + 1 - length, step, length)

    # Extracting data
    primary_bler = tf.stack([results["Primary"]["bler"] for results in performance])
    sharing_bler = tf.stack([results["Sharing"]["bler"] for results in performance])
    primary_sinr = tf.stack([results["Primary"]["sinr"] for results in performance])
    sharing_sinr = tf.stack([results["Sharing"]["sinr"] for results in performance])

    # Plot configuration
    num_ues = len(users)
    num_columns = int(np.sqrt(num_ues)) + 1
    cmap = plt.get_cmap("tab20b", num_ues)
    if (num_columns * (num_columns - 1)) >= num_ues:
        num_rows = num_columns - 1
    else: 
        num_rows = num_columns
    fig, ax = plt.subplots(num_rows, num_columns, figsize=(4.5*num_columns, 3*num_rows), constrained_layout=True)
    fig.suptitle(f"Timestep {step} UE Performance", fontsize=25, fontdict={'color': "black", 'fontweight': 'bold'})

    # Plotting per user
    ue = 0
    for r in range(num_rows):
        for c in range(num_columns):
            ax1 = ax[r,c]
            if ue >= num_ues:
                ax1.axis("off")
                continue
            x_pos = round(float(users[f"ue{ue}"]["position"][1]), 2)
            y_pos = round(float(users[f"ue{ue}"]["position"][0]), 2)
            ax1.set_title(f"UE {ue} at ({x_pos},{y_pos})", fontdict={'color': cmap(ue), 'weight': 'bold', 'fontsize': 16})
            ax1.set_xlabel("Step")
            ax1.set_ylabel("BLER")
            ax1.set_ylim([-0.05, 1.05])
            ax1.set_xlim([x[0], x[-1]])
            ax1.set_xticks(x, x)
            ax2 = ax1.twinx()
            ax2.set_ylabel("SINR [dB]")
            ax2.set_ylim([-80, 100])
            blerP0, = ax1.plot(x, primary_bler[:,0,ue], linestyle="dashed", color="orangered", alpha=0.8)
            blerP1, = ax1.plot(x, primary_bler[:,1,ue], linestyle="dashed", color="darkred", alpha=0.8)
            sinrP0, = ax2.plot(x, primary_sinr[:,0,ue], linestyle="solid", color="orangered", alpha=0.8)
            sinrP1, = ax2.plot(x, primary_sinr[:,1,ue], linestyle="solid", color="darkred", alpha=0.8)
            blerS0, = ax1.plot(x, sharing_bler[:,0,ue], linestyle="dashed", color="steelblue", alpha=0.8)
            blerS1, = ax1.plot(x, sharing_bler[:,1,ue], linestyle="dashed", color="darkturquoise", alpha=0.8)
            sinrS0, = ax2.plot(x, sharing_sinr[:,0,ue], linestyle="solid", color="steelblue", alpha=0.8)
            sinrS1, = ax2.plot(x, sharing_sinr[:,1,ue], linestyle="solid", color="darkturquoise", alpha=0.8)
            ue += 1

    fig.legend((blerP0, blerP1, sinrP0, sinrP1, blerS0, blerS1, sinrS0, sinrS1),
               ("Primary BLER TX0", "Primary BLER TX1", "Primary SINR TX0", "Primary SINR TX1", "Secondary BLER TX0", "Secondary BLER TX1", "Secondary SINR TX0", "Secondary SINR TX1"),
               loc="outside right center")
    fig.savefig(save_path + f"UE_performance.png", dpi=400)
    plt.close()

    return