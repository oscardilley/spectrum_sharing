""" plotting.py 

Key plotting functions for the simulator.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np

def prop_fair_plotter(timestep, tx, grid_alloc, num_users, user_rates, max_data_sent_per_rb, 
                        save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """
    Plot the resource allocation grid, a dual-axis bar chart (RB allocation and throughput),
    and the RB allocation time series. Now handles unallocated RBs (-1).
    
    Parameters
    ----------
      grid_alloc: np.ndarray
        2D numpy array with user IDs allocated for each RB. -1 indicates unallocated RBs.

      num_users: int
        total number of users.

      user_rates: np.ndarray
        1D with achieved throughput (bps) for each user.

      max_data_sent_per_rb: int
        Maximum bits per RB if BLER were zero.

    """
    # Compute per-user statistics
    time_slots = grid_alloc.shape[0]
    rb_per_user_per_timestep = np.array([(grid_alloc == i).sum(axis=1) for i in range(num_users)])
    total_rbs_per_user = rb_per_user_per_timestep.sum(axis=1)
    
    # Count unallocated RBs per timestep
    unallocated_per_timestep = (grid_alloc == -1).sum(axis=1)
    total_unallocated = unallocated_per_timestep.sum()
    
    # Create figure with GridSpec:
    # Top row: two columns; Bottom row: single subplot spanning both columns.
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Create colormap with extra color for unallocated RBs
    # Use tab20 for users, and add gray for unallocated
    cmap = plt.get_cmap('tab20', num_users)
    colors = [cmap(i) for i in range(num_users)]
    colors.append((0.8, 0.8, 0.8, 1.0))  # Light gray for unallocated
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap
    extended_cmap = ListedColormap(colors)
    
    # ---------------------------
    # Top-Left: Resource Allocation Grid
    # ---------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    # Shift grid values so -1 becomes the last color index
    grid_display = grid_alloc.copy()
    grid_display[grid_alloc == -1] = num_users  # Map -1 to the last color
    
    cax = ax1.imshow(grid_display.T, aspect='auto', cmap=extended_cmap, origin='lower', 
                     vmin=0, vmax=num_users)
    
    # Create custom colorbar
    cbar = fig.colorbar(cax, ax=ax1, ticks=list(range(num_users)) + [num_users])
    cbar_labels = [f"User {i}" for i in range(num_users)] + ["Unallocated"]
    cbar.ax.set_yticklabels(cbar_labels)
    
    ax1.set_xlabel('Time Slot Index', fontsize=12)
    ax1.set_ylabel('Resource Block Index', fontsize=12)
    ax1.set_title(f'Resource Block Allocation Grid (TX {tx}, Time {timestep})', fontsize=14)
    plt.setp(ax1.get_xticklabels(), fontsize=10)
    
    # ---------------------------
    # Top-Right: Dual-Axis Bar Chart for Total RB Allocation & Throughput
    # ---------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    x = np.arange(num_users + 1)  # +1 for unallocated category
    
    # Prepare data including unallocated
    rb_data = list(total_rbs_per_user) + [total_unallocated]
    throughput_data = list(user_rates / 1e6) + [0]  # Unallocated has 0 throughput
    bar_colors = colors  # Uses the same color scheme as the grid
    
    # Left axis: Total RB allocation bars
    bars_rb = ax2.bar(x - width/2, rb_data, width=width, 
                      label='Total RBs Allocated',
                      color=bar_colors)
    ax2.set_xlabel('User ID', fontsize=12)
    ax2.set_ylabel('Total RBs Allocated', color='black', fontsize=12)
    ax2.set_xticks(x)
    x_labels = [f'User {i}' for i in range(num_users)] + ['Unallocated']
    ax2.set_xticklabels(x_labels, fontsize=8, rotation=45)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Right axis: Throughput bars in Mbps (actual values, not normalized)
    ax2_twin = ax2.twinx()
    bars_tp = ax2_twin.bar(x + width/2, throughput_data, width=width, 
                           label='Throughput (Mbps)',
                           color=bar_colors, hatch='//', alpha=0.7)
    ax2_twin.set_ylabel('Throughput (Mbps)', color='black', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='black')
    
    # Manually create a combined legend
    # Use the first bar from each set to represent that series
    ax2.legend([bars_rb[0], bars_tp[0]], ['Total RBs Allocated', 'Throughput (Mbps)'], 
               loc='upper right', fontsize=10)
    ax2.set_title('Total RB Allocation & Throughput per User', fontsize=14)
    
    # ---------------------------
    # Bottom: RB Allocation Time Series (spanning full width)
    # ---------------------------
    ax3 = fig.add_subplot(gs[1, :])
    for i in range(num_users):
        ax3.plot(range(time_slots), rb_per_user_per_timestep[i], 
                label=f'User {i}', color=cmap(i), linewidth=2)
    
    # Add unallocated RBs line
    ax3.plot(range(time_slots), unallocated_per_timestep, 
             label='Unallocated', color=(0.8, 0.8, 0.8, 1.0), 
             linewidth=2, linestyle='--')
    
    ax3.set_xlabel('Time Slot Index', fontsize=12)
    ax3.set_ylabel('RBs Allocated', fontsize=12)
    ax3.set_title('RB Allocation per User Over Time', fontsize=14)
    ax3.legend(loc='upper right', fontsize=10)
    
    # Set y-axis limits considering unallocated RBs
    max_rbs = max(rb_per_user_per_timestep.max(), unallocated_per_timestep.max(), 1)
    ax3.set_ylim(0, max_rbs)
    ax3.set_xlim(0, time_slots)  # x-axis from 0 to number of time slots
    plt.setp(ax3.get_xticklabels(), fontsize=10)
    
    # Add text annotation showing total unallocated RBs
    total_rbs = time_slots * grid_alloc.shape[1]
    unallocated_percentage = (total_unallocated / total_rbs) * 100
    ax3.text(0.02, 0.98, f'Total Unallocated: {total_unallocated} RBs ({unallocated_percentage:.1f}%)', 
             transform=ax3.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout so all subplots have equal spacing
    plt.tight_layout()
    fig.savefig(save_path + f"Scheduler_TX_{tx}_Time_{timestep}.png", dpi=600)
    plt.close()

# def prop_fair_plotter(timestep, tx, grid_alloc, num_users, user_rates, max_data_sent_per_rb, 
#                         save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
#     """
#     Plot the resource allocation grid, a dual-axis bar chart (RB allocation and throughput),
#     and the RB allocation time series.
    
#     Parameters
#     ----------
#       grid_alloc: np.ndarray
#         2D numpy array with user IDs allocated for each RB.

#       num_users: int
#         total number of users.

#       user_rates: np.ndarray
#         1D with achieved throughput (bps) for each user.

#       max_data_sent_per_rb: int
#         Maximum bits per RB if BLER were zero.

#     """
#     # Compute per-user statistics
#     time_slots = grid_alloc.shape[0]
#     rb_per_user_per_timestep = np.array([(grid_alloc == i).sum(axis=1) for i in range(num_users)])
#     total_rbs_per_user = rb_per_user_per_timestep.sum(axis=1)
    
#     # Create figure with GridSpec:
#     # Top row: two columns; Bottom row: single subplot spanning both columns.
#     fig = plt.figure(figsize=(24, 16))
#     gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
#     cmap = plt.get_cmap('tab20', num_users)
    
#     # ---------------------------
#     # Top-Left: Resource Allocation Grid
#     # ---------------------------
#     ax1 = fig.add_subplot(gs[0, 0])
#     cax = ax1.imshow(grid_alloc.T, aspect='auto', cmap=cmap, origin='lower', vmin=0, vmax=num_users - 1)
#     cbar = fig.colorbar(cax, ax=ax1, ticks=range(num_users))
#     cbar.ax.set_yticklabels([f"User {i}" for i in range(num_users)])
#     ax1.set_xlabel('Time Slot Index', fontsize=12)
#     ax1.set_ylabel('Resource Block Index', fontsize=12)
#     ax1.set_title(f'Resource Block Allocation Grid (TX {tx}, Time {timestep})', fontsize=14)
#     plt.setp(ax1.get_xticklabels(), fontsize=10)
    
#     # ---------------------------
#     # Top-Right: Dual-Axis Bar Chart for Total RB Allocation & Throughput
#     # ---------------------------
#     ax2 = fig.add_subplot(gs[0, 1])
#     width = 0.45
#     x = np.arange(num_users)
    
#     # Left axis: Total RB allocation bars
#     bars_rb = ax2.bar(x - width/2, total_rbs_per_user, width=width, 
#                       label='Total RBs Allocated',
#                       color=[cmap(i) for i in range(num_users)])
#     ax2.set_xlabel('User ID', fontsize=12)
#     ax2.set_ylabel('Total RBs Allocated', color='black', fontsize=12)
#     ax2.set_xticks(x)
#     ax2.set_xticklabels([f'User {i}' for i in range(num_users)], fontsize=6)
#     ax2.tick_params(axis='y', labelcolor='black')
    
#     # Right axis: Throughput bars in Mbps (actual values, not normalized)
#     ax2_twin = ax2.twinx()
#     # Convert throughput from bps to Mbps
#     throughput_mbps = user_rates / 1e6
#     bars_tp = ax2_twin.bar(x + width/2, throughput_mbps, width=width, 
#                            label='Throughput (Mbps)',
#                            color=[cmap(i) for i in range(num_users)], hatch='//', alpha=0.7)
#     ax2_twin.set_ylabel('Throughput (Mbps)', color='black', fontsize=12)
#     ax2_twin.tick_params(axis='y', labelcolor='black')
    
#     # Manually create a combined legend
#     # Use the first bar from each set to represent that series
#     ax2.legend([bars_rb[0], bars_tp[0]], ['Total RBs Allocated', 'Throughput (Mbps)'], loc='upper right', fontsize=10) # Change to fix legend
#     ax2.set_title('Total RB Allocation & Throughput per User', fontsize=14)
#     plt.setp(ax2.get_xticklabels(), fontsize=10)
    
#     # ---------------------------
#     # Bottom: RB Allocation Time Series (spanning full width)
#     # ---------------------------
#     ax3 = fig.add_subplot(gs[1, :])
#     for i in range(num_users):
#         ax3.plot(range(time_slots), rb_per_user_per_timestep[i], label=f'User {i}', color=cmap(i))
#     ax3.set_xlabel('Time Slot Index', fontsize=12)
#     ax3.set_ylabel('RBs Allocated', fontsize=12)
#     ax3.set_title('RB Allocation per User Over Time', fontsize=14)
#     ax3.legend(loc='upper right', fontsize=10)
#     ax3.set_ylim(0, max(rb_per_user_per_timestep.max(), 1))
#     ax3.set_xlim(0, time_slots)  # x-axis from 0 to number of time slots
#     plt.setp(ax3.get_xticklabels(), fontsize=10)
    
#     # Adjust layout so all subplots have equal spacing
#     plt.tight_layout()
#     fig.savefig(save_path + f"Scheduler_TX_{tx}_Time_{timestep}.png", dpi=600)
#     plt.close()


def plot_total_rewards(episode, reward, throughput, fairness, se, pe, su, save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """
    Plot performance metrics on a grid of four subplots.

    The subplots are organized as follows:
        - Top Left: Total Throughput with Fairness (secondary y-axis).
        - Top Right: Spectral Efficiency with Spectral Utility.
        - Bottom Left: Power Efficiency.
        - Bottom Right: Average Reward with its IQR, Max, and Min.

    Parameters
    ----------
    episode : int
        The last episode index.
    
    reward : list of lists
        A 2D list of rewards where each inner list corresponds to an episode.
        Some entries may be None if the episode has not been fully executed.
    
    throughput : numpy.ndarray
        Array of throughput values per episode.
    
    fairness : numpy.ndarray
        Array of fairness values per episode.
    
    se : numpy.ndarray
        Array of spectral efficiency values per episode.
    
    pe : numpy.ndarray
        Array of power efficiency values per episode.
        
    su : numpy.ndarray
        Array of spectral utility values per episode.
    
    save_path : str, optional
        The directory path to save the resulting figure, by default
        "/home/ubuntu/spectrum_sharing/Simulations/".

    Returns
    -------
    None
        The function saves the plotted figure to the specified path.
    """    
    # Create a figure with a 2x2 grid of subplots.
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # Create x-axis values (0 to episode inclusive).
    x = np.arange(episode + 1)
    
    # Get a colormap for consistency.
    cmap = plt.get_cmap("tab10", 5)
    
    # ---- Top Left: Throughput & Fairness ----
    ax1 = axs[0, 0]
    # Plot Throughput
    ax1.plot(x, throughput[:episode+1], linewidth=2, linestyle="solid",
             color=cmap(0), alpha=0.8, label="Average Throughput")
    ax1.set_xlabel("Episode", fontsize=12, color="black")
    ax1.set_ylabel("Throughput [Mbps]", fontsize=12, color="black")
    ax1.set_title("Total Throughput & Fairness", fontsize=16, color="black")
    ax1.tick_params(axis='both', colors="black")
    for spine in ax1.spines.values():
        spine.set_color("black")
    
    # Twin axis for Fairness
    ax1b = ax1.twinx()
    ax1b.plot(x, fairness[:episode+1], linewidth=2, linestyle="dashed",
              color=cmap(1), alpha=0.8, label="Fairness")
    ax1b.set_ylabel("Throughput Fairness (JFI)", fontsize=12, color="black")
    ax1b.set_ylim(0, 1)
    ax1b.tick_params(axis='y', colors="black")
    # for spine in ax1b.spines.values():
    #     spine.set_color("black")
    # Adjust zorder to help the fairness line be visible
    # ax1b.set_zorder(1)
    # ax1.set_zorder(2)
    
    # Legends
    # ax1.legend(loc="upper left", fontsize=10)
    # ax1b.legend(loc="upper right", fontsize=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    
    # ---- Top Right: Spectral Efficiency & Spectral Utility ----
    ax2 = axs[0, 1]
    ax2.plot(x, se[:episode+1], linewidth=2, linestyle="solid",
             color=cmap(2), alpha=0.8, label="Spectral Efficiency")
    ax2.plot(x, su[:episode+1], linewidth=2, linestyle="solid", 
             color=cmap(3), alpha=0.8, label="Spectral Utility")
    ax2.set_xlabel("Episode", fontsize=12, color="black")
    ax2.set_ylabel("Spectral Metrics [b/s/Hz]", fontsize=12, color="black")
    ax2.set_title("Spectral Efficiency & Utility", fontsize=16, color="black")
    ax2.tick_params(axis='both', colors="black")
    for spine in ax2.spines.values():
        spine.set_color("black")
    ax2.legend(fontsize=10, loc="upper right")
    
    # ---- Bottom Left: Power Efficiency ----
    ax3 = axs[1, 0]
    ax3.plot(x, np.array(pe[:episode+1]) * 1e6, linewidth=2, linestyle="solid",
             color=cmap(4), alpha=0.8, label="Power Efficiency")
    ax3.set_xlabel("Episode", fontsize=12, color="black")
    ax3.set_ylabel("Power Efficiency [W/MHz]", fontsize=12, color="black")
    ax3.set_title("Power Efficiency", fontsize=16, color="black")
    ax3.tick_params(axis='both', colors="black")
    for spine in ax3.spines.values():
        spine.set_color("black")
    ax3.legend(fontsize=10)
    
    # ---- Bottom Right: Reward with IQR, Max, and Min ----
    reward_trunc = np.array(reward[:episode+1], dtype=float) # Truncate the rewards array to only include episodes with filled data.

    reward_mean = np.mean(reward_trunc, axis=1)
    reward_iqr_low = np.percentile(reward_trunc, 25, axis=1)
    reward_iqr_high = np.percentile(reward_trunc, 75, axis=1)
    reward_max = np.max(reward_trunc, axis=1)
    reward_min = np.min(reward_trunc, axis=1)
    
    ax4 = axs[1, 1]
    ax4.plot(x, reward_mean, linewidth=2, linestyle="solid",
             color="black", alpha=0.95, label="Average Reward")
    ax4.fill_between(x,
                     reward_iqr_low,
                     reward_iqr_high,
                     color="grey", alpha=0.3, label="IQR (25th-75th)")
    # Plot Max and Min rewards
    ax4.plot(x, reward_max, linewidth=1.5, linestyle="dashed",
             color="black", alpha=0.8, label="Max Reward")
    ax4.plot(x, reward_min, linewidth=1.5, linestyle="dashed",
             color="black", alpha=0.8, label="Min Reward")
    
    ax4.set_xlabel("Episode", fontsize=12, color="black")
    ax4.set_ylabel("Reward", fontsize=12, color="black")
    ax4.set_title("Average Reward with IQR, Max, and Min", fontsize=16, color="black")
    ax4.tick_params(axis='both', colors="black")
    for spine in ax4.spines.values():
        spine.set_color("black")
    ax4.legend(fontsize=10, loc="upper right")
    
    # Save the figure.
    fig.savefig(save_path + "Rewards_Tracker.png", dpi=400)
    plt.close(fig)

def plot_rewards(episode,
                 step,
                 rewards,
                 save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """
    Plot reward functions over time with fairness on a secondary y-axis.

    The subplots are organized as follows:
        - Top Left: Total Throughput with Fairness (secondary y-axis).
        - Top Right: Spectral Efficiency.
        - Bottom Left: Spectrum Utility.
        - Bottom Right: Power Efficiency.

    Parameters
    ----------
    episode : int
        The current episode number.

    step : int
        The number of steps in the episode.

    rewards : numpy.ndarray
        2D array with shape (step+1, 5) where each column corresponds to a metric:
            Column 0: Total Throughput [Mbps]
            Column 1: Throughput Fairness (JFI)
            Column 2: Spectral Efficiency [bits/s/Hz]
            Column 3: Power Efficiency [W/MHz]
            Column 4: Spectrum Utility [bits/s/Hz]

    save_path : str, optional
        Directory path to save the resulting figure, by default
        "/home/ubuntu/spectrum_sharing/Simulations/".

    Returns
    -------
    None
        The function saves the plotted figure to the specified path.
    """
    # Define labels and titles for the plots.
    reward_labels = ["Total Throughput [Mbps]",
                     "Throughput Fairness (JFI)",
                     "Spectral Efficiency [bits/s/Hz]",
                     "Power Efficiency [W/MHz]",
                     "Spectrum Utility [bits/s/Hz]"]
    
    reward_titles = ["Total Throughput",
                     "Spectral Efficiency",
                     "Spectrum Utility",
                     "Power Efficiency"]
    
    # Create figure with 2x2 grid.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    cmap = plt.get_cmap("tab10", rewards.shape[1])
    
    # X-axis values.
    upper_x = step + 1
    x = np.linspace(0, step, upper_x)
    
    # Iterate over subplots.
    for i, ax in enumerate(axes.flat):
        if i == 0:
            # Top Left: Throughput and Fairness.
            ax.plot(x, rewards[:upper_x, 0], linewidth=2, linestyle="solid",
                    color=cmap(0), alpha=0.8, label="Total Throughput")
            # ax.set_ylim([np.min(rewards[:upper_x, 0]), np.max(rewards[:upper_x, 0])])
            
            # Create secondary y-axis for Fairness.
            ax2 = ax.twinx()
            ax2.plot(x, rewards[:upper_x, 1], linewidth=2, linestyle="dashed",
                     color=cmap(1), alpha=0.8, label="Fairness (JFI)")
            ax2.set_ylim([-0.05, 1.05])  # Fairness scale 0-1.
            
            # Labels and title.
            ax.set_ylabel(reward_labels[0], fontsize=12)
            ax2.set_ylabel(reward_labels[1], fontsize=12)
            ax.set_title(reward_titles[0], fontsize=16)
            
            # Legends.
            ax.legend(loc="upper left", fontsize=10)
            ax2.legend(loc="upper right", fontsize=10)
            
        elif i == 1:
            # Top Right: Spectral Efficiency (using column 2).
            ax.plot(x, rewards[:upper_x, 2], linewidth=2, linestyle="solid",
                    color=cmap(2), alpha=0.8, label="Spectral Efficiency")
            # ax.set_ylim([np.min(rewards[:upper_x, 2]), np.max(rewards[:upper_x, 2])])
            ax.set_ylabel(reward_labels[2], fontsize=12)
            ax.set_title(reward_titles[1], fontsize=16)
            ax.legend(loc="upper left", fontsize=10)
            
        elif i == 2:
            # Bottom Left: Spectrum Utility (using column 4).
            ax.plot(x, rewards[:upper_x, 4], linewidth=2, linestyle="solid",
                    color=cmap(3), alpha=0.8, label="Spectrum Utility")
            # ax.set_ylim([np.min(rewards[:upper_x, 4]), np.max(rewards[:upper_x, 4])])
            ax.set_ylabel(reward_labels[4], fontsize=12)
            ax.set_title(reward_titles[2], fontsize=16)
            ax.legend(loc="upper left", fontsize=10)
            
        elif i == 3:
            # Bottom Right: Power Efficiency (using column 3 with conversion).
            ax.plot(x, rewards[:upper_x, 3] * 1e6, linewidth=2, linestyle="solid",
                    color=cmap(4), alpha=0.8, label="Power Efficiency")
            # ax.set_ylim([np.min(rewards[:upper_x, 3] * 1e6), np.max(rewards[:upper_x, 3] * 1e6)])
            ax.set_ylabel(reward_labels[3], fontsize=12)
            ax.set_title(reward_titles[3], fontsize=16)
            ax.legend(loc="upper left", fontsize=10)
        
        ax.set_xlabel("Step", fontsize=12)
        ax.set_xlim([0, step])
    
    # Save and close plot.
    fig.savefig(save_path + f"Rewards_Ep_{episode}.png", dpi=400)
    plt.close(fig)

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

    # # Transmitters - positions adjusted to cm coordinate system - currently got a bug due to centre of grids not being at 0,0
    # tx_x_positions = [transmitter["position"][0]/cell_size + (x_max/2/*cell_size) for transmitter in transmitters.values()]
    # tx_y_positions = [transmitter["position"][1]/cell_size + (y_max/2*cell_size) for transmitter in transmitters.values()]

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

        # ax.scatter(
        #             tx_x_positions, tx_y_positions, s=1000, c="r", marker="*"
        #           )

    else:
        [arrow.remove() for arrow in ax.findobj(match=mpl.quiver.Quiver)]
        [img.remove() for img in ax.images]
        if step > 5:
            [line.remove() for line in ax.lines[:min(10,len(ax.lines))]]

    # Plotting the coverage map
    if tf.is_tensor(cm):
        cm_max = tf.reduce_max(cm)
        cm_max_val = cm_max.numpy()  # Convert to Python scalar
    else:
        cm_max_val = np.max(cm)

    if cm_max_val > max(sinr_range):  # Example: sinr_range = [-100, 100]
        cm_db = 10 * tf.math.log(cm) / tf.math.log(tf.constant(10.0, dtype=cm.dtype))
    else:
        cm_db = cm
    map = ax.imshow(cm_db, cmap=color, origin='lower', alpha=0.7, extent=[0, x_max, 0, y_max], vmin=sinr_range[0], vmax=sinr_range[1])
    if step == 0:
        cbar = fig.colorbar(map, ax=ax, shrink=0.8)
        cbar.set_label("SINR [dB]", fontsize=40) 
        cbar.ax.tick_params(labelsize=30)
    grid = grid.numpy() if tf.is_tensor(grid) else grid
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