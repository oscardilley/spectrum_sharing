""" plotting.py 

Key plotting functions for the simulator.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np

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
    reward_labels = ["Total Throughput [MHz]", "Spectral Efficiency [bits/s/Hz]", "Power Efficiency [W/MHz]", "Spectrum Utility [bits/s/Hz]"]
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
    map = ax.imshow(cm_db, cmap=color, origin='lower', alpha=0.7, extent=[0, x_max, 0, y_max], vmin=-100, vmax=100)
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
    
    fig.savefig(save_path + f"Scene {id} Step {step}.png")
    # if id == "Sharing Band, Max SINR":
    #     fig.savefig(save_path + f"Scene {id} Step {step}.png")
    # else:
    #     fig.savefig(save_path + f"Scene {id}.png")#, bbox_inches="tight")

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