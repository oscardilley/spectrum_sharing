""" plotting.py 

Key plotting functions for the simulator.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np

def plot_motion(episode, 
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
    tx_x_positions = [transmitter["position"][0]/cell_size + (x_max/2) + 0.5 for transmitter in transmitters.values()]
    tx_y_positions = [transmitter["position"][1]/cell_size + (y_max/2) + 0.5 for transmitter in transmitters.values()]

    # Axis initialisation
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(int(x_max/4), int(y_max/4)), constrained_layout=True)
    
        # Add gridlines
        ax.set_xticks(np.arange(x_max), minor=True)
        ax.set_yticks(np.arange(y_max), minor=True)
        ax.tick_params(which='minor', size=0)  # Remove tick markers for minor grid lines

        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])

        ax.scatter(
                    tx_x_positions, tx_y_positions, s=250, c="r", marker="*"
                  )

    else:
        [arrow.remove() for arrow in ax.findobj(match=mpl.quiver.Quiver)]
        [img.remove() for img in ax.images]
        if episode > 5:
            [line.remove() for line in ax.lines[:min(10,len(ax.lines))]]

    # Plotting the coverage map
    cm_db = 10 * tf.math.log(cm) / tf.math.log(10.0)
    map = ax.imshow(cm_db, cmap=color, origin='lower', alpha=0.7, extent=[0, x_max, 0, y_max])
    if episode == 0:
        cbar = fig.colorbar(map, ax=ax, shrink=0.8)
        cbar.set_label("SINR [dB]", fontsize=16) 
    grid = grid.numpy()
    ax.imshow(np.ma.masked_where(grid > 0, grid), cmap='Set2', origin='lower', alpha=0.6, extent=[0, x_max, 0, y_max])

    # Plot motion vectors
    ax.quiver(
        x_positions, y_positions,  # Start positions of the arrows
        dx, dy,  # Vector components
        angles='xy', scale_units='xy', scale=1.5, label='Direction'
    )

    # Labels and title
    ax.set_title(f"\n{id} Episode {episode}\n", fontsize=25)
    ax.set_xlabel("X [Cells]", fontsize=16)
    ax.set_ylabel("Y [Cells]", fontsize=16)
    ax.legend()
    
    fig.savefig(save_path + f"Scene {id}.png")#, bbox_inches="tight")

    for i in range(len(users)):
        ax.plot([x_positions[i], x_positions[i] + dx[i]], [y_positions[i], y_positions[i] + dy[i]], color=cmap(i), linewidth=2.5)
    
    return fig, ax


def plot_performance(episode,
                     users, 
                     performance,
                     fig=None, 
                     ax=None, 
                     save_path="/home/ubuntu/spectrum_sharing/Simulations/"):
    """ Plotting the BER and SNR for the users. """
    length = len(performance)
    x = np.linspace(episode + 1 - length, episode, length)

    # Extracting data
    primary_ber = tf.stack([results["Primary"]["ber"] for results in performance])
    sharing_ber = tf.stack([results["Sharing"]["ber"] for results in performance])
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
    fig.suptitle(f"Episode {episode} UE Performance", fontsize=25, fontdict={'color': "black", 'fontweight': 'bold'})

    # Plotting per user
    ue = 0
    for r in range(num_rows):
        for c in range(num_columns):
            ax1 = ax[r,c]
            x_pos = round(float(users[f"ue{ue}"]["position"][1]), 2)
            y_pos = round(float(users[f"ue{ue}"]["position"][0]), 2)
            ax1.set_title(f"UE {ue} at ({x_pos},{y_pos})", fontdict={'color': cmap(ue), 'weight': 'bold', 'fontsize': 16})
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("BER")
            ax1.set_ylim([-0.05, 0.55])
            ax1.set_xlim([x[0], x[-1]])
            ax1.set_xticks(x, x)
            ax2 = ax1.twinx()
            ax2.set_ylabel("SINR [dB]")
            ax2.set_ylim([-80, 70])
            berP0, = ax1.plot(x, primary_ber[:,0,ue], linestyle="dashed", color="orangered", alpha=0.8)
            berP1, = ax1.plot(x, primary_ber[:,1,ue], linestyle="dashed", color="darkred", alpha=0.8)
            sinrP0, = ax2.plot(x, primary_sinr[:,0,ue], linestyle="solid", color="orangered", alpha=0.8)
            sinrP1, = ax2.plot(x, primary_sinr[:,1,ue], linestyle="solid", color="darkred", alpha=0.8)
            berS0, = ax1.plot(x, sharing_ber[:,0,ue], linestyle="dashed", color="steelblue", alpha=0.8)
            berS1, = ax1.plot(x, sharing_ber[:,1,ue], linestyle="dashed", color="darkturquoise", alpha=0.8)
            sinrS0, = ax2.plot(x, sharing_sinr[:,0,ue], linestyle="solid", color="steelblue", alpha=0.8)
            sinrS1, = ax2.plot(x, sharing_sinr[:,1,ue], linestyle="solid", color="darkturquoise", alpha=0.8)
            ue += 1

    fig.legend((berP0, berP1, sinrP0, sinrP1, berS0, berS1, sinrS0, sinrS1),
               ("Primary BER TX0", "Primary BER TX1", "Primary SINR TX0", "Primary SINR TX1", "Secondary BER TX0", "Secondary BER TX1", "Secondary SINR TX0", "Secondary SINR TX1"),
               loc="outside right center")
    fig.savefig(save_path + f"UE_performance.png", dpi=400)

    return