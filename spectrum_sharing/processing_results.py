""" processing_results.py

Extracting the saved training or testing results from CSV and plotting either against time or against random seed.

"""

from hydra import compose, initialize 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from spectrum_sharing.logger import logger
# from spectrum_sharing.plotting import test_reward_plotting

CONFIG_NAME = "simulation5"
TEST = True
if TEST:
    RESULTS_PATH = "./spectrum_sharing/Tests/aggregated_results_simulation5_20250605.csv"
    RESULTS_PATH = "./spectrum_sharing/Tests/aggregated_results_simulation5.csv"
else:
    RESULTS_PATH = "/home/ubuntu/spectrum_sharing_v0/spectrum_sharing/Results/results_20250605_125100.csv"
    RESULTS_PATH = "/home/ubuntu/spectrum_sharing_v0/spectrum_sharing/Results/results_20250603_173631.csv"
date = RESULTS_PATH.split("_")[-1].split(".")[0]

def main(cfg):
    """ Extracting and plotting the results."""
    df = pd.read_csv(RESULTS_PATH)
    if TEST:
        df.drop(columns=["timestamp"], inplace=True)
        columns = df.columns.tolist()
        # plotting against random seed
        test_labels = df["test_label"].unique().tolist()
        df.set_index(["test_label", "seed"], inplace=True)
        
        # Analytics for reporting
        for col in set(columns).difference(["seed", "test_label"]):
            logger.info(df.groupby("test_label")[col].mean())

        # Plotting
        plot_test_results(df)

    else:
        df
        columns = df.columns.tolist()

        if (set(columns) & set(["avg_reward", "min_reward", "max_reward"])) == set():
            # Need to precompute the reward
            norm_ranges = {'throughput': (0, 43), 
                           'fairness': (0, 1),
                           'se': (0, 7), 
                           'pe': (4.2097352e-07, 1), 
                           'su': (0, 7)} # extract from RL_simulator.py initalisation
            
            # Calculate normalized values for each metric
            norm_throughput = norm(df['avg_throughput'], *norm_ranges['throughput'])
            norm_fairness = norm(df['avg_fairness'], *norm_ranges['fairness'])
            norm_se = norm(df['avg_se'], *norm_ranges['se'])
            # (1/pe, 1/self.norm_ranges["pe"][1], 1/self.norm_ranges["pe"][0])
            norm_pe = norm(1/df['avg_pe'], 1/norm_ranges['pe'][1], 1/norm_ranges['pe'][0])
            norm_su = norm(df['avg_su'], *norm_ranges['su'])
            
            # Calculate reward as sum of normalized values
            df['avg_reward'] = norm_throughput + norm_fairness + norm_se + norm_pe + norm_su
            # Set min_reward and max_reward equal to avg_reward for now
            df['min_reward'] = df['avg_reward']
            df['max_reward'] = df['avg_reward']

        for col in columns:
            if col == "episode":
                continue
            logger.info(f"{col} Avg = {df[col].mean()}")
            logger.info(f"{col} Min = {df[col].min()}")
            logger.info(f"{col} Max = {df[col].max()}")

        plot_train_results(df)
        

    return

def plot_train_results(df, save_path="./spectrum_sharing/Tests/Images/"):
    """
    Plot performance metrics from DataFrame on a grid of four subplots (2x2).
    Each plot shows evolution over episodes (time).

    The subplots are organized as follows:
        - Top Left: Average Reward with Min/Max shaded region
        - Top Right: Throughput and Fairness
        - Bottom Left: Spectral Efficiency and Spectral Utility
        - Bottom Right: Power Efficiency

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        ['episode', 'avg_throughput', 'avg_fairness', 'avg_se', 'avg_pe', 'avg_su', 
         'avg_reward', 'min_reward', 'max_reward']
    
    save_path : str, optional
        The directory path to save the resulting figure
    """
    
    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    cmap = plt.get_cmap("tab10", 5)

    # ---- Top Left: Average Reward with Min/Max shaded region ----
    ax1 = axs[0, 0]
    ax1.plot(df['episode'], df['avg_reward'], color="k" , linewidth=2, label='Average Reward')
    if (df['min_reward'] != df['max_reward']).all():
        ax1.fill_between(df['episode'], df['min_reward'], df['max_reward'], 
                        alpha=0.15, color="k", label='Min-Max Range')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Average Reward Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(df['episode'])])
    
    # ---- Top Right: Throughput with Fairness on second axis ----
    ax2 = axs[0, 1]
    color1 = cmap(0)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Throughput')
    line1 = ax2.plot(df['episode'], df['avg_throughput'], color=color1, linewidth=2, label='Throughput')
    ax2.tick_params(axis='y')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(df['episode'])])
    
    # Create second y-axis for fairness
    ax2_twin = ax2.twinx()
    color2 = cmap(1)
    ax2_twin.set_ylabel('Fairness')
    line2 = ax2_twin.plot(df['episode'], df['avg_fairness'], color=color2, linewidth=2, label='Fairness')
    ax2_twin.tick_params(axis='y')
    ax2_twin.set_ylim([0, 1])
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.set_title('Throughput and Fairness')
    
    # ---- Bottom Left: Spectral Efficiency and Spectral Utility ----
    ax3 = axs[1, 0]
    color3 = cmap(2)
    color4 = cmap(3)
    ax3.plot(df['episode'], df['avg_se'], color=color3, linewidth=2, label='Spectral Efficiency')
    ax3.plot(df['episode'], df['avg_su'], color=color4, linewidth=2, label='Spectral Utility')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Value')
    ax3.set_title('Spectral Efficiency and Utility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, max(df['episode'])])

    # ---- Bottom Right: Power Efficiency ----
    ax4 = axs[1, 1]
    color5 = cmap(4)
    ax4.plot(df['episode'], df['avg_pe'], color=color5, linewidth=2, label='Power Efficiency')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Power Efficiency')
    ax4.set_title('Power Efficiency Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, max(df['episode'])])
    
    # Save the figure
    fig.savefig(save_path + f"Model_Training_{CONFIG_NAME}_{date}.png", dpi=400, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_test_results(df, save_path="./spectrum_sharing/Tests/Images/"):
    """
    Plot performance metrics from DataFrame on a grid of six subplots (2x3).
    Each plot shows evolution over seed values (time) with separate lines for each test_label.

    The subplots are organized as follows:
        - Top Left: Average Reward with Min/Max shaded region
        - Top Right: Throughput
        - Middle Left: Fairness
        - Middle Right: Spectral Efficiency
        - Bottom Left: Spectral Utility
        - Bottom Right: Power Efficiency

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with MultiIndex (test_label, seed) and columns:
        ['avg_throughput', 'avg_fairness', 'avg_se', 'avg_pe', 'avg_su', 
         'avg_reward', 'min_reward', 'max_reward']
    
    save_path : str, optional
        The directory path to save the resulting figure
    """
    
    # Create a figure with a 3x2 grid of subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 16), constrained_layout=True)
    
    # Get unique test_labels and seeds
    test_labels = df.index.get_level_values('test_label').unique()
    seeds = sorted(df.index.get_level_values('seed').unique())
    
    # Get a colormap for consistency
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_labels)))
    
    # Define unique markers for each test_label
    markers = ['o', 's', '^', '*', 'v', '<', '>', 'p', 'D', 'h'][:len(test_labels)]
    
    # ---- Top Left: Average Reward with Min/Max shaded region ----
    ax1 = axs[0, 0]
    
    for i, test_label in enumerate(test_labels):
        # Get data for this test_label
        test_data = df.loc[test_label]
        if isinstance(test_data, pd.Series):
            test_data = test_data.to_frame().T
        
        # Get available seeds for this test_label and sort them
        available_seeds = sorted([seed for seed in seeds if seed in test_data.index])
        
        if available_seeds:
            avg_rewards = [test_data.loc[seed, 'avg_reward'] for seed in available_seeds]
            min_rewards = [test_data.loc[seed, 'min_reward'] for seed in available_seeds]
            max_rewards = [test_data.loc[seed, 'max_reward'] for seed in available_seeds]
            
            # Plot average reward line
            ax1.plot(available_seeds, avg_rewards, linewidth=2, linestyle="solid",
                    color=colors[i], alpha=0.8, label=f"{test_label}", 
                    marker=markers[i], markersize=6)
            
            # Fill between min and max (lightly shaded)
            ax1.fill_between(available_seeds, min_rewards, max_rewards,
                            color=colors[i], alpha=0.2)
    
    ax1.set_xlabel("Random Seed", fontsize=12)
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.set_title("Average Reward", fontsize=16)
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(available_seeds)])
    
    # ---- Top Right: Throughput ----
    ax2 = axs[0, 1]
    
    for i, test_label in enumerate(test_labels):
        test_data = df.loc[test_label]
        if isinstance(test_data, pd.Series):
            test_data = test_data.to_frame().T
        
        available_seeds = sorted([seed for seed in seeds if seed in test_data.index])
        
        if available_seeds:
            throughputs = [test_data.loc[seed, 'avg_throughput'] for seed in available_seeds]
            
            ax2.plot(available_seeds, throughputs, linewidth=2, linestyle="solid",
                    color=colors[i], alpha=0.8, label=f"{test_label}", 
                    marker=markers[i], markersize=6)
    
    ax2.set_xlabel("Random Seed", fontsize=12)
    ax2.set_ylabel("Throughput [Mbps]", fontsize=12)
    ax2.set_title("Throughput", fontsize=16)
    ax2.legend(fontsize=10, loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(available_seeds)])
    
    # ---- Middle Left: Fairness ----
    ax3 = axs[1, 0]
    
    for i, test_label in enumerate(test_labels):
        test_data = df.loc[test_label]
        if isinstance(test_data, pd.Series):
            test_data = test_data.to_frame().T
        
        available_seeds = sorted([seed for seed in seeds if seed in test_data.index])
        
        if available_seeds:
            fairness_vals = [test_data.loc[seed, 'avg_fairness'] for seed in available_seeds]
            
            ax3.plot(available_seeds, fairness_vals, linewidth=2, linestyle="solid",
                    color=colors[i], alpha=0.8, label=f"{test_label}", 
                    marker=markers[i], markersize=6)
    
    ax3.set_xlabel("Random Seed", fontsize=12)
    ax3.set_ylabel("Fairness (JFI)", fontsize=12)
    ax3.set_title("Fairness", fontsize=16)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=10, loc="lower left")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, max(available_seeds)])
    
    # ---- Middle Right: Spectral Efficiency ----
    ax4 = axs[1, 1]
    
    for i, test_label in enumerate(test_labels):
        test_data = df.loc[test_label]
        if isinstance(test_data, pd.Series):
            test_data = test_data.to_frame().T
        
        available_seeds = sorted([seed for seed in seeds if seed in test_data.index])
        
        if available_seeds:
            se_vals = [test_data.loc[seed, 'avg_se'] for seed in available_seeds]
            
            ax4.plot(available_seeds, se_vals, linewidth=2, linestyle="solid",
                    color=colors[i], alpha=0.8, label=f"{test_label}", 
                    marker=markers[i], markersize=6)
    
    ax4.set_xlabel("Random Seed", fontsize=12)
    ax4.set_ylabel("Spectral Efficiency [b/s/Hz]", fontsize=12)
    ax4.set_title("Spectral Efficiency", fontsize=16)
    ax4.legend(fontsize=10, loc="lower left")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, max(available_seeds)])
    
    # ---- Bottom Left: Spectral Utility ----
    ax5 = axs[2, 0]
    
    for i, test_label in enumerate(test_labels):
        test_data = df.loc[test_label]
        if isinstance(test_data, pd.Series):
            test_data = test_data.to_frame().T
        
        available_seeds = sorted([seed for seed in seeds if seed in test_data.index])
        
        if available_seeds:
            su_vals = [test_data.loc[seed, 'avg_su'] for seed in available_seeds]
            
            ax5.plot(available_seeds, su_vals, linewidth=2, linestyle="solid",
                    color=colors[i], alpha=0.8, label=f"{test_label}", 
                    marker=markers[i], markersize=6)
    
    ax5.set_xlabel("Random Seed", fontsize=12)
    ax5.set_ylabel("Spectral Utility [b/s/Hz]", fontsize=12)
    ax5.set_title("Spectral Utility", fontsize=16)
    ax5.legend(fontsize=10, loc="lower left")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, max(available_seeds)])
    
    # ---- Bottom Right: Power Efficiency ----
    ax6 = axs[2, 1]
    
    for i, test_label in enumerate(test_labels):
        test_data = df.loc[test_label]
        if isinstance(test_data, pd.Series):
            test_data = test_data.to_frame().T
        
        available_seeds = sorted([seed for seed in seeds if seed in test_data.index])
        
        if available_seeds:
            pe_vals = [test_data.loc[seed, 'avg_pe'] * 1e6 for seed in available_seeds]  # Convert to match original scaling
            
            ax6.plot(available_seeds, pe_vals, linewidth=2, linestyle="solid",
                    color=colors[i], alpha=0.8, label=f"{test_label}", 
                    marker=markers[i], markersize=6)
    
    ax6.set_xlabel("Random Seed", fontsize=12)
    ax6.set_ylabel("Power Efficiency [W/MHz]", fontsize=12)
    ax6.set_title("Power Efficiency", fontsize=16)
    ax6.legend(fontsize=10, loc="lower left")
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, max(available_seeds)])
    
    # Save the figure
    fig.savefig(save_path + f"Model_Evaluation_{CONFIG_NAME}_{date}.png", dpi=400, bbox_inches='tight')
    plt.show()
    
    return fig

def norm(value, min_val, max_val):
    """
    Min Max Normalisation of value to range [0,1] given a range. 
    
    Parameters
    ----------
    value : float
        Value for normalisation.

    min_val : float
        Upper bound for value.

    max_val : float
        Lower bound for value.

    Returns
    -------
    value_norm : float
        Min-max normalised value between [0,1].      
    
    """
    value_clipped = np.clip(value, min_val, max_val) # avoiding inf

    return (value_clipped - min_val) / (max_val - min_val)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="Config", job_name=CONFIG_NAME):
        config = compose(config_name=CONFIG_NAME)
    main(config)

    