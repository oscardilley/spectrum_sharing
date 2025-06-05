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
RESULTS_PATH = "./spectrum_sharing/Tests/aggregated_results_simulation5.csv"
TEST = True

def main(cfg):
    """ Extracting and plotting the results."""
    df = pd.read_csv(RESULTS_PATH)
    if TEST:
        # plotting against random seed
        df.drop(columns=["timestamp"], inplace=True)
        test_labels = df["test_label"].unique().tolist()
        columns = df.columns.tolist()
        df.set_index(["test_label", "seed"], inplace=True)
        
        # Analytics for reporting
        for col in set(columns).difference(["seed", "test_label"]):
            logger.info(df.groupby("test_label")[col].mean())

        plot_dataframe_metrics(df)


    return


def plot_dataframe_metrics(df, save_path="./spectrum_sharing/Tests/Images/"):
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
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(test_labels)]
    
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
    
    # Save the figure
    fig.savefig(save_path + "DataFrame_Metrics.png", dpi=400, bbox_inches='tight')
    plt.show()
    
    return fig


if __name__ == "__main__":
    with initialize(version_base=None, config_path="Config", job_name=CONFIG_NAME):
        config = compose(config_name=CONFIG_NAME)
    main(config)