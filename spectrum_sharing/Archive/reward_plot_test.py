from plotting import plot_motion, plot_performance, plot_rewards
import tensorflow as tf
from hydra import compose, initialize 

def main(cfg):

    fig_3, ax_3 = None, None
    rewards = tf.zeros(shape=(cfg.episodes, 4), dtype=tf.float32)

    for e in range(cfg.episodes):

        throughput = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32)
        se = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32)
        pe = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32)
        su = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32)
        indices = tf.constant([[e, 0], [e, 1], [e, 2], [e, 3]])
        updates = tf.stack([throughput, se, pe, su], axis=0)
        rewards = tf.tensor_scatter_nd_update(rewards, indices, tf.reshape(updates, (4,)))

        # Plotting objectives/ rewards
        # rewards = tf.stack([total_throughput, total_se, total_pe, total_su], axis=0)
        plot_rewards(episode=e,
                     rewards=rewards,
                     save_path=cfg.images_path)


if __name__ == "__main__":
    # Configuration
    random_seed = 40
    with initialize(version_base=None, config_path="conf", job_name="simulation"):
        config = compose(config_name="simulation")
        #print(OmegaConf.to_yaml(config))
    main(config)