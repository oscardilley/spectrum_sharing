""" DQN_agent.py

Implementing an agent in Tensorflow to perform deep Q learning.  """

import tensorflow as tf
from collections import deque
import numpy as np
from itertools import product
import gymnasium as gym
import pickle

from logger import logger

class Agent:
    def __init__(self, cfg, observation_space, action_space, possible_actions, num_possible_actions, path, test=False):
        self.cfg = cfg
        self.observation_space = observation_space
        self.action_space = action_space

        self.path = path + "model" # add .h5 to switch to H5 saved model format

        print(self.observation_space)

        # Obtaining preprocessed actions
        self.actions = possible_actions
        self.num_actions = num_possible_actions

        # Hyperparameters
        self.gamma = self.cfg.gamma
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_min = self.cfg.epsilon_min
        self.epsilon_decay = self.cfg.epsilon_decay
        self.learning_rate = self.cfg.learning_rate
        
        # Initialize the Q-network and target network
        if test:
            # Testing the loaded network
            with open(self.path + "/saved_model.pb", "r"):
                logger.warning("Loading existing model.")
            self.model, self.target_model = self.load_model()
            self.epsilon = 0.0 # zero exploration in test mode
            logger.warning(f"Epsilon initialised at {self.epsilon}")
        else:
            try:
                with open(self.path + "/saved_model.pb", "r"):
                    logger.warning("Loading existing model.")
                self.model, self.target_model = self.load_model()
                self.epsilon = self.cfg.epsilon_quick_start # initalised epsilon to reduce exploration on pre-trained model
                logger.warning(f"Epsilon initialised at {self.epsilon}")
            except FileNotFoundError:
                logger.warning("Starting new model.")
                self.model = self.build_model()
                self.target_model = self.build_model()
            
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.loss_function = tf.keras.losses.Huber(delta=5.0, reduction="sum_over_batch_size", name="huber_loss") # for balance between L1 and L2
        
        # Synchronize the target network
        self.update_target_network()
    
    def build_model(self):
        """Build the Q-network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.observation_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_actions, activation='relu') # relu instead of 'linear' for case that Q values cannot be negative
        ])
        
        return model

    def update_target_network(self):
        """Synchronize target network with main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, observation):
        """Select an action using epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon: # decaying epsilon
            idx = np.random.choice(len(self.actions))
            logger.info("Random Action.")
            return self.actions[idx], idx
        else:
            q_values = self.model.predict(observation[np.newaxis], verbose=0) # add extra axis for batch

            # consider discouraging the selection of the same action again

            logger.info(f"Q-values: Mean={np.mean(q_values)}, Max={np.max(q_values)}, Min={np.min(q_values)}")
            idx = np.argmax(q_values[0])
            logger.info("Q Action.")
            return self.actions[idx], idx
    
    def train(self, replay_buffer, batch_size, timestep):
        """Train the Q-network using experience from the replay buffer."""
        if len(replay_buffer) < batch_size:
            return
        logger.info("Training.")
        
        for e in range(self.cfg.training_epochs):
            observations, actions, rewards, next_observations, terminateds = replay_buffer.sample(batch_size)

            # Compute target Q-values using the Bellman equation:
            next_qs = self.target_model.predict(next_observations, verbose=0)
            max_next_qs = np.max(next_qs, axis=1)
            target_qs = rewards + ((1 - terminateds) * self.gamma * max_next_qs)

            # Also need to consider applying Q-value clipping to a realistic range based on knowledge and number of episodes
            
            # Train Q-network
            with tf.GradientTape() as tape:
                q_values = self.model(observations, training=True)
                indices = tf.stack([tf.range(batch_size), actions], axis=1)
                selected_qs = tf.gather_nd(q_values, indices)
                loss = self.loss_function(target_qs, selected_qs)
                logger.info(f"Training loss for epoch {e}: {loss}")
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logger.info(f"New Epsilon: {self.epsilon}")

        if timestep % int(self.cfg.step_limit) == 0:
            logger.info("Periodically saving model.")
            self.save_model()
        
        return

    def save_model(self):
        """ Saving the neural network."""
        self.model.save(self.path)
        self.target_model.save(self.path + "_target")
        pass

    def load_model(self):
        """Check if model already exists and load it."""
        model = tf.keras.models.load_model(self.path)
        target_model = tf.keras.models.load_model(self.path + "_target")
        return model, target_model


class ReplayBuffer:
    def __init__(self, max_size, path):
        self.path = path + "buffer.pickle"

        try:
            with open(self.path, "rb") as file:
                self.buffer = pickle.load(file)
                logger.warning(f"Loaded buffer of length: {self.__len__()}")
        except FileNotFoundError:
            logger.warning("Starting new buffer.")
            self.buffer = deque(maxlen=max_size)

    def add(self, experience, timestep):
        self.buffer.append(experience) 

        if timestep % 33 == 0:
            logger.info(f"Periodic saving buffer. Length: {self.__len__()}")
            self.save_buffer()

        return

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False) # random to break temporal correlations
        batch = [self.buffer[idx] for idx in indices]
        # Unzip the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
    
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
    
    def save_buffer(self):
        """ Save the buffer object in pickle. """
        with open(self.path, "wb") as file:
            pickle.dump(self.buffer, file)

