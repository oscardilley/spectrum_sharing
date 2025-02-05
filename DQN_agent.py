""" DQN_agent.py

Implementing an agent in Tensorflow to perform deep Q learning.  """

import tensorflow as tf
from collections import deque
import numpy as np
from itertools import product
import gymnasium as gym
from gymnasium import spaces
import pickle

from logger import logger

class Agent:
    def __init__(self, cfg, observation_space, num_tx, action_space, possible_actions, num_possible_actions, path, test=False):
        self.cfg = cfg
        self.transmitters = dict(self.cfg.transmitters)
        self.num_tx = num_tx
        self.observation_space = observation_space
        self.action_space = action_space

        self.path = path + "model" # add .h5 to switch to H5 saved model format

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
        transmitter_heads = []
        processed_features = []

        for tx in range(self.num_tx):
            tx_input = tf.keras.layers.Input(shape=int(spaces.utils.flatdim(self.observation_space) / self.num_tx), name=f"tx_{tx}")
            transmitter_heads.append(tx_input)
            # Feature extraction per transmitter
            x = tf.keras.layers.Dense(128, activation='relu')(tx_input)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            processed_features.append(x)

        # Concatenate extracted features from all transmitters
        merged_features = tf.keras.layers.Concatenate()(processed_features)
        
        # Shared decision layers
        shared = tf.keras.layers.Dense(self.num_tx * 128, activation='relu')(merged_features)
        shared = tf.keras.layers.Dense(128, activation='relu')(shared)

        # Single output head predicting Q-values for all possible actions
        output = tf.keras.layers.Dense(self.num_actions, activation='relu')(shared)
        model = tf.keras.Model(inputs=transmitter_heads, outputs=output)

        return model
    
    def _preprocess_observation(self, observation):
        """Convert the nested observation dict into a list of flattened arrays for each transmitter."""
        processed_inputs = []
        
        for tx_id in range(self.num_tx):
            flattened = spaces.utils.flatten(self.observation_space[tx_id], observation[tx_id])
            processed_inputs.append(flattened) 
        
        return processed_inputs

    def update_target_network(self):
        """Synchronize target network with main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, observation):
        """Select an action using epsilon-greedy strategy."""
        valid_mask = self.get_valid_action_mask(observation)  # producing the action mask

        if np.random.rand() <= self.epsilon: # decaying epsilon
            valid_indices= np.where(valid_mask)[0]
            idx = np.random.choice(valid_indices)
            logger.info("Masked Random Action.")
            return self.actions[idx], idx
        else:
            # Predicting q-values and then masking them
            observation = [obs[np.newaxis] for obs in self._preprocess_observation(observation)] # for tf batch dimension, using [np.newaxis]
            q_values = self.model.predict(observation, verbose=0) # add extra axis for batch
            q_values = np.where(valid_mask, q_values, 0) # if valid, return the q value, else return zero

            logger.info(f"Q-values: Mean={np.mean(q_values)}, Max={np.max(q_values)}, Min={np.min(q_values)}")
            idx = np.argmax(q_values[0])
            logger.info(f"Q Action {self.actions[idx]}.")

            return self.actions[idx], idx
        
    def get_valid_action_mask(self, observation):
        """Create a mask for valid actions based on the power constraints."""
        valid_mask = np.ones(self.num_actions, dtype=bool)  # Start with all valid

        for id, action in enumerate(self.actions):
            for tx_id, tx in enumerate(action):
                power = round((float(observation[tx_id]["tx_power"][0]) * (self.cfg.max_power - self.cfg.min_power)) + self.cfg.min_power) # denormalising
                if tx[1] != 1:
                    if tx[0] == 0:
                         valid_mask[id] = 0 # power is only allowed to stay the same if turning transmitter off
                         continue
                    power = round((float(observation[tx_id]["tx_power"][0]) * (self.cfg.max_power - self.cfg.min_power)) + self.cfg.min_power) # denormalising
                    if (tx[1] == 2 and power == self.transmitters[f"tx{tx_id}"]["max_power"]) or (tx[1] == 0 and power == self.transmitters[f"tx{tx_id}"]["min_power"]):
                        valid_mask[id] = 0 # do not increase or decrease power beyond transmitter limit
                        continue
                else:
                    continue

        return valid_mask
    
    def train(self, replay_buffer, batch_size, timestep):
        """Train the Q-network using experience from the replay buffer."""
        if len(replay_buffer) < batch_size:
            return
        logger.info("Training.")
        
        for e in range(self.cfg.training_epochs):
            observations, actions, rewards, next_observations, terminateds = replay_buffer.sample(batch_size)

            # Preprocess all observations in the batch
            processed_obs = [self._preprocess_observation(obs) for obs in observations]
            processed_next_obs = [self._preprocess_observation(obs) for obs in next_observations]
            
            # Reshape the processed observations to match the model's input format
            valid_mask = np.array([self.get_valid_action_mask(next_observation) for next_observation in next_observations])
            observations = [np.stack([obs[i] for obs in processed_obs]) for i in range(self.num_tx)]
            next_observations = [np.stack([obs[i] for obs in processed_next_obs]) for i in range(self.num_tx)]

            # Compute target Q-values using the Bellman equation:
            next_qs = self.target_model.predict(next_observations, verbose=0)
            next_qs = np.where(valid_mask, next_qs, 0)
            max_next_qs = np.max(next_qs, axis=1)
            target_qs = rewards + ((1 - terminateds) * self.gamma * max_next_qs)
            
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

