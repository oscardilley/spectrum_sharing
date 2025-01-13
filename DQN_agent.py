""" DQN_agent.py

Implementing an agent in Tensorflow to perform deep Q learning.  """

import tensorflow as tf
from collections import deque
import numpy as np

class Agent:
    def __init__(self, cfg, state_shape, action_shape):
        self.cfg = cfg
        self.state_shape = state_shape
        self.num_actions = action_shape
        self.gamma = self.cfg.gamma
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_min = self.cfg.epsilon_min
        self.epsilon_decay = self.cfg.epsilon_decay
        self.learning_rate = self.cfg.learning_rate
        
        # Initialize the Q-network and target network
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        
        # Synchronize the target network
        self.update_target_network()
    
    def build_model(self):
        """Build the Q-network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_actions, activation='linear')
        ])
        return model

    def update_target_network(self):
        """Synchronize target network with main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        """Select an action using epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])
    
    def train(self, replay_buffer, batch_size):
        """Train the Q-network using experience from the replay buffer."""
        if len(replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Compute target Q-values
        next_qs = self.target_model.predict(next_states, verbose=0)
        max_next_qs = np.max(next_qs, axis=1)
        target_qs = rewards + (1 - dones) * self.gamma * max_next_qs
        
        # Train Q-network
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            indices = tf.stack([tf.range(batch_size), actions], axis=1)
            selected_qs = tf.gather_nd(q_values, indices)
            loss = self.loss_function(target_qs, selected_qs)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)#

    def add(self, experience):
        self.buffer.append(experience) 

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return map(np.array, zip(*batch))  # Returns states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)