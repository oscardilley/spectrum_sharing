""" DQN_agent.py

Implementing an agent in Tensorflow to perform deep Q learning.  """

import tensorflow as tf
from collections import deque
import numpy as np
from itertools import product
import gymnasium as gym

class Agent:
    def __init__(self, cfg, observation_space, action_space):
        self.cfg = cfg
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_shape = list(self.observation_space.shape)[0]

        # Preprocessing actions
        if hasattr(self.action_space, 'spaces'):  # Tuple space
            # Enumerate all possible actions for discrete sub-spaces
            actions=[]
            sub_spaces=self.action_space.spaces
            if all(hasattr(sub_space, 'n') for sub_space in sub_spaces):
                for sub_space in sub_spaces:
                    if isinstance(sub_space, gym.spaces.MultiBinary):
                        num_actions = range(2**sub_space.n)
                        actions.append([np.array([int(bit) for bit in np.binary_repr(i, sub_space.n)], dtype=np.int8) for i in num_actions])
                    elif isinstance(sub_space, gym.spaces.Discrete):
                        actions.append(range(sub_space.start, sub_space.start + sub_space.n))
                    else:
                        raise ValueError("Unsupported Space.")
                    self.actions = list(product(*actions))
                    self.num_actions = len(self.actions)
            else:
                raise ValueError("Unsupported action space: contains non-discrete sub-spaces.")
        else:
                raise ValueError("Action space not yet supported.")

        # Hyperparameters
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
            tf.keras.layers.Input(shape=self.observation_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_actions, activation='linear')
        ])
        return model

    def update_target_network(self):
        """Synchronize target network with main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, observation):
        """Select an action using epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon: # decaying epsilon
            idx = np.random.choice(len(self.actions))
            print("Random Action.")
            return self.actions[idx], idx
        else:
            q_values = self.model.predict(observation[np.newaxis], verbose=2) # add extra axis for batch
            idx = np.argmax(q_values[0])
            print("Q Action.")
            return self.actions[idx], idx
    
    def train(self, replay_buffer, batch_size):
        """Train the Q-network using experience from the replay buffer."""
        if len(replay_buffer) < batch_size:
            return
        
        observations, actions, rewards, next_observations, terminateds = replay_buffer.sample(batch_size)

        # Compute target Q-values using the Bellman equation:
        next_qs = self.target_model.predict(next_observations, verbose=2)
        max_next_qs = np.max(next_qs, axis=1)
        target_qs = rewards + (1 - terminateds) * self.gamma * max_next_qs
        
        # Train Q-network
        with tf.GradientTape() as tape:
            q_values = self.model(observations, training=True)
            indices = tf.stack([tf.range(batch_size), actions], axis=1)
            selected_qs = tf.gather_nd(q_values, indices)
            loss = self.loss_function(target_qs, selected_qs)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience) 

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False) # random to break temporal correlations
        batch = [self.buffer[idx] for idx in indices]
        # Unzip the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
    
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)