import numpy as np
import tensorflow as tf

class Buffer(object):

    def __init__(self):
        self.states = []
        self.directions = []
        self.positions = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []

    def storeTransition(self, state, direction, position, action, reward, value, probs, done):
        self.states.append(state)
        self.directions.append(direction)
        self.positions.append(position)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(probs)
        self.dones.append(done)

    def reset(self):
        self.states = []
        self.directions = []
        self.positions = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []

    def calculate_returns(self, gamma):
        g = np.zeros(len(self.rewards))
        for i in range(len(g)):
            g_sum = 0
            discount = 1
            for j in range(i, len(g)):
                g_sum += self.rewards[j] * discount
                discount *= gamma
            g[i] = g_sum
        return g

class DQN_Buffer():
    def __init__(self, memory_size, x, y):
        self.memory_size = memory_size
        self.memory_counter = 0

        self.states = np.zeros((self.memory_size, 3, x, y), dtype=np.float32)
        self.new_states = np.zeros((self.memory_size, 3, x, y), dtype=np.float32)
        self.actions = np.zeros(self.memory_size, dtype=np.float32)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.dones = np.zeros(self.memory_size, dtype=np.float32)

    def storeTransition(self, old_state, new_state, action, reward, done):
        # fill up from start when full
        index = self.memory_counter % self.memory_size
        self.states[index] = old_state
        self.new_states[index] = new_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.memory_counter += 1

    def create_batch(self, batch_size):
        batch = np.random.choice(min(self.memory_size, self.memory_counter), batch_size, replace=False)
        old_states = self.states[batch]
        new_states = self.new_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        dones = self.dones[batch]
        return old_states, new_states, actions, rewards, dones

class Direction_DQN_Buffer():
    def __init__(self, memory_size, x, y):
        self.memory_size = memory_size
        self.memory_counter = 0

        self.states = np.zeros((self.memory_size, 3, x, y), dtype=np.float32)
        self.directions = np.zeros((self.memory_size, 4))
        self.positions = np.zeros((self.memory_size, x*y))
        self.new_states = np.zeros((self.memory_size, 3, x, y), dtype=np.float32)
        self.actions = np.zeros(self.memory_size, dtype=np.float32)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.dones = np.zeros(self.memory_size, dtype=np.float32)

    def storeTransition(self, old_state, new_state, direction, position, action, reward, done):
        # fill up from start when full
        index = self.memory_counter % self.memory_size
        self.states[index] = old_state
        self.directions[index] = direction
        self.positions[index] = position
        self.new_states[index] = new_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.memory_counter += 1
    def create_batch(self, batch_size):
        batch = np.random.choice(min(self.memory_size, self.memory_counter), batch_size, replace=False)
        old_states = self.states[batch]
        new_states = self.new_states[batch]
        directions = self.directions[batch]
        positions = self.positions[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        dones = self.dones[batch]
        return old_states, new_states, directions, positions, actions, rewards, dones

class Vectorized_PPO_Buffer():
    def __init__(self, batch_size):
        self.states = []
        self.directions = []
        self.positions = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.batch_size = batch_size

    def storeTransition(self, state, direction, position, action, reward, value, probs, done):
        self.states.append(state)
        self.directions.append(direction)
        self.positions.append(position)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(probs)
        self.dones.append(done)

    def reset(self):
        self.states = []
        self.directions = []
        self.positions = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []

    def calculate_advantages(self, gamma):
        advantages = np.zeros((len(self.rewards), len(self.rewards[0])))
        returns = np.zeros((len(self.rewards), len(self.rewards[0])))
        # calculate for every env seperate
        self.values = np.squeeze(self.values)
        for env_id in range(len(self.rewards[0])):
            # get values
            rewards = np.array(self.rewards)[:, env_id]
            values = np.array(self.values)[:, env_id]
            probs = np.array(self.probs)[:, env_id]
            # create mask
            dones = np.array(self.dones)[:, env_id]
            mask = 1 - dones
            # remove done values

            values = values * mask
            current_advantage = 0
            current_return = 0

            for t in reversed(range(len(rewards))):
                '''if (t == len(rewards) - 1): # For the last step, there's no next state
                    next_value = 0  # Replace with critic estimation as found on blog
                    next_done = 0
                else:'''
                next_value = values[t + 1]
                next_done = dones[t + 1]
                if dones[t] == 1:
                    current_advantage = 0
                    current_return = 0
                    next_value = 0  # set to zero as value would be from following episode?
                # calc GAE
                delta = rewards[t] + gamma * next_value - values[t]
                current_advantage = delta + gamma * current_advantage

                advantages[t][env_id] = current_advantage
                # calc returns
                current_return = rewards[t] + gamma * current_return
                returns[t][env_id] = current_return

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        self.advantages = advantages
        self.returns = returns
        return advantages, returns

    def generate_batches(self, n_states):
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return batches

    # rolls the buffered envs out to have 1D lists
    def flatten_buffer(self):
        flat_states = []
        flat_directions = []
        flat_positions = []
        flat_actions = []
        flat_rewards = []
        flat_values = []
        flat_probs = []
        flat_dones = []
        flat_advantages = []
        flat_returns = []
        for env_id in range(len(self.rewards[0])):
            # get values
            states = np.array(self.states)[:, env_id]
            directions = np.array(self.directions)[:, env_id]
            positions = np.array(self.positions)[:, env_id]
            actions = np.array(self.actions)[:, env_id]
            rewards = np.array(self.rewards)[:, env_id]
            values = np.array(self.values)[:, env_id]
            probs = np.array(self.probs)[:, env_id]
            dones = np.array(self.dones)[:, env_id]
            advantages = np.array(self.advantages)[:, env_id]
            returns = np.array(self.returns)[:, env_id]
            # add values
            flat_states.extend(states)
            flat_directions.extend(directions)
            flat_positions.extend(positions)
            flat_actions.extend(actions)
            flat_rewards.extend(rewards)
            flat_values.extend(values)
            flat_probs.extend(probs)
            flat_dones.extend(dones)
            flat_advantages.extend(advantages)
            flat_returns.extend(returns)
        return flat_states, flat_directions, flat_positions, flat_actions, flat_rewards, flat_values, flat_probs, flat_dones, flat_advantages, flat_returns
