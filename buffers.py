import numpy as np

class Buffer(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []

    def storeTransition(self, state, action, reward, value, probs, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(probs)
        self.dones.append(done)

    def reset(self):
        self.states = []
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


class PPO_Buffer():
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []
        self.batch_size = batch_size

    def storeTransition(self, state, action, reward, value, probs, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(probs)
        self.dones.append(done)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []

    def calculate_advantages(self, gamma, use_true_mask = False, normalize_over = 'minibatch'):
        a = np.zeros(len(self.rewards), dtype=np.float32)
        episode_indices, true_mask = self.get_episode_indices()
        if (use_true_mask):
            episode_indices = true_mask
        n_indices = len(episode_indices) - 1
        if (n_indices == 2):
            n_indices += 1
        for i in range(n_indices):
            if (episode_indices[i] == 0):
                episode_range = range(episode_indices[i], episode_indices[i + 1] + 1)
            else:
                episode_range = range(episode_indices[i] + 1, episode_indices[i + 1] + 1)
            for j in episode_range:
                discount = 1
                a_t = 0
                steps = 1
                for k in range(j, episode_indices[i + 1]+1):
                    current_reward = self.rewards[k]
                    current_value = self.values[k]
                    if(k + 1 == len(a)):
                        next_value = current_value
                    else:
                        next_value = self.values[k + 1]
                    current_done = self.dones[k]
                    done_flag = 1 - current_done
                    a_t += current_reward + (pow(gamma,steps) * (next_value - current_value)) * done_flag
                    discount *= gamma
                    steps += 1
                a[j] = a_t

            # normalize episode
            if (normalize_over == 'episode'):
                minibatch_mean = a[episode_range].mean()
                minibatch_std = (a[episode_range].std() + 1e-5)
                normalized_advantages = (a[episode_range] - minibatch_mean) / minibatch_std

        # normalize minibatch
        if (normalize_over == 'minibatch'):
            minibatch_mean = a.mean()
            minibatch_std = (a.std() + 1e-5)
            normalized_advantages = (a - minibatch_mean) / minibatch_std
            a = normalized_advantages


        return a

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return batches
        # np.array(self.states)[batches],\
        # np.array(self.actions)[batches],\
        # np.array(self.probs)[batches],\
        # np.array(self.values)[batches],\
        # np.array(self.rewards)[batches],\
        # np.array(self.dones)[batches]

    def get_episode_indices(self):
        mask = np.array(self.dones)
        rewards = np.array(self.rewards)
        mask_indices = np.where(mask == 1)[0]
        true_mask = []
        for i in mask_indices:
            if (rewards[i] > 0):
                true_mask.append(i)
        if (mask_indices[0] != 0):
            mask_indices = np.insert(mask_indices, 0, 0)
        if (len(true_mask) > 0 and true_mask[0] != 0):
            true_mask = np.insert(true_mask, 0, 0)
        if (len(mask) - 1 != mask_indices[-1]):
            mask_indices = np.append(mask_indices, len(mask) - 1)
        if (len(true_mask) > 0 and len(mask) - 1 != true_mask[-1]):
            true_mask = np.append(true_mask, len(mask) - 1)
        return mask_indices, true_mask