import buffers
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import helper
from envs import Env_map
import gymnasium as gym

class Reinforce_Adversary():

    def __init__(self, alpha, gamma, adversary_network, map_width, map_height, block_budget_multiplier=0.3):
        self.alpha = alpha
        self.gamma = gamma
        self.map_width = map_width
        self.map_height = map_height
        self.n_possible_state_values = map_width * map_height
        self.buffer = buffers.Adversary_Buffer()
        self.adversary_network = adversary_network
        self.optimizer = tf.keras.optimizers.Adam(alpha)
        self.block_budget = int(np.round(self.n_possible_state_values * block_budget_multiplier) + 2)
        self.initial_map = np.zeros((3, map_height, map_width))

    def choose_action(self, observation, timestep, rand_vec):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probabilities = self.adversary_network(state, timestep, rand_vec)
        action_probabilities = tfp.distributions.Categorical(probs = probabilities)
        action = action_probabilities.sample()
        return action.numpy()[0], probabilities

    def create_map(self):
        old_map = np.copy(self.initial_map)
        rand_vec = np.random.standard_normal(self.block_budget).astype('f')
        rand_vec = np.expand_dims(rand_vec, -1)
        rand_vec = np.transpose(rand_vec)
        rand_vec = tf.convert_to_tensor(rand_vec)
        self.buffer.rand_vec = rand_vec
        done = False
        used_positions = []
        for i in range(self.block_budget):
            if(i == self.block_budget):
                done = True
            time_step = tf.convert_to_tensor([i], dtype= tf.float32)
            time_step = tf.expand_dims(time_step, -1)

            position, probs = self.choose_action(old_map, time_step, rand_vec)
            y, x = helper.calculate_coordinates(position, self.map_height)
            self.buffer.storeTransition(state=old_map, action=position, reward=0, value=0, probs=probs, done=done)
            # insert position
            new_map = np.copy(old_map)
            if(i == 0):
                new_map[2][y][x] = 1
            elif(i == 1):
                if(position in used_positions):
                    remaining_positions = np.arange(self.n_possible_state_values)
                    remaining_positions = np.delete(remaining_positions, np.where(remaining_positions == used_positions[0]))
                    random_position = np.random.choice(remaining_positions)
                    used_positions.append(random_position)
                    rand_y, rand_x = helper.calculate_coordinates(random_position, self.map_height)
                    new_map[0][rand_y][rand_x] = 1
                else:
                    new_map[0][y][x] = 1
            elif(position in used_positions):
                pass
            else:
                new_map[1][y][x] = 1
            old_map = new_map
            used_positions.append(position)

        return new_map

    def collect_trajectories(self, adv_map, agent, episodes, max_steps):
        env_map = Env_map(adv_map)
        char_map = env_map.deone_hot_map_with_start(adv_map)
        squeezed_map = helper.squeeze_map(char_map)
        env = gym.make('FrozenLake-v1', desc=squeezed_map, map_name="4x4", is_slippery=False, render_mode="human")
        # metric lists
        rewards = []
        losses = []
        steps_per_episode = []
        # training loop
        old_state, _ = env.reset()
        _, old_state = env_map.map_step(old_state)

        episode_steps = 0
        wins = 0
        lose = 0
        for e in range(episodes):
            done = False
            steps = 0
            while not done:
                action, probs = agent.choose_action(old_state)
                new_state, reward, done, second_flag, info = env.step(action)
                _, new_state = env_map.map_step(new_state)
                reward = helper.reward_function(reward, done)
                if (steps > max_steps and not done):
                    done = True
                    reward = -2
                    episode_steps = 0

                agent.buffer.storeTransition(state=old_state, action=action, reward=reward, value=0, probs=probs, done=done)
                old_state = new_state
                episode_steps += 1
                steps += 1

            loss = agent.train()
            losses.append(loss)

            old_state, _ = env.reset()
            _, old_state = env_map.map_step(old_state)
            steps_per_episode.append(episode_steps)
            rewards.append(reward)
            if (reward > 0):
                wins += 1
                episode_steps = 0
            else:
              lose += 1
            if (e != 0 and e % 250 == 0):

                print(f'Episode: {e}')
                print(f'Mean last hundred steps: {np.mean(steps_per_episode[-100:])}')
                print(f'Games won: {wins}%')
                print(f'Score: {np.mean(rewards)}')
                print(f'Loss: {np.mean(losses)}')
                wins = 0
                lose = 0
        return losses, wins/250, rewards, steps_per_episode

    def calculate_regret(self, pro_rewards, ant_rewards):
      return np.max(ant_rewards) - np.mean(pro_rewards)


    def train(self, regret):
        actions = tf.convert_to_tensor(self.buffer.actions, dtype=tf.float32)
        self.buffer.rewards[-1] = regret
        g = self.buffer.calculate_returns(self.gamma)

        with tf.GradientTape() as tape:
          loss = 0
          for i, (g,state) in enumerate(zip(g, self.buffer.states)):
            timestep = tf.convert_to_tensor([i], dtype=tf.float32)
            timestep = tf.expand_dims(timestep, -1)
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            #state = tf.expand_dims(state,-1)
            probs = self.adversary_network(state, timestep, self.buffer.rand_vec)
            action_probs = tfp.distributions.Categorical(probs = probs)
            log_prob = action_probs.log_prob(actions[i])
            loss += -g * tf.squeeze(log_prob)
          gradient = tape.gradient(loss, self.adversary_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.adversary_network.trainable_variables))

        self.buffer.reset()
        loss = loss.numpy()

        return loss

class DQN_Adversary():

    def __init__(self, alpha, gamma, adversary_network, map_width, map_height, block_budget_multiplier=0.3):
        self.alpha = alpha
        self.gamma = gamma
        self.map_width = map_width
        self.map_height = map_height
        self.n_possible_state_values = map_width * map_height
        self.buffer = buffers.Adversary_Buffer()
        self.adversary_network = adversary_network
        self.optimizer = tf.keras.optimizers.Adam(alpha)
        self.block_budget = int(np.round(self.n_possible_state_values * block_budget_multiplier) + 2)
        self.initial_map = np.zeros((3, map_height, map_width))

    def choose_action(self, observation, timestep, rand_vec):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probabilities = self.adversary_network(state, timestep, rand_vec)
        action_probabilities = tfp.distributions.Categorical(probs = probabilities)
        action = action_probabilities.sample()
        return action.numpy()[0], probabilities

    def create_map(self):
        old_map = np.copy(self.initial_map)
        rand_vec = np.random.standard_normal(self.block_budget).astype('f')
        rand_vec = np.expand_dims(rand_vec, -1)
        rand_vec = np.transpose(rand_vec)
        rand_vec = tf.convert_to_tensor(rand_vec)
        self.buffer.rand_vec = rand_vec
        done = False
        used_positions = []
        for i in range(self.block_budget):
            if(i == self.block_budget):
                done = True
            time_step = tf.convert_to_tensor([i], dtype= tf.float32)
            time_step = tf.expand_dims(time_step, -1)

            position, probs = self.choose_action(old_map, time_step, rand_vec)
            y, x = helper.calculate_coordinates(position, self.map_height)
            self.buffer.storeTransition(state=old_map, action=position, reward=0, value=0, probs=probs, done=done)
            # insert position
            new_map = np.copy(old_map)
            if(i == 0):
                new_map[2][y][x] = 1
            elif(i == 1):
                if(position in used_positions):
                    remaining_positions = np.arange(self.n_possible_state_values)
                    remaining_positions = np.delete(remaining_positions, np.where(remaining_positions == used_positions[0]))
                    random_position = np.random.choice(remaining_positions)
                    used_positions.append(random_position)
                    rand_y, rand_x = helper.calculate_coordinates(random_position, self.map_height)
                    new_map[0][rand_y][rand_x] = 1
                else:
                    new_map[0][y][x] = 1
            elif(position in used_positions):
                pass
            else:
                new_map[1][y][x] = 1
            old_map = new_map
            used_positions.append(position)

        return new_map

    def collect_trajectories(self, adv_map, agent, episodes, max_steps):
        env_map = Env_map(adv_map)
        char_map = env_map.deone_hot_map_with_start(adv_map)
        squeezed_map = helper.squeeze_map(char_map)
        env = gym.make('FrozenLake-v1', desc=squeezed_map, map_name="4x4", is_slippery=False, render_mode="human")
        # metric lists
        rewards = []
        losses = []
        steps_per_episode = []
        # training loop
        old_state, _ = env.reset()
        _, old_state = env_map.map_step(old_state)

        episode_steps = 0
        wins = 0
        lose = 0
        for e in range(episodes):
            done = False
            steps = 0
            while not done:
                action, probs = agent.choose_action(old_state)
                new_state, reward, done, second_flag, info = env.step(action)
                _, new_state = env_map.map_step(new_state)
                reward = helper.reward_function(reward, done)
                if (steps > max_steps and not done):
                    done = True
                    reward = -2
                    episode_steps = 0

                agent.buffer.storeTransition(state=old_state, action=action, reward=reward, value=0, probs=probs, done=done)
                old_state = new_state
                episode_steps += 1
                steps += 1

            loss = agent.train()
            losses.append(loss)

            old_state, _ = env.reset()
            _, old_state = env_map.map_step(old_state)
            steps_per_episode.append(episode_steps)
            rewards.append(reward)
            if (reward > 0):
                wins += 1
                episode_steps = 0
            else:
              lose += 1
            if (e != 0 and e % 250 == 0):

                print(f'Episode: {e}')
                print(f'Mean last hundred steps: {np.mean(steps_per_episode[-100:])}')
                print(f'Games won: {wins}%')
                print(f'Score: {np.mean(rewards)}')
                print(f'Loss: {np.mean(losses)}')
                wins = 0
                lose = 0
        return losses, wins/250, rewards, steps_per_episode

    def calculate_regret(self, pro_rewards, ant_rewards):
      return np.max(ant_rewards) - np.mean(pro_rewards)


    def train(self, regret):
        actions = tf.convert_to_tensor(self.buffer.actions, dtype=tf.float32)
        self.buffer.rewards[-1] = regret
        g = self.buffer.calculate_returns(self.gamma)

        with tf.GradientTape() as tape:
          loss = 0
          for i, (g,state) in enumerate(zip(g, self.buffer.states)):
            timestep = tf.convert_to_tensor([i], dtype=tf.float32)
            timestep = tf.expand_dims(timestep, -1)
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            #state = tf.expand_dims(state,-1)
            probs = self.adversary_network(state, timestep, self.buffer.rand_vec)
            action_probs = tfp.distributions.Categorical(probs = probs)
            log_prob = action_probs.log_prob(actions[i])
            loss += -g * tf.squeeze(log_prob)
          gradient = tape.gradient(loss, self.adversary_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.adversary_network.trainable_variables))

        self.buffer.reset()
        loss = loss.numpy()

        return loss