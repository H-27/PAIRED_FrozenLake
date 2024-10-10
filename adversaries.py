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
                reward, distance_bonus = helper.reward_function(reward, done)
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

    def __init__(self, alpha, gamma, epsilon, adversary_memory_size, adversary_batch_size, adversary_network, map_width, map_height, block_budget_multiplier=0.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.map_width = map_width
        self.map_height = map_height
        self.batch_size = adversary_batch_size
        self.n_possible_state_values = map_width * map_height
        self.n_placements = int(np.round(self.n_possible_state_values * 0.3) + 2)
        self.buffer = buffers.Direction_DQN_Buffer(self.n_placements, map_width, map_height)
        self.buffer.directions = np.zeros((adversary_memory_size, 1)) # timestep has different depth
        self.adversary_network = adversary_network
        self.optimizer = tf.keras.optimizers.Adam(alpha)

        self.initial_map = np.zeros((3, map_height, map_width))

    def epsilon_decay(self, epsilon_decay = 0.00002):
        self.epsilon = max(self.epsilon - epsilon_decay, 0)

    def choose_action(self, observation, timestep, rand_vec):
        action_range = range(self.map_height * self.map_width)

        rand = np.random.random()
        if rand < self.epsilon:
            log_probabilities = np.zeros((1, 4))[0]
            probabilities = np.zeros((1, 4))
            action_probabilities = np.zeros((1, 4))[0]
            action = np.random.choice(action_range)
        else:
            state = tf.convert_to_tensor(observation, dtype=tf.float32)
            state = tf.expand_dims(state, 0)
            #timestep = tf.expand_dims(timestep, 0)
            #rand_vec = tf.expand_dims(rand_vec, 0)
            probabilities = self.adversary_network(state, timestep, rand_vec)
            action_probabilities = tfp.distributions.Categorical(probs=probabilities)
            action = action_probabilities.sample()
            log_probabilities = action_probabilities.log_prob(action)
            log_probabilities = log_probabilities.numpy()[0]
            action = action.numpy()[0]
        return action, probabilities

    def create_map(self):
        old_map = np.copy(self.initial_map)
        rand_vec = np.random.standard_normal(self.map_width * self.map_height).astype('f')
        rand_vec = np.expand_dims(rand_vec, -1)
        rand_vec = np.transpose(rand_vec)
        rand_vec = tf.convert_to_tensor(rand_vec, dtype= tf.float32)
        done = False
        used_positions = []
        for i in range(self.n_placements):
            if(i == self.n_placements):
                done = True
            time_step = tf.convert_to_tensor([i], dtype= tf.float32)
            time_step = tf.expand_dims(time_step, -1)
            position, probs = self.choose_action(old_map, time_step, rand_vec)
            y, x = helper.calculate_coordinates(position, self.map_height)
            # insert position
            new_map = np.copy(old_map)
            if(i == 0):
                new_map[0][y][x] = 1
            # in case goal position is placed on start, choose random position
            elif(i == 1):
                if(position in used_positions):
                    remaining_positions = np.arange(self.n_possible_state_values)
                    remaining_positions = np.delete(remaining_positions, np.where(remaining_positions == used_positions[0]))
                    random_position = np.random.choice(remaining_positions)
                    used_positions.append(random_position)
                    rand_y, rand_x = helper.calculate_coordinates(random_position, self.map_height)
                    new_map[2][rand_y][rand_x] = 1
                else:
                    new_map[2][y][x] = 1
            else:
                if (position in used_positions):
                    pass
                else:
                    new_map[1][y][x] = 1
            self.buffer.storeTransition(old_state=old_map, new_state=new_map, direction = time_step, position = rand_vec, action = position, reward = 0, done = done)
            old_map = new_map
            used_positions.append(position)
        return new_map

    def collect_trajectories(self, adv_map, agent, agent_max_episodes):
        env_map = Env_map(adv_map)
        map_dims = (adv_map.shape[1], adv_map.shape[2])
        max_steps = map_dims[0] * map_dims[1]
        char_map = env_map.deone_hot_map_with_start(adv_map)
        squeezed_map = helper.squeeze_map(char_map)
        env = gym.make('FrozenLake-v1', desc=squeezed_map, is_slippery=False)#, render_mode="human")
        # metric lists
        rewards = []
        losses = []
        steps_per_episode = []
        episode_reward = []
        shaped_episode_reward = []
        cumulative_discounted_rewards = []
        shaped_cumulative_discounted_rewards = []
        # training loop
        position, _ = env.reset()
        direction = tf.one_hot(0, 4)
        _, old_state = env_map.map_step(position)
        position = tf.one_hot(position, map_dims[0] * map_dims[1])

        episode_steps = 0
        wins = 0
        first_win = 0
        converged = False
        e = 0
        while not converged:
        #for e in range(episodes):
            win = False
            steps = 0
            done = False
            while not done:
                action, probs = agent.choose_action(old_state, direction, position)
                direction = tf.one_hot(action, 4)
                position, reward, done, second_flag, info = env.step(action)
                if (reward == 1):
                    win = True
                    wins += 1
                    if (first_win == 0):
                        first_win = e
                _, new_state = env_map.map_step(position)
                position = tf.one_hot(position, (map_dims[0] * map_dims[1]))
                distance = helper.get_distance(new_state)#

                # get and add reward
                episode_reward.append(reward)
                reward, distance_bonus = helper.reward_function(reward=reward, done=done, new_reward=2.7,
                                                                punishment=-5,
                                                                step_penalty=-0.0001,
                                                                distance=distance, sigma=4,
                                                                scaling_factor=1)
                shaped_episode_reward.append(reward)

                # add punishment to episode when exceeding max_steps
                if (steps > max_steps and not done):
                    done = True
                    reward += -10
                # store step
                agent.buffer.storeTransition(old_state=old_state, new_state=new_state, direction=direction,
                                             position=position, action=action, reward=reward, done=done)
                # update values
                old_state = new_state
                episode_steps += 1
                steps += 1
                if (steps > max_steps and not done):
                    done = True
                    reward += -10
            # get loss
            loss = agent.train_on_stack()
            losses.append(loss)
            agent.epsilon_decay()

            # calculate cumulative discounted rewards
            cumulative_discounted_rewards.append(helper.calculate_cumulative_discounted_reward(episode_reward, agent.gamma))
            shaped_cumulative_discounted_rewards.append(helper.calculate_cumulative_discounted_reward(shaped_episode_reward, agent.gamma))
            episode_reward = []
            shaped_episode_reward = []

            # check for convergence
            if len(losses) < 100:
                converged = False
            else:
                converged = all(abs(loss) < 0.15 for loss in losses[-100:])
            if (e > agent_max_episodes):
                converged = True

            old_state, _ = env.reset()
            _, old_state = env_map.map_step(old_state)
            steps_per_episode.append(episode_steps)
            episode_steps = 0
            if (e != 0 and e % 500 == 0):
                print(f'Episode: {e}')
                print(f'Mean last hundred steps: {np.mean(steps_per_episode[-100:])}')
                print(f'Epsilon: {agent.epsilon}')
                print(f'Games won: {wins/5}%')
                print(f'Score: {np.mean(cumulative_discounted_rewards)}')
                print(f'Loss: {np.mean(losses)}')
                wins = 0
            e += 1
        episodes_until_convergence = e
        win_ratio = 0 if wins == 0 else wins/e
        return losses, win_ratio, shaped_cumulative_discounted_rewards, cumulative_discounted_rewards, steps_per_episode, episodes_until_convergence

    def calculate_regret(self, pro_rewards, ant_rewards, env):
        env_reward = np.max(ant_rewards) - np.mean(pro_rewards)
        env_reward += helper.compute_adversary_block_budget(np.max(ant_rewards), env)
        return env_reward
    def calculate_min_max(self, pro_rewards):
      return -np.mean(pro_rewards)


    def train(self, regret):
        self.buffer.rewards[-1] = regret
        if (self.buffer.memory_counter < self.batch_size):
            return 0
        old_states, new_states, time_steps, rand_vecs, actions, rewards, dones = self.buffer.create_batch(self.n_placements)
        with tf.GradientTape() as tape:
            new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
            old_states = tf.convert_to_tensor(old_states, dtype=tf.float32)
            time_steps = tf.convert_to_tensor(time_steps, dtype=tf.float32)
            rand_vecs = tf.convert_to_tensor(rand_vecs, dtype=tf.float32)
            target_q = self.adversary_network(new_states, time_steps, rand_vecs)
            chosen_action = tf.argmax(target_q, axis=1) # correct axis?
            target_value = tf.reduce_sum(tf.one_hot(chosen_action, self.n_possible_state_values) * target_q, axis=1)
            target_value = (1-dones) * self.gamma * target_value + rewards
            main_q = self.adversary_network(old_states, time_steps, rand_vecs)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.n_possible_state_values) * main_q, axis=1)

            loss = tf.square(main_value - target_value) * 0.5
            loss = tf.reduce_mean(loss)


            grads = tape.gradient(loss, self.adversary_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.adversary_network.trainable_variables))
        return loss.numpy()