import buffers
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Reinforce_Agent:
    def __init__(self, alpha, gamma, n_actions, network, use_direction=False, use_position=False):
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions
        self.buffer = buffers.Buffer()
        self.network = network
        self.optimizer = tf.keras.optimizers.Adam(alpha)
        self.use_direction = use_direction
        self.use_position = use_position

    def choose_action(self, observation, direction, position):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        direction = tf.convert_to_tensor([direction], dtype=tf.float32)
        position = tf.convert_to_tensor([position], dtype=tf.float32)
        #state = np.expand_dims(state, -1)
        probabilities = self.network(state, direction, position)
        action_probabilities = tfp.distributions.Categorical(probs = probabilities)
        action = action_probabilities.sample()
        return action.numpy()[0], probabilities

    def train(self):
        actions = tf.convert_to_tensor(self.buffer.actions, dtype=tf.float32)
        g = self.buffer.calculate_returns(self.gamma)

        with tf.GradientTape() as tape:
            loss = 0
            for i, (g,state, direction, position) in enumerate(zip(g, self.buffer.states, self.buffer.directions, self.buffer.positions)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                direction = tf.convert_to_tensor([direction], dtype=tf.float32)
                position = tf.convert_to_tensor([position], dtype=tf.float32)
                #state = tf.expand_dims(state,-1)
                probs = self.network(state, direction, position)
                action_probs = tfp.distributions.Categorical(probs = probs)
                log_prob = action_probs.log_prob(actions[i])
                loss += -g * tf.squeeze(log_prob)
            gradient = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.network.trainable_variables))

        self.buffer.reset()
        loss = loss.numpy()

        return loss

class DQN_Agent():

    def __init__(self, alpha, gamma, epsilon, epsilon_decay, n_actions, map_dims, memory_size, training_batch_size, network, use_direction, use_position):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_value = epsilon_decay
        self.n_actions = n_actions
        self.batch_size = training_batch_size
        self.buffer = buffers.Direction_DQN_Buffer(memory_size, map_dims[0], map_dims[1])
        self.network = network
        self.optimizer = tf.keras.optimizers.Adam(alpha)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.use_direction = use_direction
        self.use_position = use_position

    def choose_action(self, observation, direction, position):
        action_range = range(self.n_actions)

        rand = np.random.random()
        if rand < self.epsilon:
            log_probabilities = np.zeros((1, 4))[0]
            probabilities = np.zeros((1, 4))
            action_probabilities = np.zeros((1, 4))[0]
            action = np.random.choice(action_range)
        else:
            observation = tf.convert_to_tensor(observation, dtype=tf.float32)
            observation = tf.expand_dims(observation,0)
            direction = tf.expand_dims(direction, 0)
            position = tf.expand_dims(position, 0)
            probabilities = self.network(observation, direction, position, self.use_direction, self.use_position)
            action_probabilities = tfp.distributions.Categorical(probs=probabilities)
            action = action_probabilities.sample()
            log_probabilities = action_probabilities.log_prob(action)
            log_probabilities = log_probabilities.numpy()[0]
            action = action.numpy()[0]
        return action, probabilities

    def train_on_stack(self):
        if (self.buffer.memory_counter < self.batch_size):
            return 0
        old_states, new_states, directions, positions, actions, rewards, dones = self.buffer.create_batch(self.batch_size)
        with tf.GradientTape() as tape:
            directions = tf.convert_to_tensor(directions, dtype=tf.float32)
            positions = tf.convert_to_tensor(positions, dtype=tf.float32)
            target_q = self.network(tf.convert_to_tensor(new_states, dtype=tf.float32), directions, positions,
                                    self.use_direction, self.use_position)
            chosen_action = tf.argmax(target_q, axis=1) # correct axis?
            target_value = tf.reduce_sum(tf.one_hot(chosen_action, self.n_actions) * target_q, axis=1)
            target_value = (1-dones) * self.gamma * target_value + rewards

            main_q = self.network(tf.convert_to_tensor(old_states, dtype=tf.float32), directions, positions,
                                  self.use_direction, self.use_position)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.n_actions) * main_q, axis=1)

            loss = tf.square(main_value - target_value) * 0.5
            loss = tf.reduce_mean(loss)


            grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return loss.numpy()

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay_value, 0)


class PPO_Agent:
    def __init__(self, alpha, gamma, epsilon, n_actions, actor, critic, batch_size, use_direction, use_position):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.buffer = buffers.Vectorized_PPO_Buffer(batch_size)
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = tf.keras.optimizers.Adam(alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(alpha)
        self.use_direction = use_direction
        self.use_position = use_position


    def choose_action(self, observations, directions, positions):
        observations_tensor = tf.convert_to_tensor(observations)
        directions_tensor = tf.convert_to_tensor(directions)
        positions_tensor = tf.convert_to_tensor(positions)
        probs = self.actor(observations_tensor, directions_tensor, positions_tensor)
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_probabilities = action_probabilities.log_prob(action)

        value = self.critic(observations_tensor, directions_tensor, positions_tensor)

        action = action.numpy()
        value = value.numpy()
        probs = tf.math.log(probs)
        return action, value, probs

    def train(self):
        # calculate

        losses = []
        actor_losses = []
        critic_losses = []
        kls = []
        counter = 0
        states, directions, positions, actions, rewards, values, probs, dones, advantages, returns = self.buffer.flatten_buffer()
        batches = self.buffer.generate_batches(len(states))
        for batch in batches:
            # prep values
            actions, states, directions, positions, probs, advantages, returns = self.prep_buffer_values(batch, actions,
                                                                                                         states,
                                                                                                         directions,
                                                                                                         positions,
                                                                                                         probs,
                                                                                                         advantages,
                                                                                                         returns)
            old_probabilities = tf.gather(probs, actions, batch_dims=1)
            with tf.GradientTape(persistent=True) as tape:
                # create probabilties and values for saved states
                probs = self.actor(states, directions, positions)
                new_probabilities = tf.math.log(probs)
                new_probabilities = tf.gather(new_probabilities, actions, batch_dims=1)
                new_values = self.critic(states, directions, positions)
                # create weighted and clipped probabilities
                probability_ratio = tf.math.exp(new_probabilities - old_probabilities)
                weighted_probabilities = advantages * probability_ratio
                clipped_probabilities = tf.clip_by_value(probability_ratio, 1 - self.epsilon, 1 + self.epsilon)
                weighted_clipped_probabilities =  advantages * clipped_probabilities

                actor_loss = tf.minimum(weighted_probabilities, weighted_clipped_probabilities)
                actor_loss = -tf.reduce_mean(actor_loss)

                critic_loss = tf.square(returns - new_values)
                critic_loss = tf.reduce_mean(critic_loss)

                kl_divergence = ((probability_ratio - 1.0) - tf.math.log(probability_ratio)).numpy().mean()
                kls.append(kl_divergence)
                entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10),
                                         axis=-1)  # Add 1e-10 for numerical stability
                entropy = tf.reduce_mean(entropy)
                loss = actor_loss - entropy * 0.01 + critic_loss * 0.5
                losses.append(loss)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            states, directions, positions, actions, rewards, values, probs, dones, advantages, returns = self.buffer.flatten_buffer()

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            # print(actor_grads)
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            # print(critic_grads)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            counter += 1

        return np.mean(losses), np.mean(actor_losses), np.mean(critic_losses), np.mean(kls)


    def prep_buffer_values(self, batch, actions, states, directions, positions, probs, advantages, returns):
        # prep values
        actions = tf.convert_to_tensor(np.array(actions.copy())[batch], dtype=tf.int32)
        actions = tf.expand_dims(actions, -1)
        states = tf.convert_to_tensor(np.array(states.copy())[batch], dtype=tf.float32)
        directions = tf.convert_to_tensor(np.array(directions.copy())[batch], dtype=tf.float32)
        positions = tf.convert_to_tensor(np.array(positions.copy())[batch], dtype=tf.float32)
        probs = tf.convert_to_tensor(np.array(probs.copy())[batch], dtype=tf.float32)
        advantages = tf.convert_to_tensor(np.array(advantages).copy()[batch], dtype=tf.float32)
        advantages = tf.expand_dims(advantages, -1)
        returns = tf.convert_to_tensor(np.array(returns).copy()[batch], dtype=tf.float32)
        returns = tf.expand_dims(returns, -1)

        return actions, states, directions, positions, probs, advantages, returns

