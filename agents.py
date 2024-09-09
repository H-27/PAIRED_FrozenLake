import buffers
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Reinforce_Agent:
    def __init__(self, alpha, gamma, n_actions, network):
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions
        self.buffer = buffers.Buffer()
        self.network = network
        self.optimizer = tf.keras.optimizers.Adam(alpha)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        #state = np.expand_dims(state, -1)
        probabilities = self.network(state)
        action_probabilities = tfp.distributions.Categorical(probs = probabilities)
        action = action_probabilities.sample()
        return action.numpy()[0], probabilities

    def train(self):
        actions = tf.convert_to_tensor(self.buffer.actions, dtype=tf.float32)
        g = self.buffer.calculate_returns(self.gamma)

        with tf.GradientTape() as tape:
          loss = 0
          for i, (g,state) in enumerate(zip(g, self.buffer.states)):
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            #state = tf.expand_dims(state,-1)
            probs = self.network(state)
            action_probs = tfp.distributions.Categorical(probs = probs)
            log_prob = action_probs.log_prob(actions[i])
            loss += -g * tf.squeeze(log_prob)
          gradient = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.network.trainable_variables))

        self.buffer.reset()
        loss = loss.numpy()

        return loss

class DQN_Agent():

    def __init__(self, alpha, gamma, epsilon, n_actions, map_dims, memory_size, training_batch_size, network):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.batch_size = training_batch_size
        self.buffer = buffers.DQN_Buffer(memory_size, map_dims[0], map_dims[1])
        self.network = network
        self.optimizer = tf.keras.optimizers.Adam(alpha)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def choose_action(self, observation):
        actions = range(self.n_actions)
        log_probabilities = np.zeros((1,4))[0]
        probabilities = np.zeros((1,4))
        action_probabilities = np.zeros((1,4))[0]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(actions)
        else:
            observation = tf.convert_to_tensor(observation, dtype=tf.float32)
            observation = tf.expand_dims(observation,0)
            probabilities = self.network(observation)
            action_probabilities = tfp.distributions.Categorical(probs=probabilities)
            action = action_probabilities.sample()
            log_probabilities = action_probabilities.log_prob(action)
            log_probabilities = log_probabilities.numpy()[0]
            action = action.numpy()[0]
        return action, probabilities

    def train(self):
        if (self.buffer.memory_counter < self.batch_size):
            return 0
        losses = []
        old_states, new_states, actions, rewards, dones = self.buffer.create_batch(self.batch_size)

        with tf.GradientTape() as tape:

            for batch_index in range(len(actions)):

                old_state = tf.convert_to_tensor(old_states[batch_index], dtype=tf.float32)
                old_state = tf.expand_dims(old_state,0)
                new_state = tf.convert_to_tensor(new_states[batch_index], dtype=tf.float32)
                new_state = tf.expand_dims(new_state,0)
                action = tf.convert_to_tensor(actions[batch_index], dtype=tf.float32)
                reward = tf.convert_to_tensor(rewards[batch_index], dtype=tf.float32)
                done = tf.convert_to_tensor(dones[batch_index], dtype=tf.float32)

                old_q_probs = self.network(old_state)
                old_q_value = old_q_probs.numpy()[0][int(action.numpy())]

                next_state_probs = self.network(new_state, training=False)
                next_state_value = tf.reduce_max(next_state_probs)

                q_target = reward + (self.gamma * next_state_value) * (1 - done)

                loss = tf.square(q_target - old_q_value) * 0.5
                loss = tf.reduce_mean(loss)
                #loss = self.loss_function(q_target, old_q_value)

                losses.append(loss)

            grads = tape.gradient(losses, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return np.mean(losses)

    def train_on_stack(self):
        if (self.buffer.memory_counter < self.batch_size):
            return 0
        old_states, new_states, actions, rewards, dones = self.buffer.create_batch(self.batch_size)
        with tf.GradientTape() as tape:

            target_q = self.network(tf.convert_to_tensor(new_states, dtype=tf.float32))
            chosen_action = tf.argmax(target_q, axis=1) # correct axis?
            target_value = tf.reduce_sum(tf.one_hot(chosen_action, self.n_actions) * target_q, axis=1)
            target_value = (1-dones) * self.gamma * target_value + rewards

            main_q = self.network(tf.convert_to_tensor(old_states, dtype=tf.float32))
            main_value = tf.reduce_sum(tf.one_hot(actions, self.n_actions) * main_q, axis=1)

            loss = tf.square(main_value - target_value) * 0.5
            loss = tf.reduce_mean(loss)


            grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return loss.numpy()

    def epsilon_decay(self, epsilon_decay = 0.00066):
        self.epsilon = max(self.epsilon - epsilon_decay, 0)

class PPO_Agent:
    def __init__(self, alpha, gamma, epsilon, n_actions, actor, critic, batch_size):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.buffer = buffers.PPO_Buffer(batch_size)
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = tf.keras.optimizers.Adam(alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(alpha)


    def choose_action(self, observation):
        observation = tf.convert_to_tensor(observation)
        observation = tf.expand_dims(observation,0)
        probs = self.actor(observation)
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_probabilities = action_probabilities.log_prob(action)

        value = self.critic(observation)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_probability = log_probabilities.numpy()[0]

        return action, value, log_probability, probs

    def train(self):
        advantages = self.buffer.calculate_advantages(self.gamma)
        advantages = np.array(advantages)
        #print(advantages)
        batches = self.buffer.generate_batches()
        losses = []
        actor_losses = []
        critic_losses = []
        for batch in batches:
            states = tf.convert_to_tensor(np.array(self.buffer.states)[batch], dtype=tf.float32)
            old_probabilities = tf.convert_to_tensor(np.array(self.buffer.probs)[batch], dtype=tf.float32)
            old_probabilities = tf.expand_dims(old_probabilities, -1)
            actions = tf.convert_to_tensor(np.array(self.buffer.actions)[batch], dtype=tf.float32)
            actions = tf.expand_dims(actions, -1)
            with tf.GradientTape(persistent=True) as tape:
                #create probabilties and values for saved states
                probs = self.actor(states)
                new_probabilities = tfp.distributions.Categorical(probs=probs)
                new_probabilities = new_probabilities.log_prob(actions)
                new_values = self.critic(states)
                #new_values = tf.squeeze(new_values, 1)

                actor_loss, critic_loss = self.calculate_losses(new_probabilities, old_probabilities, new_values, advantages, batch)

                loss = actor_loss + 0.5 * critic_loss
                losses.append(loss)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.buffer.reset()
        if(np.isnan(np.mean(losses))):
            print('No training yet.')
        return np.mean(losses), np.mean(actor_losses), np.mean(critic_losses)

    def calculate_losses(self, new_probabilities, old_probabilities, new_values, advantages, batch):
        # create weighted and clipped probabilities
        probability_ratio = tf.math.exp(new_probabilities - old_probabilities)
        weighted_probabilities = advantages[batch] * probability_ratio
        clipped_probabilities = tf.clip_by_value(probability_ratio, 1 - self.epsilon, 1 + self.epsilon)
        weighted_clipped_probabilities = clipped_probabilities * probability_ratio
        batched_advantages = np.expand_dims(advantages[batch], -1)
        batched_values = np.array(self.buffer.values)[batch]
        returns = batched_advantages + batched_values
        returns = tf.convert_to_tensor(returns)

        actor_loss = tf.minimum(weighted_probabilities, weighted_clipped_probabilities)
        actor_loss = tf.reduce_mean(actor_loss)

        critic_loss = tf.square(returns - new_values) * 0.5
        # critic_loss = tf.keras.losses.MeanSquaredError(returns, new_values)
        critic_loss = tf.reduce_mean(critic_loss)

        return actor_loss, critic_loss

class AC_Agent:
    def __init__(self, alpha, gamma, epsilon, n_actions, actor, critic, batch_size):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.buffer = buffers.PPO_Buffer(batch_size)
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = tf.keras.optimizers.Adam(alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(alpha)


    def choose_action(self, observation):
        observation = tf.convert_to_tensor(observation)
        observation = tf.expand_dims(observation,0)
        probs = self.actor(observation)
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_probabilities = action_probabilities.log_prob(action)

        value = self.critic(observation)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_probability = log_probabilities.numpy()[0]

        return action, value, log_probability

    def train(self):
        advantages = self.buffer.calculate_advantages(self.gamma)
        advantages = np.array(advantages)
        batches = self.buffer.generate_batches()
        losses = []
        actor_losses = []
        critic_losses = []
        for batch in batches:
            states = tf.convert_to_tensor(np.array(self.buffer.states)[batch], dtype=tf.float32)
            old_probabilities = tf.convert_to_tensor(np.array(self.buffer.probs)[batch], dtype=tf.float32)
            old_probabilities = tf.expand_dims(old_probabilities, -1)
            actions = tf.convert_to_tensor(np.array(self.buffer.actions)[batch], dtype=tf.float32)
            actions = tf.expand_dims(actions, -1)
            with tf.GradientTape(persistent=True) as tape:
                #create probabilties and values for saved states
                probs = self.actor(states)
                new_probabilities = tfp.distributions.Categorical(probs=probs)
                new_probabilities = new_probabilities.log_prob(actions)
                new_values = self.critic(states)
                #new_values = tf.squeeze(new_values, 1)

                actor_loss, critic_loss = self.calculate_losses(new_probabilities, old_probabilities, new_values, advantages, batch)

                loss = actor_loss + 0.5 * critic_loss
                losses.append(loss)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.buffer.reset()
        return np.mean(losses), np.mean(actor_losses), np.mean(critic_losses)

    def calculate_losses(self, new_probabilities, old_probabilities, new_values, advantages, batch):
        # create weighted and clipped probabilities
        probability_ratio = tf.math.exp(new_probabilities - old_probabilities)
        weighted_probabilities = advantages[batch] * probability_ratio
        clipped_probabilities = tf.clip_by_value(probability_ratio, 1 - self.epsilon, 1 + self.epsilon)
        weighted_clipped_probabilities = clipped_probabilities * probability_ratio
        batched_advantages = np.expand_dims(advantages[batch], -1)
        batched_values = np.array(self.buffer.values)[batch]
        returns = batched_advantages + batched_values
        returns = tf.convert_to_tensor(returns)

        actor_loss = tf.minimum(weighted_probabilities, weighted_clipped_probabilities)
        actor_loss = tf.reduce_mean(actor_loss)

        critic_loss = tf.square(returns - new_values) * 0.5
        # critic_loss = tf.keras.losses.MeanSquaredError(returns, new_values)
        critic_loss = tf.reduce_mean(critic_loss)

        return actor_loss, critic_loss