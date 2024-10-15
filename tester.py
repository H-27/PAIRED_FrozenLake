import networks
import helper
import agents
import adversaries
import envs
import buffers
import numpy as np
import tensorflow as tf
import nvidia.cudnn
import datetime
import os
import gymnasium as gym
import nvidia.cudnn
import maps
from buffers import Direction_DQN_Buffer
import unittest
from buffers import Buffer

class TestBuffer(unittest.TestCase):
    def test_store_and_retrieve(self):
        buffer = Buffer()

        # Store some transitions
        for _ in range(10):
            buffer.storeTransition(
                np.random.rand(3, 8, 8),
                np.random.randint(0, 4),
                np.random.randint(0, 64),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.randint(0, 2)
            )

        # Check the lengths of the stored attributes
        self.assertEqual(len(buffer.states), 10)
        self.assertEqual(len(buffer.directions), 10)
        self.assertEqual(len(buffer.positions), 10)
        self.assertEqual(len(buffer.actions), 10)
        self.assertEqual(len(buffer.rewards), 10)
        self.assertEqual(len(buffer.values), 10)
        self.assertEqual(len(buffer.probs), 10)
        self.assertEqual(len(buffer.dones), 10)

    def test_calculate_returns(self):
        buffer = Buffer()

        # Store some transitions with rewards
        buffer.storeTransition(np.random.rand(3, 8, 8), 0, 0, 0, 1, 0, 0, 0)
        buffer.storeTransition(np.random.rand(3, 8, 8), 0, 0, 0, 2, 0, 0, 0)
        buffer.storeTransition(np.random.rand(3, 8, 8), 0, 0, 0, 3, 0, 0, 0)

        # Calculate returns with gamma=0.9
        returns = buffer.calculate_returns(0.9)
        self.assertAlmostEqual(returns[0], 1 + 0.9 * 2 + 0.9**2 * 3, places=5)
        self.assertAlmostEqual(returns[1], 2 + 0.9 * 3, places=5)
        self.assertAlmostEqual(returns[2], 3, places=5)

    def test_save_and_load(self):
        buffer = Buffer()

        # Store some transitions
        for _ in range(10):
            buffer.storeTransition(
                np.random.rand(3, 8, 8),
                np.random.randint(0, 4),
                np.random.randint(0, 64),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.randint(0, 2)
            )

        # Save the buffer
        buffer.save("test_buffer.pkl")

        # Load the buffer and check its contents
        loaded_buffer = Buffer()
        loaded_buffer.load("test_buffer.pkl")

        self.assertEqual(len(buffer.states), len(loaded_buffer.states))
        self.assertEqual(len(buffer.directions), len(loaded_buffer.directions))
        self.assertEqual(len(buffer.positions), len(loaded_buffer.positions))
        self.assertEqual(len(buffer.actions), len(loaded_buffer.actions))
        self.assertEqual(len(buffer.rewards), len(loaded_buffer.rewards))
        self.assertEqual(len(buffer.values), len(loaded_buffer.values))
        self.assertEqual(len(buffer.probs), len(loaded_buffer.probs))
        self.assertEqual(len(buffer.dones), len(loaded_buffer.dones))
        np.testing.assert_array_equal(buffer.states, loaded_buffer.states)
        np.testing.assert_array_equal(buffer.directions, loaded_buffer.directions)
        np.testing.assert_array_equal(buffer.positions, loaded_buffer.positions)
        np.testing.assert_array_equal(buffer.actions, loaded_buffer.actions)
        np.testing.assert_array_equal(buffer.rewards, loaded_buffer.rewards)
        np.testing.assert_array_equal(buffer.values, loaded_buffer.values)
        np.testing.assert_array_equal(buffer.probs, loaded_buffer.probs)
        np.testing.assert_array_equal(buffer.dones, loaded_buffer.dones)

class TestDirectionDQNBuffer(unittest.TestCase):
    def test_store_and_retrieve(self):
        buffer = Direction_DQN_Buffer(100, 8, 8)

        # Store some transitions
        for _ in range(10):
            buffer.storeTransition(
                np.random.rand(3, 8, 8),
                np.random.rand(3, 8, 8),
                np.random.randint(0, 4),
                np.random.randint(0, 64),
                np.random.rand(),
                np.random.rand(),
                np.random.randint(0, 2)
            )

        # Retrieve a batch and check its shape
        batch = buffer.create_batch(5)
        self.assertEqual(batch[0].shape, (5, 3, 8, 8))
        self.assertEqual(batch[1].shape, (5, 3, 8, 8))
        self.assertEqual(batch[2].shape, (5, 4))
        self.assertEqual(batch[3].shape, (5, 64))
        self.assertEqual(batch[4].shape, (5,))
        self.assertEqual(batch[5].shape, (5,))
        self.assertEqual(batch[6].shape, (5,))

    def test_save_and_load(self):
        buffer = Direction_DQN_Buffer(100, 8, 8)

        # Store some transitions
        for _ in range(10):
            buffer.storeTransition(
                np.random.rand(3, 8, 8),
                np.random.rand(3, 8, 8),
                np.random.randint(0, 4),
                np.random.randint(0, 64),
                np.random.rand(),
                np.random.rand(),
                np.random.randint(0, 2)
            )

        # Save the buffer
        buffer.save("test_buffer.pkl")

        # Load the buffer and check its contents
        loaded_buffer = Direction_DQN_Buffer(100, 8, 8)
        loaded_buffer.load("test_buffer.pkl")

        self.assertEqual(buffer.memory_size, loaded_buffer.memory_size)
        self.assertEqual(buffer.memory_counter, loaded_buffer.memory_counter)
        np.testing.assert_array_equal(buffer.states, loaded_buffer.states)
        np.testing.assert_array_equal(buffer.directions, loaded_buffer.directions)
        np.testing.assert_array_equal(buffer.positions, loaded_buffer.positions)
        np.testing.assert_array_equal(buffer.new_states, loaded_buffer.new_states)
        np.testing.assert_array_equal(buffer.actions, loaded_buffer.actions)
        np.testing.assert_array_equal(buffer.rewards, loaded_buffer.rewards)
        np.testing.assert_array_equal(buffer.dones, loaded_buffer.dones)

def test_DQN_map(test_map, episodes, load_weights=None):
    #train writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'DQN_training/logs/fit/' + current_time
    train_writer = tf.summary.create_file_writer(train_log_dir)
    # params
    alpha = 0.0001
    gamma = 0.7
    epsilon = 0.3339
    sigma = 7
    loss_threshold = 0.05
    use_direction = False
    use_position = False
    print(f'Run with: use_direction={use_direction}_and_position={use_position}')
    map_dims = test_map.shape
    max_steps = ((map_dims[0] * map_dims[1]) * 5)
    adv_map = test_map
    env = gym.make('FrozenLake-v1', desc=adv_map, is_slippery=False, render_mode="rgb_array")
    print(helper.get_shortest_possible_length(adv_map))
    adv_map = envs.Env_map(np.zeros((3, adv_map.shape[0], adv_map.shape[1]))).one_hot_map(adv_map)
    env_map = envs.Env_map(adv_map)
    # network configurations
    network = networks.Actor_Network(4)
    if (load_weights != None):
        helper.load_model(network, load_weights)
    agent = agents.DQN_Agent(alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=0.0007574974429757832, n_actions=4, map_dims=map_dims, memory_size=100000,
                      training_batch_size=64, network=network, use_direction = use_direction, use_position=use_position)

    # metric lists
    scores = []
    episode_reward = []
    losses = []
    steps_per_episode = []
    epsilons = []
    total_timesteps = 0
    consecutive_episodes_below_threshold = 0
    rewards = []
    episode_steps = 0
    wins = 0
    wins_in_a_row = 0
    win_ratio = []
    first_win = 0


    # training loop
    position, _ = env.reset()
    direction = tf.one_hot(0, 4)
    _, old_state = env_map.map_step(position)
    position = tf.one_hot(position, map_dims[0] * map_dims[1])
    converged = False
    e = 0
    #for e in range(episodes):
    while (not converged):
        steps = 0
        done = False
        while not done:
            action, probs = agent.choose_action(old_state, direction, position)
            position, reward, done, second_flag, info = env.step(action)
            episode_reward.append(reward)
            _, new_state = env_map.map_step(position)
            position = tf.one_hot(position, (map_dims[0] * map_dims[1]))
            distance = helper.get_distance(new_state)
            if (distance == 0):
                win = True
                wins += 1
                if (first_win == 0):
                    first_win = e
            reward, distance_bonus = helper.reward_function(reward=reward, done=done, new_reward=2.1, punishment=-5.1,
                                                            step_penalty = -0.0042, distance=distance, scaling_factor=1, sigma=sigma)

            # add punishment to episode when exceeding max_steps
            if (steps > max_steps and not done):
                done = True
                reward += -10


            agent.buffer.storeTransition(old_state=old_state, new_state=new_state, direction=direction,
                                         position=position, action=action, reward=reward, done=done)
            old_state = new_state
            last_action = action
            episode_steps += 1
            steps += 1
            total_timesteps += 1

        # check and log win
        if (reward > 0):
            if (distance != 0):
                with train_writer.as_default():
                    tf.summary.scalar('false_wins', reward, step=e)

            # print('reward greater 0')

            wins_in_a_row += 1
            episode_steps = 0
        else:
            wins_in_a_row = 0

        # check performance before training
        '''if (wins_in_a_row >= 5):
            performance = helper.evaluate_policy(env, env_map, agent)
            print(f'PERFORMANCE: {performance}')
            if (performance >= 70):
                print("Agent has performed!")
                helper.save_model(network, 'DQN_training')
                print('tensorboard --logdir=.venv/' + train_log_dir)
                break'''

        # train agent
        loss = agent.train_on_stack()
        losses.append(loss)
        rewards.append(episode_reward)
        scores.append(helper.calculate_cumulative_discounted_reward(episode_reward, gamma))
        if len(losses) < 100:
            converged = False
        else:
            converged = all(abs(loss) < 0.15 for loss in losses[-100:])

        # epsilon calculation

        percentage = 0.5
        decay = 1 / (episodes / (1 / percentage))
        agent.epsilon_decay()
        epsilons.append(agent.epsilon)

        # save to tb
        with train_writer.as_default():
            tf.summary.scalar('Reward', reward, step=e)
            tf.summary.scalar('Loss', loss, step=e)
            tf.summary.scalar('Epsilon', agent.epsilon, step=e)

        # reset
        old_state, _ = env.reset()
        _, old_state = env_map.map_step(old_state)
        steps_per_episode.append(episode_steps)
        episode_steps = 0
        episode_reward = []



        if (e != 0 and e % 100 == 0):
            win_ratio.append(wins / 100)
            print(f'Episode: {e}')
            print(f'Epsilon: {agent.epsilon}')
            print(f'Last hundred episodes mean steps: {np.mean(steps_per_episode[-100:])}')
            print(f'Games won: {wins}%')
            with train_writer.as_default():
                tf.summary.scalar('Win ratio', wins / 100, step=e)
                tf.summary.scalar('Mean Steps', np.mean(steps_per_episode[-100:]), step=e)
            print(f'Score: {np.mean(scores[-100:])}')
            # print(losses)
            print(f'Loss: {np.mean(losses[-100:])}')
            #helper.show_DQN_probs(map_dims, agent, env_map)
            wins = 0
            lose = 0
            # save weights every 100 episodes
            helper.save_model(network, 'DQN_training')
        e += 1

    mean_rewards = []
    a = []
    for i in scores:
        a.append(i)
        mean_rewards.append(np.mean(a))

    if (not win_ratio):
        win_ratio.append(0)
    vis_win_ratio = (np.array(win_ratio), 'Win Ratio')
    vis_steps = (np.array(steps_per_episode), 'Steps')
    vis_losses = (np.array(losses), 'Losses')
    vis_rewards = (np.array(mean_rewards), 'Rewards')
    vis_epsilon = (np.array(epsilons), 'Epsilons')

    metrics_to_plot = [vis_win_ratio, vis_steps, vis_losses, vis_rewards, vis_epsilon]
    if (agent.epsilon <= 0):
        pass
    helper.visualize_and_save_metrics(metrics_to_plot, 'DQN_training')
    print('tensorboard --logdir=.venv/' + train_log_dir)
    print(total_timesteps)

def show_map(current_map):
    env = gym.make('FrozenLake-v1', desc=current_map, map_name="4x4", is_slippery=False, render_mode="human")
    env.reset()
    import time
    time.sleep(50)


def test_PPO_map(test_map, episodes, use_conv_block=False, load_weights=None):
    # train writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'DQN_training/logs/fit/' + current_time
    train_writer = tf.summary.create_file_writer(train_log_dir)
    # params
    alpha = 0.0003
    gamma = 0.75
    epsilon = 0.2
    episodes = 10000
    batch_size = 4
    training_size = 2048
    map_dims = (4, 4)
    # create map
    map_dims = test_map.shape
    max_steps = ((map_dims[0] * map_dims[1]) * 10)
    adv_map = test_map
    env = gym.make('FrozenLake-v1', desc=adv_map, is_slippery=False)#, render_mode="human")
    print(helper.get_shortest_possible_length(adv_map))
    adv_map = envs.Env_map(np.zeros((3, adv_map.shape[0], adv_map.shape[1]))).one_hot_map(adv_map)
    env_map = envs.Env_map(adv_map)
    # network configurations
    actor = networks.Conv_Network(4, map_dims)
    critic = networks.Critic_Network(1, map_dims)
    agent = agents.PPO_Agent(alpha=0.0001, gamma=gamma, epsilon=epsilon, n_actions=4, actor=actor, critic=critic, batch_size=512)
    agent.critic.trainable = True

    # metric lists
    scores = []
    losses = []
    steps_per_episode = []
    # training loop
    position, _ = env.reset()
    position = tf.one_hot(position, map_dims[0] * map_dims[1])
    direction = tf.one_hot(0, 4)
    _, old_state = env_map.map_step(position)
    rewards = []
    losses = []
    steps_per_episode = []
    win_ratio = []
    mean_losses = []
    steps_over_optimal = []
    episodes = 100000
    steps = 0
    wins = 0
    tb_steps = 0
    episode_reward = 0
    steps = 0

    for e in range(episodes):
        done = False
        episode_steps = 0
        max_step_counter = 0
        while not done:
            action, probs = agent.choose_action(old_state, position, direction)
            position, reward, done, second_flag, info = env.step(action)
            _, new_state = env_map.map_step(position)
            distance = helper.get_distance(new_state)
            reward, distance_bonus = helper.reward_function(reward=reward, done=done, new_reward=2.7, punishment=-4,
                                                            step_penalty=-0.0001, sigma=4, distance=distance, scaling_factor=1)
            #print(distance, distance_bonus)
            with train_writer.as_default():
                tf.summary.scalar('Bonus', distance_bonus, step=tb_steps)
            agent.buffer.storeTransition(old_state=old_state, new_state=new_state, direction=direction,
                                         position=position, action=action, reward=reward, done=done)
            old_state = new_state
            direction = tf.one_hot(action, 4)
            steps += 1
            tb_steps += 1
            episode_steps += 1
            max_step_counter += 1
            # add punishment to episode when exceeding max_steps
            if (max_step_counter > max_steps and not done):
                done = True
                reward += -10
                max_step_counter = 0
            episode_reward += reward
            if (steps % training_size == 0):
                loss, actor_loss, critic_loss = agent.train()
                losses.append(loss)
                with train_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=e)
        # reset
        old_state, _ = env.reset()
        _, old_state = env_map.map_step(old_state)
        steps_per_episode.append(episode_steps)
        mean_losses.append(np.mean(losses))
        rewards.append(episode_reward)
        scores.append(episode_reward)
        episode_reward = 0

        with train_writer.as_default():
            tf.summary.scalar('Reward', reward, step=e)
        if (reward > 0):
            wins += 1
            wins_in_a_row += 1
        else:
            wins_in_a_row = 0
        if (wins_in_a_row >= 5):
            #performance = helper.evaluate_policy(env, env_map, agent)

            episode_steps = 0
        if (e != 0 and e % 100 == 0):
            win_ratio.append(wins / 100)
            with train_writer.as_default():
                tf.summary.scalar('Win ratio', wins / 100, step=steps)
                tf.summary.scalar('Mean Steps', np.mean(steps_per_episode[-100:]), step=steps)
                tf.summary.scalar('Mean Rewards', np.mean(scores[-100:]), step=e)
            episode_steps = 0
            print(f'Episode: {e}')
            print(f'Mean last hundred steps: {np.mean(steps_per_episode[-100:])}')
            print(f'Games won: {wins}%')
            print(f'Score: {np.mean(scores[-100:])}')
            # print(losses)
            print(f'Loss: {np.mean(losses[-100:])}')
            helper.show_PPO_probs(map_dims, agent, env_map)
            helper.show_PPO_value(map_dims, agent, env_map)
            wins = 0
            lose = 0
            # save weights every 100 episodes
            #helper.save_model(network, 'PPO_training')


    mean_rewards = []
    a = []
    for i in scores:
        a.append(i)
        mean_rewards.append(np.mean(a))

    vis_win_ratio = (np.array([win_ratio]), 'Win Ratio')
    vis_steps = (np.array([steps_per_episode]), 'Steps')
    vis_losses = (np.array([losses]), 'Losses')
    vis_rewards = (np.array([mean_rewards]), 'Rewards')
    vis_epsilon = (np.array([]))
    metrics = [vis_win_ratio, vis_steps, vis_losses, vis_rewards, ]
    helper.visualize_and_safe([vis_win_ratio, vis_steps, vis_rewards, vis_losses], 'PPO_test', False)
    print('tensorboard --logdir=.venv/' + train_log_dir)
def test_buffer_save_and_load():
    # Create instances of both buffers
    dqn_buffer = Direction_DQN_Buffer(100, 8, 8)
    simple_buffer = Buffer()
    for i in range(20):
        # Demonstrate usage of `Direction_DQN_Buffer`
        dqn_buffer.storeTransition(
            np.random.rand(3, 8, 8),
            np.random.rand(3, 8, 8),
            np.random.randint(0, 4),
            np.random.randint(0, 64),
            np.random.rand(),
            np.random.rand(),
            np.random.randint(0, 2)
        )
        # Demonstrate usage of `Buffer`
        simple_buffer.storeTransition(
            np.random.rand(3, 8, 8),
            np.random.randint(0, 4),
            np.random.randint(0, 64),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.rand(),
            np.random.randint(0, 2)
        )
    batch = dqn_buffer.create_batch(5)
    print(dqn_buffer.actions)
    dqn_buffer.save("dqn_buffer.pkl")
    dqn_buffer.load("dqn_buffer.pkl")
    print(dqn_buffer.actions)

    returns = simple_buffer.calculate_returns(0.9)
    simple_buffer.save("simple_buffer.pkl")
    simple_buffer.load("simple_buffer.pkl")

    # Run unit tests
    unittest.main()


if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    #print(tf.config.list_physical_devices('GPU'))
    #show_map()
    '''env = gym.make('FrozenLake-v1', is_slippery=False)
    shortest_path, shortest_path_length = helper.get_shortest_possible_length(maps.map_ten_x_ten_test_training)
    print(shortest_path, shortest_path_length)'''
    '''os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.device('/GPU:0'):
        #current_map = maps.initial_test
        current_map = maps.map_ten_x_ten_test_training
        #test_PPO_map(current_map, 10000)
        test_DQN_map(current_map, 10001)'''
    #test_buffer_save_and_load()
    current_map = maps.map_ten_x_ten_test_training
    show_map(maps.thirty_blocks_room)
    from tensorflow.keras.utils import plot_model

'''    map_dims = (10, 10)
    map_shape = (1, 3, map_dims[0], map_dims[1])  # Assuming a 10x10 map with 3 channels
    direction_shape = (1, 4)
    position_shape = (1, map_dims[0] * map_dims[1])
    protagonist_network = networks.Actor_Network(4)
    dummy_map = tf.random.normal(map_shape)
    dummy_direction = tf.random.normal(direction_shape)
    dummy_position = tf.random.normal(position_shape)'''


    # protagonist_network.build((None, 3, map_dims[0], map_dims[1]), (None, map_dims[0]* map_dims[1]), (None,4))
    #protagonist_network(dummy_map, dummy_direction, dummy_position)
    #helper.load_model(network=protagonist_network, filepath='DQN_PAIRED/protagonist')

'''    plot_model(protagonist_network,

               to_file='keras_model_plot.png',

               show_shapes=True,

               show_layer_names=True)'''


