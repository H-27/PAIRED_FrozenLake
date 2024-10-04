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

def run_DQN_PAIRED(episodes, map_dims, continue_training, continue_on_episode = 0):
    # load weights if to continue training
    map_shape = (1, 3, map_dims[0], map_dims[1])  # Assuming a 10x10 map with 3 channels
    direction_shape = (1, 4)
    position_shape = (1, map_dims[0]* map_dims[1])
    protagonist_network = networks.Actor_Network(4)
    dummy_map = tf.random.normal(map_shape)
    dummy_direction = tf.random.normal(direction_shape)
    dummy_position = tf.random.normal(position_shape)

    if (continue_training):
        #protagonist_network.build((None, 3, map_dims[0], map_dims[1]), (None, map_dims[0]* map_dims[1]), (None,4))
        protagonist_network(dummy_map, dummy_direction, dummy_position)
        helper.load_model(network=protagonist_network, filepath='DQN_PAIRED/protagonist')
    antagonist_network = networks.Actor_Network(4)
    if (continue_training):
        antagonist_network(dummy_map, dummy_direction, dummy_position)
        helper.load_model(network=antagonist_network, filepath='DQN_PAIRED/antagonist')
    adversary_network = networks.Adversary_Network(map_dims, True, True)
    if (continue_training):
        timestep_shape = (1, 1)
        dummy_timestep = tf.random.normal(timestep_shape)
        adversary_network(dummy_map, dummy_timestep, dummy_position)
        helper.load_model(network=adversary_network, filepath='DQN_PAIRED/adversary')
    # initialize agents
    agent_alpha = 0.001
    agent_gamma = 0.7
    agent_epsilon = 0.3339
    agent_epsilon_decay = 0.0006678
    agent_memory_size = 100000
    agent_training_batch_size = 64
    protagonist = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon, epsilon_decay=agent_epsilon_decay,
                                   n_actions=4, map_dims=map_dims,
                                   memory_size=agent_memory_size, training_batch_size=agent_training_batch_size,
                                   network=protagonist_network,
                                   use_direction=False, use_position=False)

    antagonist_network = networks.Actor_Network(4)
    antagonist = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon, epsilon_decay=agent_epsilon_decay,
                                  n_actions=4, map_dims=map_dims,
                                  memory_size=agent_memory_size, training_batch_size=agent_training_batch_size, network=antagonist_network,
                                  use_direction=False, use_position=False)
    # initialize_adversary
    adversary_alpha = 0.001
    adversary_gamma = 0.995
    adversary_epsilon = 0.5
    adversary_memory_size = 100000
    adversary_batch_size = 64
    adversary = adversaries.DQN_Adversary(alpha=adversary_alpha, gamma=adversary_gamma, epsilon=adversary_epsilon,
                                          adversary_memory_size=adversary_memory_size, adversary_batch_size=adversary_batch_size,
                                          adversary_network=adversary_network, map_width=map_dims[1], map_height=map_dims[0])
    if continue_training:
        for _ in range(continue_on_episode):
            adversary.epsilon_decay()

    # remaining values
    max_steps = (map_dims[0]*map_dims[1]) * 5
    agent_max_episodes = 5001

    # train writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'DQN_PAIRED/logs/fit/' + current_time
    train_writer = tf.summary.create_file_writer(train_log_dir)

    #metrics
    protagonist_steps = []
    antagonist_steps = []
    protagonist_win_ratio = []
    antagonist_win_ratio = []
    losses = []
    maps = []

    start_value = continue_on_episode if continue_training else 0
    for e in range(start_value, episodes):
        # create map and get values
        adv_map = adversary.create_map()
        adv_map = envs.Env_map(np.zeros((3, map_dims[0], map_dims[1]))).deone_hot_map_with_start(adv_map)
        shortest_path, shortest_path_length = helper.get_shortest_possible_length(adv_map)
        if shortest_path == None:
            solvable = False
        else:
            solvable = True
        tf.summary.scalar("solvable", solvable, step=0)
        num_blocks = helper.get_num_blocks(adv_map)
        env_map = envs.Env_map(np.zeros((3, map_dims[0], map_dims[0]))).one_hot_map(adv_map)
        maps.append(adv_map)
        #training protagonist
        print('training protagonist')
        pro_losses, pro_win_ratio, pro_shaped_episode_reward, pro_episode_reward, pro_steps_per_episode, pro_episodes_until_convergence = adversary.collect_trajectories(env_map, protagonist, agent_max_episodes)
        #training antagonist
        print('training antagonist')
        ant_losses, ant_win_ratio, ant_shaped_episode_reward, ant_episode_reward, ant_steps_per_episode, ant_episodes_until_convergence = adversary.collect_trajectories(env_map, antagonist, agent_max_episodes)

        # save agents after training
        helper.save_model(protagonist_network, 'DQN_PAIRED/protagonist')
        helper.save_model(antagonist_network, 'DQN_PAIRED/antagonist')

        regret = np.max(ant_episode_reward) - np.mean(pro_episode_reward)

        protagonist_win_ratio.append(pro_win_ratio)
        antagonist_win_ratio.append(ant_win_ratio)
        protagonist_steps.append(np.mean(pro_steps_per_episode))
        antagonist_steps.append(np.mean(ant_steps_per_episode))
        with train_writer.as_default():
            tf.summary.scalar('shortest_path_length', shortest_path_length, step=e)
            tf.summary.scalar('num_blocks', num_blocks, step=e)
            tf.summary.scalar('pro_losses', np.mean(pro_losses), step=e)
            tf.summary.scalar('pro_win_ratio', pro_win_ratio, step=e)
            tf.summary.scalar('pro_rewards', np.mean(pro_episode_reward), step=e)
            tf.summary.scalar('pro_shaped_rewards', np.mean(pro_shaped_episode_reward), step=e)
            tf.summary.scalar('pro_steps', np.mean(protagonist_steps), step=e)
            tf.summary.scalar('ant_losses', np.mean(ant_losses), step=e)
            tf.summary.scalar('ant_win_ratio', ant_win_ratio, step=e)
            tf.summary.scalar('ant_rewards', np.mean(ant_episode_reward), step=e)
            tf.summary.scalar('ant_shaped_episode_reward', np.mean(ant_shaped_episode_reward), step=e)
            tf.summary.scalar('ant_steps', np.mean(antagonist_steps), step=e)
            tf.summary.scalar('regret', regret, step=e)


            #tf.summary.scalar('value', value, step=e)

        # could use regret with reward function to get closer to target or
        # use original rewards and negative reward if all are 0 i.e. no wins
        if (pro_win_ratio == 0 and ant_win_ratio == 0): regret = -0.0001
        loss = adversary.train(regret)
        losses.append(loss)
        adversary.epsilon_decay()
        # reset agent epsilon
        protagonist.epsilon = agent_epsilon
        antagonist.epsilon = agent_epsilon
        # save adversary after training
        helper.save_model(adversary_network, 'DQN_PAIRED/adversary')
        print(f'Episode: {e}')
        save_episode(e)
        save_tensorboard_name()
        print(f'regret: {regret}')
    print('tensorboard --logdir=' + train_log_dir)

def load_episode():
    with open("DQN_PAIRED-episode_value.txt", "r") as f:
        string = f.read()
        value = int(string)
    return value
def save_episode(value):
    with open("DQN_PAIRED-episode_value.txt", "w") as f:
        f.write(str(value))

def save_tensorboard_name():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'DQN_PAIRED/logs/fit/' + current_time
    with open("save_DQN_PAIRED-tensorboard_name.txt", "w") as f:
        f.write(str(train_log_dir))

if __name__ == '__main__':
    episodes= 500000
    map_dims = (10,10)
    continue_training = True
    if continue_training:
        continue_on_episode = load_episode()
        continue_on_episode += 1  # as saved episode should be complete
    else:
        continue_on_episode = 0
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    with tf.device('/GPU:0'):
        run_DQN_PAIRED(episodes, map_dims, continue_training, continue_on_episode)


