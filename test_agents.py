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


if __name__ == '__main__':
    # create map and load agents
    map_dims = (10, 10)
    map_shape = (1, 3, map_dims[0], map_dims[1])  # Assuming a 10x10 map with 3 channels
    direction_shape = (1, 4)
    position_shape = (1, map_dims[0] * map_dims[1])
    dummy_map = tf.random.normal(map_shape)
    dummy_direction = tf.random.normal(direction_shape)
    dummy_position = tf.random.normal(position_shape)
    # load PAIRED agents
    paired_pro = networks.Actor_Network(4)
    paired_pro(dummy_map, dummy_direction, dummy_position)
    helper.load_model(network=paired_pro, filepath='DQN_PAIRED/protagonist')
    paired_ant = networks.Actor_Network(4)
    paired_ant(dummy_map, dummy_direction, dummy_position)
    helper.load_model(network=paired_ant, filepath='DQN_PAIRED/antagonist')
    # initialize agents
    agent_alpha = 0.001
    agent_gamma = 0.7
    agent_epsilon = 0.3339
    agent_epsilon_decay = 0.0006678
    agent_memory_size = 100000
    agent_training_batch_size = 64
    paired_pro_agent = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon,
                                   epsilon_decay=agent_epsilon_decay,
                                   n_actions=4, map_dims=map_dims,
                                   memory_size=agent_memory_size, training_batch_size=agent_training_batch_size,
                                   network=paired_pro,
                                   use_direction=False, use_position=False)

    paired_ant_agent = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon,
                                  epsilon_decay=agent_epsilon_decay,
                                  n_actions=4, map_dims=map_dims,
                                  memory_size=agent_memory_size, training_batch_size=agent_training_batch_size,
                                  network=paired_ant,
                                  use_direction=False, use_position=False)
    # load minimax agents
    minimax_pro = networks.Actor_Network(4)
    minimax_pro(dummy_map, dummy_direction, dummy_position)
    helper.load_model(network=minimax_pro, filepath='DQN_minimax/protagonist')
    minimax_ant = networks.Actor_Network(4)
    minimax_ant(dummy_map, dummy_direction, dummy_position)
    helper.load_model(network=minimax_ant, filepath='DQN_minimax/antagonist')
    minimax_pro_agent = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon,
                                        epsilon_decay=agent_epsilon_decay,
                                        n_actions=4, map_dims=map_dims,
                                        memory_size=agent_memory_size, training_batch_size=agent_training_batch_size,
                                        network=minimax_pro,
                                        use_direction=False, use_position=False)

    minimax_ant_agent = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon,
                                        epsilon_decay=agent_epsilon_decay,
                                        n_actions=4, map_dims=map_dims,
                                        memory_size=agent_memory_size, training_batch_size=agent_training_batch_size,
                                        network=minimax_ant,
                                        use_direction=False, use_position=False)
    # load domain randomization agents
    random_pro = networks.Actor_Network(4)
    random_pro(dummy_map, dummy_direction, dummy_position)
    helper.load_model(network=random_pro, filepath='DQN_random/protagonist')
    random_ant = networks.Actor_Network(4)
    random_ant(dummy_map, dummy_direction, dummy_position)
    helper.load_model(network=random_ant, filepath='DQN_random/antagonist')
    random_pro_agent = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon,
                                        epsilon_decay=agent_epsilon_decay,
                                        n_actions=4, map_dims=map_dims,
                                        memory_size=agent_memory_size, training_batch_size=agent_training_batch_size,
                                        network=random_pro,
                                        use_direction=False, use_position=False)

    random_ant_agent = agents.DQN_Agent(alpha=agent_alpha, gamma=agent_gamma, epsilon=agent_epsilon,
                                        epsilon_decay=agent_epsilon_decay,
                                        n_actions=4, map_dims=map_dims,
                                        memory_size=agent_memory_size, training_batch_size=agent_training_batch_size,
                                        network=random_ant,
                                        use_direction=False, use_position=False)
    maps_dict = {
        #"plain_map": maps.plain,
        "thirty_blocks_room": maps.thirty_blocks_room,
        "four_rooms_map": maps.four_rooms_map,
        "six_rooms_map": maps.six_rooms_map,
        "labyrinth": maps.labyrinth,
        "maze": maps.maze
    }
    # test paired
    for map_name, test_map in maps_dict.items():
        #create env
        map_dims = test_map.shape
        max_steps = ((map_dims[0] * map_dims[1]) * 5)
        adv_map = test_map
        env = gym.make('FrozenLake-v1', desc=adv_map, is_slippery=False, render_mode="rgb_array")
        shortest_path, shortest_path_length = helper.get_shortest_possible_length(adv_map)
        adv_map = envs.Env_map(np.zeros((3, adv_map.shape[0], adv_map.shape[1]))).one_hot_map(adv_map)
        env_map = envs.Env_map(adv_map)
        paired_pro_performance, paired_pro_solved_path, paired_pro_returns = helper.evaluate_policy(env, env_map, paired_pro_agent, 'DQN_PAIRED_pro' + map_name)
        print(f'Protagonist Performance in {map_name}: Performance = {paired_pro_performance}, Solved Path Length = {paired_pro_solved_path}, Shortest Path Length = {shortest_path_length}, Returns = {paired_pro_returns}')

        #paired_ant_performance, paired_ant_solved_path, paired_ant_returns = helper.evaluate_policy(env, env_map, paired_ant_agent, 'DQN_PAIRED_ant' + map_name)
        #print(f'Antagonist Performance in {map_name}: Performance = {paired_ant_performance}, Solved Path Length = {paired_ant_solved_path}, Shortest Path Length = {shortest_path_length} , Returns = {paired_ant_returns}')

    # test minimax
    for map_name, test_map in maps_dict.items():
        #create env
        map_dims = test_map.shape
        max_steps = ((map_dims[0] * map_dims[1]) * 5)
        adv_map = test_map
        env = gym.make('FrozenLake-v1', desc=adv_map, is_slippery=False, render_mode="rgb_array")
        shortest_path, shortest_path_length = helper.get_shortest_possible_length(adv_map)
        adv_map = envs.Env_map(np.zeros((3, adv_map.shape[0], adv_map.shape[1]))).one_hot_map(adv_map)
        env_map = envs.Env_map(adv_map)
        paired_pro_performance, paired_pro_solved_path, paired_pro_returns = helper.evaluate_policy(env, env_map, minimax_pro_agent, 'DQN_minimax_pro' + map_name)
        print(
            f'Minimax Protagonist Performance in {map_name}: Performance = {paired_pro_performance}, Solved Path Length = {paired_pro_solved_path}, Returns = {paired_pro_returns}, Shortest Path Length = {shortest_path_length}')

        # Evaluate the minimax antagonist
        #paired_ant_performance, paired_ant_solved_path, paired_ant_returns = helper.evaluate_policy(env, env_map,
        #                                                                                            minimax_ant,
        #                                                                                           'DQN_minimax_ant' + map_name)
        #print(
        #    f'Minimax Antagonist Performance in {map_name}: Performance = {paired_ant_performance}, Solved Path Length = {paired_ant_solved_path}, Returns = {paired_ant_returns}, Shortest Path Length = {shortest_path_length}')

    #test random
    for map_name, test_map in maps_dict.items():
        #create env
        map_dims = test_map.shape
        max_steps = ((map_dims[0] * map_dims[1]) * 5)
        adv_map = test_map
        env = gym.make('FrozenLake-v1', desc=adv_map, is_slippery=False, render_mode="rgb_array")
        shortest_path, shortest_path_length = helper.get_shortest_possible_length(adv_map)
        adv_map = envs.Env_map(np.zeros((3, adv_map.shape[0], adv_map.shape[1]))).one_hot_map(adv_map)
        env_map = envs.Env_map(adv_map)
        paired_pro_performance, paired_pro_solved_path, paired_pro_returns = helper.evaluate_policy(env, env_map, random_pro_agent, 'DQN_random_pro' + map_name)
        print(
            f'Random Protagonist Performance in {map_name}: Performance = {paired_pro_performance}, Solved Path Length = {paired_pro_solved_path}, Returns = {paired_pro_returns}, Shortest Path Length = {shortest_path_length}')

        # Evaluate the random policy antagonist
        #paired_ant_performance, paired_ant_solved_path, paired_ant_returns = helper.evaluate_policy(env, env_map,
        #                                                                                            random_ant,
        #                                                                                            'DQN_random_ant' + map_name)
        #print(
        #    f'Random Antagonist Performance in {map_name}: Performance = {paired_ant_performance}, Solved Path Length = {paired_ant_solved_path}, Returns = {paired_ant_returns}, Shortest Path Length = {shortest_path_length}')
