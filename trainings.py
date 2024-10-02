import networks
import helper
import agents
import adversaries
import envs
import buffers
import numpy as np
import tensorflow as tf
import nvidia.cudnn
import gymnasium as gym

def run_Reinforce_PAIRED(size, episodes):
    protagonist_name = 'Reinforce_Protagonist'
    antagonist_name = 'Reinforce_Antagonist'
    adversary_name = 'Reinforce_Adversary'
    map_dims=(size[0],size[1])
    max_steps = (size[0] * size[1]) * 10
    alpha = 0.0001
    gamma = 0.995
    adversary_network = networks.Adversary_Network(map_dims[1]* map_dims[0], map_dims)
    protagonist_network = networks.Conv_Network(4, map_dims)
    protagonist = agents.Reinforce_Agent(alpha=alpha, gamma=gamma, n_actions=4, network=protagonist_network)
    antagonist_network = networks.Conv_Network(4, map_dims)
    antagonist = agents.Reinforce_Agent(alpha=alpha, gamma=gamma, n_actions=4, network=antagonist_network)
    adversary = adversaries.Reinforce_Adversary(alpha, gamma, adversary_network, map_dims[1], map_dims[0], block_budget_multiplier=0.3)

    protagonist_steps = []
    antagonist_steps = []
    protagonist_win_ratio = []
    antagonist_win_ratio = []
    losses = []
    maps = []

    for e in range(episodes):

        new_map = adversary.create_map()
        env_map = envs.Env_map(new_map)
        char_map = env_map.deone_hot_map_with_start(new_map)
        maps.append(char_map)
        print(char_map)
        #env = gym.make('FrozenLake-v1', desc=squeezed_map, map_name="4x4", is_slippery=False, render_mode="human")
        pro_losses, pro_win_ratio, pro_rewards, pro_steps = adversary.collect_trajectories(new_map, protagonist, 5, max_steps)
        ant_losses, ant_win_ratio, ant_rewards, ant_steps = adversary.collect_trajectories(new_map, antagonist, 5, max_steps)

        regret = np.max(ant_rewards) - np.mean(pro_rewards)

        protagonist_win_ratio.append(pro_win_ratio)
        antagonist_win_ratio.append(ant_win_ratio)
        protagonist_steps.append(np.mean(pro_steps))
        antagonist_steps.append(np.mean(ant_steps))

        if(pro_win_ratio == 0 and ant_win_ratio == 0): regret = -0.0001
        loss = adversary.train(regret)
        losses.append(loss)

        print(f'Episode: {e}')
        if (np.mean(pro_steps) == 125.5 and np.mean(ant_steps) == 125.5):
            print('<<< first step always hole >>>')
        print(f'regret: {regret}')
        if (e != 0 and e % 100 == 0):
            pass
        print(f'pro_win_ratio: {pro_win_ratio}')
        print(f'ant_win_ratio: {ant_win_ratio}')
        print(f'np.mean(pro_steps): {np.mean(pro_steps)}')
        print(f'np.mean(ant_steps): {np.mean(ant_steps)}')
        print(f'np.mean(losses): {np.mean(losses)}')

def run_DQN_PAIRED(size, episodes):
    protagonist_name = 'Reinforce_Protagonist'
    antagonist_name = 'Reinforce_Antagonist'
    adversary_name = 'Reinforce_Adversary'
    map_dims=(size[0],size[1])
    max_steps = (size[0] * size[1]) * 10
    alpha = 0.0001
    gamma = 0.995
    adversary_network = networks.Adversary_Network(map_dims[1]* map_dims[0], map_dims)
    protagonist_network = networks.Actor_Network(4)
    protagonist = agents.DQN_Agent(alpha=alpha, gamma=gamma, epsilon=1, n_actions=4, map_dims=map_dims, memory_size=100000,
                      training_batch_size=64, network=protagonist_network)

    antagonist_network = networks.Actor_Network(4)
    antagonist = agents.DQN_Agent(alpha=alpha, gamma=gamma, epsilon=1, n_actions=4, map_dims=map_dims, memory_size=100000,
                      training_batch_size=64, network=antagonist_network)
    adversary = adversaries.DQN_Adversary(alpha, gamma, adversary_network, map_dims[1], map_dims[0], block_budget_multiplier=0.3)

    protagonist_steps = []
    antagonist_steps = []
    protagonist_win_ratio = []
    antagonist_win_ratio = []
    losses = []
    maps = []

    for e in range(episodes):

        new_map = adversary.create_map()
        env_map = envs.Env_map(new_map)
        char_map = env_map.deone_hot_map_with_start(new_map)
        maps.append(char_map)
        print(char_map)
        #env = gym.make('FrozenLake-v1', desc=squeezed_map, map_name="4x4", is_slippery=False, render_mode="human")
        pro_losses, pro_win_ratio, pro_rewards, pro_steps = adversary.collect_trajectories(new_map, protagonist, 5, max_steps)
        ant_losses, ant_win_ratio, ant_rewards, ant_steps = adversary.collect_trajectories(new_map, antagonist, 5, max_steps)

        regret = np.max(ant_rewards) - np.mean(pro_rewards)

        protagonist_win_ratio.append(pro_win_ratio)
        antagonist_win_ratio.append(ant_win_ratio)
        protagonist_steps.append(np.mean(pro_steps))
        antagonist_steps.append(np.mean(ant_steps))

        if(pro_win_ratio == 0 and ant_win_ratio == 0): regret = -0.0001
        loss = adversary.train(regret)
        losses.append(loss)

        print(f'Episode: {e}')
        if (np.mean(pro_steps) == 125.5 and np.mean(ant_steps) == 125.5):
            print('<<< first step always hole >>>')
        print(f'regret: {regret}')
        if (e != 0 and e % 100 == 0):
            pass
        print(f'pro_win_ratio: {pro_win_ratio}')
        print(f'ant_win_ratio: {ant_win_ratio}')
        print(f'np.mean(pro_steps): {np.mean(pro_steps)}')
        print(f'np.mean(ant_steps): {np.mean(ant_steps)}')
        print(f'np.mean(losses): {np.mean(losses)}')

def run_PPO_PAIRED(size, episodes):
    pass

if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    #print(tf.config.list_physical_devices('GPU'))
    with tf.device('/GPU:0'):
        run_DQN_PAIRED((10, 10), 100)