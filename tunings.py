from matplotlib import pyplot as plt

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
import optuna
from optuna_integration.tensorboard import TensorBoardCallback
import plotly
import sklearn

def DQN_objective(trial):
    start_time = datetime.datetime.now()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tune_log_dir = 'DQN_tuning/logs/fit/' + current_time
    train_writer = tf.summary.create_file_writer(tune_log_dir)
    # create map and env
    adv_map = maps.map_ten_x_ten_test_training.copy()
    shortest_path, shortest_path_length = helper.get_shortest_possible_length(adv_map)
    map_dims = adv_map.shape
    env = gym.make('FrozenLake-v1', desc=adv_map, is_slippery=False)  # , render_mode="human")
    adv_map[adv_map == 'S'] = 'P'
    adv_map = envs.Env_map(np.zeros((3,adv_map.shape[0],adv_map.shape[1]))).one_hot_map(adv_map)
    env_map = envs.Env_map(adv_map)

    # general params
    alpha = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.5, 0.9999, log=True)
    max_step_size_multiplicator = trial.suggest_float('max_step_size_mult', 1, 5)
    epsilon = trial.suggest_float('epsilon', 0.1, 0.9999 , log=True)
    epsilon__decay = trial.suggest_float('epsilon_decay', 1e-5, 1e-1, log=True)
    # reward params
    scaling_factor = trial.suggest_float('scaling_factor', 0.5, 1, log=True)
    sigma = trial.suggest_float('sigma', 5, 10, log=True)
    reward_per_goal = trial.suggest_float('reward', 1, 5)
    punishment_per_hole = -trial.suggest_float('punishment_per_hole', 4, 9)
    punishment_per_step = -trial.suggest_float('punishment_per_step', 1e-5, 0.1, log=True)
    punishment_for_max_step = -trial.suggest_float('punishment_for_max_step', 9, 10)
    use_direction = trial.suggest_categorical("use_direction", [True, False])
    use_position = trial.suggest_categorical("use_position", [True, False])
    params = ["learning_rate", "gamma", "max_step_size_mult", "epsilon", "epsilon_decay",
              "scaling_factor", "sigma", "reward", "punishment_per_hole", "punishment_per_step", "punishment_for_max_step"]
    print(f'learning_rate {alpha}\n, gamma {gamma}\n, max_step_size_mult {max_step_size_multiplicator}\n, epsilon {epsilon}\n, epsilon_decay {epsilon__decay}\n, '
          f'scaling_factor {scaling_factor}\n, sigma {sigma}\n, reward_per_goal {reward_per_goal}\n, punishment_per_hole {punishment_per_hole}\n, '
          f'punishment_per_step {punishment_per_step}\n, punishment_for_max_step {punishment_for_max_step}\n use_direction {use_direction}\n, use_position {use_position}')

    # create network & agent
    # network
    network = networks.Actor_Network(n_actions = 4)
    # agent
    agent = agents.DQN_Agent(alpha=alpha, gamma=gamma, epsilon=epsilon, n_actions=4,
                             map_dims=map_dims, memory_size=100000, training_batch_size=64, network=network,
                             use_direction=use_direction, use_position=use_position)

    # training loop
    episodes = 5000 # for small map, just as a test start
    max_steps = ((map_dims[0] * map_dims[1]) * max_step_size_multiplicator)
    # metric lists
    scores = []
    episode_reward = 0
    losses = []
    steps_per_episode = []
    epsilons = []
    rewards = []
    first_win = 0
    steps_over_optimal = []
    episode_steps = 0
    wins = 0
    wins_in_a_row = 0
    win_ratio = []
    # training loop
    position, _ = env.reset()
    direction = tf.one_hot(0, 4)
    _, old_state = env_map.map_step(position)
    position = tf.one_hot(position, map_dims[0] * map_dims[1])

    for e in range(episodes):
        win = False
        steps = 0
        done = False
        while not done:
            action, probs = agent.choose_action(old_state, direction, position)
            direction = tf.one_hot(action, 4)
            position, reward, done, second_flag, info = env.step(action)
            _, new_state = env_map.map_step(position)
            position = tf.one_hot(position, (map_dims[0] * map_dims[1]))
            distance = helper.get_distance(new_state)
            if (distance == 0):
                win = True
                if (first_win == 0):
                    first_win = e
            reward, distance_bonus = helper.reward_function(reward=reward, done=done, new_reward=reward_per_goal, punishment=punishment_per_hole, step_penalty = punishment_per_step,
                                                     distance=distance,  sigma=sigma, scaling_factor=scaling_factor)

            # add punishment to episode when exceeding max_steps
            if (steps > max_steps and not done):
                done = True
                reward += punishment_for_max_step
            episode_reward += reward

            agent.buffer.storeTransition(old_state=old_state, new_state=new_state, direction=direction,
                                         position=position, action=action, reward=reward, done=done)
            old_state = new_state

            last_action = action
            episode_steps += 1
            steps += 1
        # loss = agent.train()
        loss = agent.train_on_stack()
        losses.append(loss)
        rewards.append((episode_reward))

        # epsilon calculation
        agent.epsilon_decay(epsilon__decay)
        epsilons.append(agent.epsilon)

        # save to tb
        with train_writer.as_default():
            tf.summary.scalar('Reward', episode_reward, step=e)
            tf.summary.scalar('Loss', loss, step=e)
            tf.summary.scalar('Epsilon', agent.epsilon, step=e)

        # reset
        old_state, _ = env.reset()
        _, old_state = env_map.map_step(old_state)
        steps_per_episode.append(episode_steps)
        scores.append(episode_reward)


        if(win == True):
            steps_over_optimal.append(episode_steps-shortest_path_length)
            wins += 1
        episode_steps = 0
        episode_reward = 0

        if (e != 0 and e % 100 == 0):
            win_ratio.append(wins)
            #if (e > episodes * percentage):
                #agent.epsilon = helper.adaptive_decay(e, agent.epsilon, scores)
            print(f'Episode: {e}')
            print(f'Epsilon: {agent.epsilon}')
            print(f'Last hundred episodes mean steps: {np.mean(steps_per_episode[-100:])}')
            print(f'Games won: {wins}%')
            with train_writer.as_default():
                tf.summary.scalar('Win ratio', wins / 100, step=e)
                tf.summary.scalar('Mean Steps', np.mean(steps_per_episode[-100:]), step=e)
            print(f'Score: {np.mean(scores[-100:])}')
            print(f'Loss: {np.mean(losses[-100:])}')
            if(wins > 0):
                print(f'Mean steps over optimal: {np.mean(steps_over_optimal[-100:])}')
            wins = 0
    print("No wins" if first_win == 0 else f"First win at episode: {first_win}\nHighest ratio: {np.max(win_ratio)}")
    if len(steps_over_optimal) > 0:
        result = (np.mean(win_ratio)) + (0.25 * np.mean(rewards)) - (0.25 *np.mean(steps_over_optimal))
        print(f'Win ratio: {np.mean(win_ratio)}')
        print(f'Mean reward: {np.mean(scores)}')
        print(f'Losses: {np.mean(losses)}')
        print(f'Result: {result}')
    else:
        result = -10 + (0.35 * np.mean(rewards))
        print(f'Win ratio: Empty')
        print(f'Mean reward: {np.mean(scores)}')
        print(f'Losses: {np.mean(losses)}')
        print(f'Result: {result}')
    # print duration
    finish_time = datetime.datetime.now()
    elapsed_time = finish_time - start_time
    print("Execution time:", f"{elapsed_time} seconds.")
    return result

if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'DQN_tuning/logs/fit/' + current_time
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    algo = 'DQN'

    if (algo == 'DQN'):
        start_time = datetime.datetime.now()
        with tf.device('/GPU:0'):
            tensorboard_callback = TensorBoardCallback(log_dir, metric_name="result")
            finish_training = False

            study = optuna.create_study(direction="maximize", study_name='DQN-study', storage='sqlite:///DQN_tuning/Direction-DQN-study.db', load_if_exists=True)

            study.optimize(DQN_objective, n_trials=128)#, callbacks=[tensorboard_callback])
        # print duration
        finish_time = datetime.datetime.now()
        elapsed_time = finish_time - start_time
        print("Execution time:", f"{elapsed_time} seconds.")
        # Plot optimization history: Shows the intermediate values of the objective function during optimization.
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        # Plot parameter importance: Shows the relative importance of each parameter in the optimization.
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
        # Plot parallel coordinate plot: Visualize the relationship between multiple parameters and the objective function.
        fig = optuna.visualization.plot_parallel_coordinate(study,
                                                      params=["learning_rate", "gamma", "max_step_size_mult", "epsilon",
                                                              "epsilon_decay",
                                                              "scaling_factor", "sigma", "reward", "punishment_per_hole",
                                                              "punishment_per_step", "punishment_for_max_step"])
        fig.show()
        print(f"Best parameters: {study.best_params}")
        plt.show()



