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

def PPO_objective(trial):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tune_log_dir = 'PPO_tuning/logs/fit/' + current_time
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
    gamma = trial.suggest_float('gamma', 0.1, 0.9999, log=True)
    max_step_size_multiplicator = trial.suggest_float('max_step_size_mult', 1, 10)
    epsilon = trial.suggest_float('epsilon', 0.1, 0.9999 , log=True)
    epsilon__decay = trial.suggest_float('epsilon_decay', 1e-5, 1e-1, log=True)
    # reward params
    scaling_factor = trial.suggest_float('scaling_factor', 1e-1, 1, log=True)
    sigma = trial.suggest_float('sigma', 1e-1, 10, log=True)
    reward = trial.suggest_float('reward', 1, 5)
    punishment_per_hole = -trial.suggest_float('punishment_per_hole', 1e-5, 10)
    punishment_per_step = -trial.suggest_float('punishment_per_step', 1e-5, 1, log=True)
    punishment_for_max_step = -trial.suggest_float('punishment_for_max_step', 2, 10)
    params = ["learning_rate", "gamma", "max_step_size_mult", "epsilon", "epsilon_decay",
              "scaling_factor", "sigma", "reward", "punishment_per_hole", "punishment_per_step", "punishment_for_max_step"]
    print(f'learning_rate {alpha}, gamma {gamma}, max_step_size_mult {max_step_size_multiplicator}, epsilon {epsilon}, epsilon_decay {epsilon__decay}, '
          f'scaling_factor {scaling_factor}, sigma {sigma}, reward {reward}, punishment_per_hole {punishment_per_hole}, '
          f'punishment_per_step {punishment_per_step}, punishment_for_max_step {punishment_for_max_step}')

    # network configurations
    actor = networks.Conv_Network(4, map_dims)
    critic = networks.Critic_Network(1, map_dims)
    agent = agents.PPO_Agent(alpha=0.0001, gamma=0.995, epsilon=0.2, n_actions=4, actor=actor, critic=critic,
                             batch_size=512)
    agent.critic.trainable = True

    # metric lists
    scores = []
    losses = []
    steps_per_episode = []
    # training loop
    old_state, _ = env.reset()
    _, old_state = env_map.map_step(old_state)
    rewards = []
    losses = []
    steps_per_episode = []
    win_ratio = []
    mean_losses = []
    steps_over_optimal = []
    episodes = 10000
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
            action, value, probs, probs_two = agent.choose_action(old_state)
            new_state, reward, done, second_flag, info = env.step(action)
            _, new_state = env_map.map_step(new_state)
            distance = helper.get_distance(new_state)
            reward, distance_bonus = helper.reward_function(reward=reward, done=done, new_reward=2, punishment=-2,
                                                            step_penalty=-0.02, distance=distance, scaling_factor=1)
            # print(distance, distance_bonus, difference)
            with train_writer.as_default():
                tf.summary.scalar('Distance', difference[0], step=tb_steps)
                tf.summary.scalar('Bonus', distance_bonus, step=tb_steps)
            agent.buffer.storeTransition(state=old_state, action=action, reward=reward, value=value, probs=probs,
                                         done=done)
            old_state = new_state
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
            # performance = helper.evaluate_policy(env, env_map, agent)

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
            # helper.show_PPO_probs(map_dims, agent, env_map)
            # helper.show_PPO_value(map_dims, agent, env_map)
            wins = 0
            lose = 0
            # save weights every 100 episodes
            # helper.save_model(network, 'PPO_training')


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

    return result

if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'DQN_tuning/logs/fit/' + current_time
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    algo = 'REINFORCE'

    if (algo == 'PPO'):
        start_time = datetime.datetime.now()
        with tf.device('/GPU:0'):
            tensorboard_callback = TensorBoardCallback(log_dir, metric_name="result")
            finish_training = False

            study = optuna.create_study(direction="maximize", study_name='PPO-study',
                                        storage='sqlite:///PPO_tuning/DQN-study.db', load_if_exists=True)

        study.optimize(DQN_objective, n_trials=2)  # , callbacks=[tensorboard_callback])
        # print duration
        finish_time = datetime.datetime.now()
        elapsed_time = finish_time - start_time
        print("Execution time:", f"{elapsed_time} seconds.")
        # Plot optimization history: Shows the intermediate values of the objective function during optimization.
        optuna.visualization.plot_optimization_history(study)
        # Plot parameter importance: Shows the relative importance of each parameter in the optimization.
        optuna.visualization.plot_param_importances(study)
        # Plot parallel coordinate plot: Visualize the relationship between multiple parameters and the objective function.
        optuna.visualization.plot_parallel_coordinate(study,
                                                      params=["learning_rate", "gamma", "max_step_size_mult", "epsilon",
                                                              "epsilon_decay",
                                                              "scaling_factor", "sigma", "reward", "punishment_per_hole",
                                                              "punishment_per_step", "punishment_for_max_step"])
        print(f"Best parameters: {study.best_params}")