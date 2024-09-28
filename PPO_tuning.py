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
    epsilon = trial.suggest_float('epsilon', 0.1, 0.9999 , log=True)
    # reward params
    scaling_factor = trial.suggest_float('scaling_factor', 1e-1, 1, log=True)
    sigma = trial.suggest_float('sigma', 1e-1, 10, log=True)
    reward = trial.suggest_float('reward', 1, 5)
    punishment_per_hole = -trial.suggest_float('punishment_per_hole', 1e-5, 10)
    punishment_per_step = -trial.suggest_float('punishment_per_step', 1e-5, 1, log=True)
    punishment_for_max_step = -trial.suggest_float('punishment_for_max_step', 2, 10)
    num_envs = -trial.suggest_int('num_envs', 2, 10) = -trial.suggest_float('num_envs', 2, 10)
    n_mini_batches = -trial.suggest_int('n_mini_batches', 2, 20)
    training_epochs = -trial.suggest_int('training_epochs', 2, 10)
    params = ["learning_rate", "gamma", "max_step_size_mult", "epsilon", "epsilon_decay",
              "scaling_factor", "sigma", "reward", "punishment_per_hole", "punishment_per_step", "punishment_for_max_step"]
    print(f'learning_rate {alpha}, gamma {gamma}, epsilon {epsilon}'
          f'scaling_factor {scaling_factor}, sigma {sigma}, reward {reward}, punishment_per_hole {punishment_per_hole}, '
          f'punishment_per_step {punishment_per_step}, punishment_for_max_step {punishment_for_max_step}'
          f', num_envs {num_envs}, n_mini_batches {n_mini_batches}, training_epochs {training_epochs}')

    # network configurations
    n_samples = 128
    batch_size = num_envs * n_samples
    mini_batch_size = batch_size // n_mini_batches
    total_timesteps = 25000

    # Create actor and critic networks
    actor = networks.Actor_Network(4)
    critic = networks.Critic_Network()
    agent = agents.PPO_Agent(alpha=alpha, gamma=gamma, epsilon=epsilon, n_actions=4,
                      actor=actor, critic=critic, batch_size=mini_batch_size, n_envs=num_envs, map_dims=map_dims)
    # training loop
    training_rewards = []
    losses = []
    games_won = 0
    games_done = 0
    mean_losses = []
    # start reset
    next_dones = [False for _ in range(num_envs)]
    next_directions = [0 for _ in range(num_envs)]
    next_directions = tf.one_hot(next_directions, 4)
    next_positions = [0 for _ in range(num_envs)]
    next_positions = tf.one_hot(next_positions, 4)
    next_states = [env.reset()[0] for env in envs]
    next_states = [env_map.map_step(state)[1] for state in next_states]

    for update in range(1, total_timesteps // (num_envs * n_samples)):
        agent.buffer.reset()
        for step in range(n_samples):
            # update state and done
            obs = next_states
            directions = next_directions
            positions = next_positions
            dones = next_dones

            actions, values, probs = agent.choose_action(obs, directions, positions)
            next_directions = [tf.one_hot(action, 4) for action in actions]
            next_states, next_positions, rewards, next_dones, truncateds, infos = env_map.vectorized_step(actions, envs)
            distances = [helper.get_distance(state) for state in next_states]
            wins = distances.count(0)
            if (wins):
                for win in range(wins):
                    games_won += 1
            results = [helper.reward_function(reward=r, done=d, new_reward=reward, punishment=punishment_per_hole, step_penalty=punishment_per_step, distance=dist,
                                       sigma=sigma, scaling_factor=scaling_factor)
                       for r, d, dist in zip(rewards, next_dones, distances)]
            rewards, distance_bonuses = zip(*results)
            agent.buffer.storeTransition(state=obs, direction=directions, position=positions, action=actions,
                                         reward=rewards,
                                         value=values, probs=probs, done=dones)
            # reset done envs
            true_indices = np.where(next_dones)[0]
            for i in true_indices:
                next_state = envs[i].reset()[0]
                next_state = env_map.map_step(next_state)[1]
                next_states[i] = next_state
                games_done += 1
            training_rewards.append(rewards)
        # prep buffer
        # convert to tensor
        next_states_tensor = tf.convert_to_tensor(next_states)
        next_directions_tensor = tf.convert_to_tensor(next_directions)
        next_positions_tensor = tf.convert_to_tensor(next_positions)
        # prep next state for buffer
        agent.buffer.dones.append(next_dones)
        next_value = agent.critic(next_states_tensor, next_directions_tensor, next_positions_tensor)
        agent.buffer.values.append(next_value)
        agent.buffer.calculate_advantages(gamma)
        for k in range(training_epochs):
            loss, actor_loss, critic_loss = agent.train()
            losses.append(loss)
        agent.buffer.reset()

        print(update)
        mean_losses.append(np.mean(losses))
        if (update != 0 and update % 10 == 0):
            print(f'Games Won: {games_won}')
            print(f'Games Done: {games_done}')
            print(f'Games won: {games_won / games_done}%')
            print(f'Score: {np.mean(training_rewards)}')
            print(f'Loss: {np.mean(losses)}')
            games_won = 0
            games_done = 0


    result = (np.mean(win_ratio)) + (0.25 * np.mean(rewards)) - (0.25 *np.mean(steps_over_optimal))


    return result

if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'DQN_tuning/logs/fit/' + current_time
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = datetime.datetime.now()
    with tf.device('/GPU:0'):
        tensorboard_callback = TensorBoardCallback(log_dir, metric_name="result")
        finish_training = False

        study = optuna.create_study(direction="maximize", study_name='PPO-study',
                                    storage='sqlite:///PPO_tuning/PPO-study.db', load_if_exists=True)

        study.optimize(PPO_objective, n_trials=2)  # , callbacks=[tensorboard_callback])
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