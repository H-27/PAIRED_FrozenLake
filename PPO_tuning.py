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