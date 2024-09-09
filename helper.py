import datetime
import matplotlib
from tkinter import *
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import pandas as pd

def reward_function(reward, done):
    # for every step
    if (not done): # normal step
        reward = -0.001
    # case for hole
    elif (done and reward == 0): # ran into hole
        reward = -2
    # case for goal
    elif (done and reward > 0): # reached goal
        reward = 10
    return reward

old_global_distance = None
def reward_function(reward, done, new_reward, punishment, step_penalty, distance = 0, map_size = (4,4), scaling_factor = 0.5):
    global old_global_distance
    sigma = 5
    if distance != 0:
        #distance_bonus = 0.1 * (1 / (distance + 1))
        old_distance = old_global_distance
        if old_distance is None:  # Handle the first step when there's no previous distance
            distance_bonus = np.array(0)
        else:
            distance_bonus = np.exp(-(distance ** 2) / (2 * sigma ** 2)) - np.exp(-(old_distance ** 2) / (2 * sigma ** 2))
            distance_bonus *= scaling_factor
    else:
        old_distance = old_global_distance
        distance_bonus = np.exp(-(distance ** 2) / (2 * sigma ** 2)) - np.exp(-(old_distance ** 2) / (2 * sigma ** 2))
        distance_bonus *= scaling_factor
    # for every step
    if (not done): # normal step
        reward = step_penalty
    # case for hole
    elif (done and reward == 0): # ran into hole
        reward = punishment
    # case for goal
    elif (done and reward > 0): # reached goal
        reward = new_reward

    old_global_distance = distance
    #print(f'Distance {distance} and bonus {distance_bonus}')
    return reward + distance_bonus.item(), distance_bonus.item()

def gaussian_potential_function(distance_to_goal, map_size):
    # Increase sigma for a wider Gaussian curve
    sigma = max(map_size) / 2
    return -np.exp(-(distance_to_goal ** 2) / (2 * sigma ** 2))

def calculate_coordinates(position, n_rows):
    y = (position / n_rows) - (position % n_rows / n_rows)
    x = position - y * n_rows
    return int(y), int(x)

def squeeze_map(env_map):
    squeezed_map = []
    for row in env_map:
        squeezed_row = ""
        for entry in row:
            squeezed_row += entry
        squeezed_map.append(squeezed_row)
    return  squeezed_map
# old
def visualize_and_safe(tuples, filename, save=True):
    fig, ax = plt.subplots(len(tuples), 1)
    for i in range(len(tuples)):
        ax[i].plot(tuples[i][0])
        ax[i].set_title(tuples[i][1])
        # Show/save figure as desired.
    #plt.show()
    if(save):

        now = datetime.datetime.now()
        date_time = now.strftime("%m.%d.%Y %H:%M:%S")
        save_name = filename + " " + date_time + '.png'

        plt.savefig(save_name)
        print(f"Plot saved as: {save_name}")

def save_model(network, filename, use_timestamp = False):
    if(use_timestamp):
        now = datetime.datetime.now()
        date_time = now.strftime("%m.%d.%Y %H:%M:%S")
        filename = filename + date_time + '/cp.ckpt'
    folder = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/Models/')
    # pathname = folder + save_name
    pathname = folder + filename + '/cp.ckpt'
    network.save_weights(pathname)

def load_model(network, filepath):
    network.built = True
    filepath = '/home/jens/Desktop/Models/' + filepath + '/cp.ckpt'
    network.load_weights(filepath)


def get_distance(map):
    p_y, p_x = np.where(map[0] == 1)
    g_y, g_x  = np.where(map[2] == 1)
    distance = abs(g_x - p_x) + abs(g_y - p_y)
    return distance


def adaptive_decay(episode, epsilon, rewards, window_size=100, epsilon_increase=0.05, epsilon_decrease=0.02, epsilon_start = 0.15):

  if episode < window_size:
    return epsilon  # Not enough data for comparison

  recent_rewards = rewards[-window_size:]
  current_avg_reward = np.mean(recent_rewards)
  previous_avg_reward = np.mean(rewards[-2*window_size:-window_size])

  if current_avg_reward < previous_avg_reward:
    epsilon = min(epsilon_start, epsilon + epsilon_increase)
  elif current_avg_reward > previous_avg_reward:
    epsilon = max(0, epsilon - epsilon_increase)

  return epsilon


def create_df(metrics):
    df_data = {}
    max_length = max([len(m) for m, _ in metrics])
    for metric, name in metrics:
        if metric.size == 0:
            df_data[name] = [pd.NA] * max_length
        else:
            df_data[name] = list(metric) + [pd.NA] * (max_length - len(metric))
    df = pd.DataFrame(df_data)
    return df

def visualize_and_save_metrics(metrics, output_filename='metrics.png'):
    df = create_df(metrics)

    # Visualize all metrics in a single figure with subplots, excluding NaNs
    num_metrics = len(df.columns)
    fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(8, 6 * num_metrics))  # Adjust figsize as needed

    for i, column in enumerate(df.columns):
        valid_indices = ~df[column].isna()
        valid_values = df[column][valid_indices]
        valid_index = df.index[valid_indices]

        axes[i].plot(valid_index, valid_values, marker='o')
        axes[i].set_title(f'Metric: {column}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)

    fig.tight_layout()  # Adjust spacing between subplots

    # Save the plot
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()

    # Save the DataFrame to a CSV file
    df.to_csv('metrics.csv', index=False)
    print("DataFrame saved to metrics.csv")

def evaluate_policy(env, env_map, agent):
    wins = 0
    # play game
    old_state, _ = env.reset()
    _, old_state = env_map.map_step(old_state)
    for e in range(100):
        action, probs = agent.choose_action(old_state)
        new_state, reward, done, second_flag, info = env.step(action)
        _, new_state = env_map.map_step(new_state)
        old_state = new_state
        # check win
        if done:
            old_state, _ = env.reset()
            _, old_state = env_map.map_step(old_state)
        if (done and reward > 1):
            # log win
            wins += 1
            old_state, _ = env.reset()
            _, old_state = env_map.map_step(old_state)
    #print win rate
    #print(f'Policy Win rate: {wins}%')
    old_state, _ = env.reset()
    return wins