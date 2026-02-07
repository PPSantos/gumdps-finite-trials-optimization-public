import os
import json
import pathlib
import numpy as np
from datetime import datetime

from utils import NumpyEncoder
from algos.utils import estimate_dpi_from_policy
from envs.envs_gym import get_gym_env


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

def create_exp_name(env, algo) -> str:
    return env + \
        '_' + algo + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


if __name__ == "__main__":

    env_name = "frozen_lake_entropy"

    num_episodes_dpi_estimation = 5_000
    dpi_estimation_epsilon = 0.25
    gamma = 0.99
    episode_length = 100

    # FrozenLake actions:
    # 0: LEFT
    # 1: DOWN
    # 2: RIGHT
    # 3: UP

    policy = np.array([[0.05,0.45,0.45,0.05], # 0
                       [0.20,0.00,0.75,0.05], # 1
                       [0.10,0.75,0.10,0.05], # 2
                       [0.80,0.00,0.10,0.10], # 3
                       [0.10,0.80,0.00,0.10], # 4
                       [0.25,0.25,0.25,0.25], # 5
                       [0.00,0.80,0.00,0.20], # 6
                       [0.25,0.25,0.25,0.25], # 7
                       [0.05,0.00,0.80,0.15], # 8
                       [0.10,0.45,0.45,0.00], # 9
                       [0.10,0.80,0.00,0.10], # 10
                       [0.25,0.25,0.25,0.25], # 11
                       [0.25,0.25,0.25,0.25], # 12
                       [0.00,0.10,0.80,0.10], # 13
                       [0.10,0.10,0.70,0.10], # 14
                       [0.25,0.25,0.25,0.25], # 15
                       ]) # Define behaviour policy.

    # Setup experiment data folder.
    exp_name = create_exp_name(env_name, "q_learning")
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)

    env, _ = get_gym_env(env_name)
    print("env.num_states", env.num_states)
    print("env.num_actions", env.num_actions)

    assert policy.shape[0] == env.num_states
    assert policy.shape[1] == env.num_actions

    # Estimate d_\pi from eps-greedy policy induced by the Q-vals.
    estimated_d_pi = estimate_dpi_from_policy(env, policy, gamma, num_episodes_dpi_estimation, episode_length)
    print('-'*20)
    print(estimated_d_pi)
    print(estimated_d_pi.shape)
    print(np.sum(estimated_d_pi))
    train_data = {}
    train_data["estimated_d_pi"] = estimated_d_pi

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)
