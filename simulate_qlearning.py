import os
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils import NumpyEncoder
from algos.utils import estimate_dpi
from envs.envs_gym import get_gym_env
from algos.q_learning import QLearning


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

def create_exp_name(env, algo) -> str:
    return env + \
        '_' + algo + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


if __name__ == "__main__":

    env_name = "frozen_lake_entropy"

    learning_steps = 500_000
    episode_length = 100
    num_episodes_dpi_estimation = 1_000
    dpi_estimation_epsilon = 0.05

    ql_args = {
        "alpha_init": 0.1,
        "alpha_final": 0.001,
        "alpha_steps": 400_000,
        "gamma": 0.99,

        'replay_buffer_size': 500_000,
        'replay_buffer_batch_size': 128,
        'num_samples_per_sa_pair': 1_000,
    }

    # Setup experiment data folder.
    exp_name = create_exp_name(env_name, "q_learning")
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)

    env, _ = get_gym_env(env_name)
    print("env.num_states", env.num_states)
    print("env.num_actions", env.num_actions)

    algo = QLearning(env, qlearning_args=ql_args)
    
    train_data = algo.train_offline(learning_steps, ql_args["num_samples_per_sa_pair"], episode_length)
    # train_data = algo.train(learning_steps, episode_length)

    plt.figure()
    plt.plot(train_data["rollouts_steps"], train_data["rollouts_rewards"])
    plt.show()

    # Estimate d_\pi from eps-greedy policy induced by the Q-vals.
    estimated_d_pi = estimate_dpi(env, train_data["Q_vals"], ql_args["gamma"],
                                dpi_estimation_epsilon, num_episodes_dpi_estimation, episode_length)
    print('-'*20)
    print(estimated_d_pi)
    print(estimated_d_pi.shape)
    print(np.sum(estimated_d_pi))
    train_data["estimated_d_pi"] = estimated_d_pi

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)
