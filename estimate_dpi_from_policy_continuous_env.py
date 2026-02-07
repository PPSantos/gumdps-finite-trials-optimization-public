import os
import json
import pathlib
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils import NumpyEncoder
from algos.utils import estimate_d_pi_from_trajectory
from envs.envs_gym import get_gym_env


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

def create_exp_name(env, algo) -> str:
    return env + \
        '_' + algo + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


def sample_trajectory_from_policy(env, policy, traj_length):

    states = []
    actions = []

    s, s_continuous = env.reset()
    policy.reset()

    for t in range(traj_length):
        states.append(s)

        # if s_continuous[0] > 0.5:
        #     print("REACHEEDED")

        a = policy.select_action(s_continuous)
        actions.append(a)

        s, _, _, _, s_continuous = env.step(s_continuous, a)

    return states, actions

def estimate_dpi_from_policy(env, policy, gamma, num_episodes, episode_length=100):

    d_pi = np.zeros((env.num_states, env.num_actions))

    for k in tqdm(range(num_episodes)):
        traj_states, traj_actions = sample_trajectory_from_policy(env, policy, traj_length=episode_length)
        d_pi += estimate_d_pi_from_trajectory(env, traj_states, traj_actions, gamma)
    
    return d_pi / num_episodes

class MountainCarPolicy:

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.reached = False

    def select_action(self, state):

        if state[0] < -0.85:
            self.reached = True

        if np.random.rand() < self.epsilon:
            action = 1 # Do not accelerate.
        else:
            if self.reached:
                action = 2
            else:
                action = 0

        return action

    def reset(self):
        self.reached = False


if __name__ == "__main__":

    env_name = "mountain_car_continuous_entropy"

    num_episodes_dpi_estimation = 10_000
    dpi_estimation_epsilon = 0.1
    gamma = 0.9
    episode_length = 100

    policy = MountainCarPolicy(epsilon=dpi_estimation_epsilon)

    # Setup experiment data folder.
    exp_name = create_exp_name(env_name, "estimate_dpi")
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)

    env, _ = get_gym_env(env_name)
    print("env.num_states", env.num_states)
    print("env.num_actions", env.num_actions)

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
