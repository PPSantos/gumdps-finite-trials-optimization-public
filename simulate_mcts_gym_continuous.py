import os
import json
import pathlib
import numpy as np
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

from utils import NumpyEncoder
from algos.mcts_continuous import MCTS_Continuous
from envs.envs_gym import get_gym_env, Occupancy_MDP_Gym_v2

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

CONFIG = {
    "N": 10, # Number of experiments to run.
    "num_processors": 10,
    "env": "mountain_car_continuous_entropy",
    "gamma": 0.9,
    "H": 10, # Truncation length.
    "n_iter_per_timestep": 4_000, # MCTS number of tree expansion steps per timestep.
}

def create_exp_name(args: dict) -> str:
    return args['env'] + '_' + args['algo'] + '_gamma_' + str(args['gamma']) + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


def simulate_MCTS(env, H, gamma, obj_f, n_iter_per_timestep=1_000):

    # Instantiate occupancy MDP.
    occupancy_mdp = Occupancy_MDP_Gym_v2(env, gamma, H, obj_f)

    # Sample initial state from the occupancy MDP.
    extended_state = occupancy_mdp.sample_initial_state()

    # Simulate until termination.
    cumulative_reward = 0.0
    for _ in tqdm(range(H)):

        mcts = MCTS_Continuous(initial_state=extended_state, env=occupancy_mdp, K_ucb=np.sqrt(2), rollout_policy=None)
        mcts.learn(n_iters=n_iter_per_timestep)
        selected_action = mcts.best_action()

        # Environment step.
        extended_state, reward, terminated = occupancy_mdp.step(extended_state, selected_action)
        cumulative_reward += reward
        
    return (-1.0) * cumulative_reward # Multiply by -1 to convert from rewards to costs.


def run(cfg, seed):

    print('Running seed=', seed)

    np.random.seed(seed)

    # Instantiate environment.
    env, obj_f = get_gym_env(cfg["env"])
    print("env", env)
    print("obj_f", obj_f)

    mcts_f_val = simulate_MCTS(env=env,
                               H=cfg["H"],
                               gamma=cfg["gamma"],
                               obj_f=obj_f,
                               n_iter_per_timestep=cfg["n_iter_per_timestep"])

    return mcts_f_val


def main(cfg):

    # Setup experiment data folder.
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': "mcts",
                                'gamma': cfg['gamma']})
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('Config:')
    print(cfg)

    # Simulate.
    print('\nSimulating...')

    with mp.Pool(processes=cfg["num_processors"]) as pool:
        f_vals = pool.starmap(run, [(cfg, t) for t in range(cfg["N"])])
        pool.close()
        pool.join()

    f_vals = np.array(f_vals)

    exp_data = {}
    exp_data["config"] = cfg
    exp_data["f_vals"] = f_vals
    # exp_data["env"]["f"] = None

    # Dump dict.
    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return exp_name


if __name__ == "__main__":
    main(cfg = CONFIG)
