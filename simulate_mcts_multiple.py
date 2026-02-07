import json
import pathlib
from datetime import datetime

from simulate_mcts_gym import main as mcts

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

CONFIG = {
    "N": 10, # Number of experiments to run.
    "num_processors": 10,
    "env": "pendulum_entropy",
    "gamma": 0.99,
    "H": 100, # Truncation length.
    "n_iter_per_timestep": 1_000, # MCTS number of tree expansion steps per timestep.
}

if __name__ == "__main__":

    exp_ids = []

    for exp_steps in [10,20,50,100,500,1_000,2_000,3_000,4_000,5_000]:
        CONFIG["n_iter_per_timestep"] = exp_steps
        exp_id = mcts(CONFIG)
        print('-'*50)
        print("exp_id:", exp_id)
        print('-'*50)
        exp_ids.append(exp_id)

    print(exp_ids)

    # Dump exp ids.
    exp_path = DATA_FOLDER_PATH
    f_name = "simulate_" + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) + ".json"

    f = open(exp_path + f_name, "w")
    dumped = json.dumps(exp_ids)
    json.dump(dumped, f)
    f.close()

