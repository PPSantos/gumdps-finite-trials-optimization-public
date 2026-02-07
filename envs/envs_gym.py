import json
import pathlib
import numpy as np
from envs.frozen_lake_env import FrozenLakeEnv

from envs.env_discretizer import GymEnvDiscretizer, GymEnvDiscretizer_v2
from envs.pendulum_env import PendulumEnv
from envs.mountain_car import MountainCarEnv
from envs.taxi_env import TaxiEnv

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent) + '/data/'
print("DATA_FOLDER_PATH, envs_gym:", DATA_FOLDER_PATH)

# FROZEN_LAKE_D_BETA = "frozen_lake_entropy_q_learning_2025-04-15-22-41-47" # Distribution to imitate (FrozenLake env.)
# FROZEN_LAKE_D_BETA = "frozen_lake_entropy_q_learning_2025-04-18-10-18-10" # Distribution to imitate (FrozenLake env.)
FROZEN_LAKE_D_BETA = "frozen_lake_entropy_q_learning_2025-04-18-12-27-29" # Distribution to imitate (FrozenLake env.)
# PENDULUM_D_BETA = "pendulum_entropy_q_learning_2025-04-16-14-13-48" # Distribution to imitate (Pendulum env.)
# PENDULUM_D_BETA = "pendulum_entropy_q_learning_2025-04-18-14-02-15" # Distribution to imitate (Pendulum env.)
PENDULUM_D_BETA = "pendulum_entropy_q_learning_2025-04-19-14-10-16" # Distribution to imitate (Pendulum env.) - 20 bins.
MOUNTAINCAR_D_BETA = "mountain_car_entropy_q_learning_2025-04-17-13-03-44" # Distribution to imitate (MountainCar env.)
MOUNTAINCAR_D_BETA_CONTINUOUS = "mountain_car_continuous_entropy_estimate_dpi_2025-04-29-15-34-57"
TAXI_D_BETA = "taxi_entropy_q_learning_2025-04-21-17-38-47"

def get_gym_env(env_name, normalize_obj = True):

    if env_name in ["frozen_lake_entropy"]:

        env = FrozenLakeEnv(map_name="4x4", is_slippery=True)

        # Define objective function for entropy maximization.
        d_lower_bounds_eps = 1e-08
        SA = env.num_states * env.num_actions
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x))
        obj = obj_f
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
        
        return env, obj
    
    elif env_name in ["frozen_lake_imitation_learning"]:

        env = FrozenLakeEnv(map_name="4x4", is_slippery=True)

        # Load distribution to imitate.
        with open(DATA_FOLDER_PATH + FROZEN_LAKE_D_BETA + "/train_data.json", 'r') as f:
            ql_data = json.load(f)
            ql_data = json.loads(ql_data)
        f.close()

        imitation_learning_d_beta = ql_data["estimated_d_pi"]

        def obj_f(x):
            return np.sum((x - imitation_learning_d_beta)**2)
        obj = obj_f

        # Normalize.
        if normalize_obj:
            f_min = 0.0
            f_max = 2.0
            def obj_f_normalized(x):
                aux = obj_f(x)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized

        return env, obj
    
    elif env_name in ["taxi_entropy"]:

        env = TaxiEnv()

        # Define objective function for entropy maximization.
        d_lower_bounds_eps = 1e-10
        SA = env.num_states * env.num_actions
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            # x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x + 1e-10))
        obj = obj_f
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
        
        return env, obj
    
    elif env_name in ["taxi_imitation_learning"]:

        # Load distribution to imitate.
        with open(DATA_FOLDER_PATH + TAXI_D_BETA + "/train_data.json", 'r') as f:
            ql_data = json.load(f)
            ql_data = json.loads(ql_data)
        f.close()

        imitation_learning_d_beta = ql_data["estimated_d_pi"]

        env = TaxiEnv()

        def obj_f(x):
            return np.sum((x - imitation_learning_d_beta)**2)
        obj = obj_f

        # Normalize.
        if normalize_obj:
            f_min = 0.0
            f_max = 2.0
            def obj_f_normalized(x):
                aux = obj_f(x)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized

        return env, obj
    
    elif env_name in ["pendulum_entropy_continuous"]:

        env = GymEnvDiscretizer_v2(PendulumEnv(), dim_bins=20)
    
        # Define objective function for entropy maximization.
        d_lower_bounds_eps = 1e-08
        SA = env.num_states * env.num_actions
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            # x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x + 1e-10))
        obj = obj_f
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
        
        return env, obj
    
    elif env_name in ["mountain_car_continuous_entropy"]:

        env = GymEnvDiscretizer_v2(MountainCarEnv(), dim_bins=10)
    
        # Define objective function for entropy maximization.
        d_lower_bounds_eps = 1e-08
        SA = env.num_states * env.num_actions
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x))
        obj = obj_f
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
        
        return env, obj
    
    elif env_name in ["mountain_car_continuous_imitation_learning"]:

        env = GymEnvDiscretizer_v2(MountainCarEnv(), dim_bins=10)

        # Load distribution to imitate.
        with open(DATA_FOLDER_PATH + MOUNTAINCAR_D_BETA_CONTINUOUS + "/train_data.json", 'r') as f:
            ql_data = json.load(f)
            ql_data = json.loads(ql_data)
        f.close()

        imitation_learning_d_beta = ql_data["estimated_d_pi"]

        def obj_f(x):
            return np.sum((x - imitation_learning_d_beta)**2)
        obj = obj_f

        # Normalize.
        if normalize_obj:
            f_min = 0.0
            f_max = 2.0
            def obj_f_normalized(x):
                aux = obj_f(x)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized

        return env, obj
    
    elif env_name in ["pendulum_entropy"]:

        env = GymEnvDiscretizer(PendulumEnv(), dim_bins=20)
    
        # Define objective function for entropy maximization.
        d_lower_bounds_eps = 1e-08
        SA = env.num_states * env.num_actions
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            # x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x + 1e-10))
        obj = obj_f
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
        
        return env, obj
    
    elif env_name in ["pendulum_imitation_learning"]:

        env = GymEnvDiscretizer(PendulumEnv(), dim_bins=20)

        # Load distribution to imitate.
        with open(DATA_FOLDER_PATH + PENDULUM_D_BETA + "/train_data.json", 'r') as f:
            ql_data = json.load(f)
            ql_data = json.loads(ql_data)
        f.close()

        imitation_learning_d_beta = ql_data["estimated_d_pi"]

        def obj_f(x):
            return np.sum((x - imitation_learning_d_beta)**2)
        obj = obj_f

        # Normalize.
        if normalize_obj:
            f_min = 0.0
            f_max = 2.0
            def obj_f_normalized(x):
                aux = obj_f(x)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized

        return env, obj
    
    elif env_name in ["mountain_car_entropy"]:

        env = GymEnvDiscretizer(MountainCarEnv(), dim_bins=20)
    
        # Define objective function for entropy maximization.
        d_lower_bounds_eps = 1e-08
        SA = env.num_states * env.num_actions
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x))
        obj = obj_f
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
        
        return env, obj
    
    elif env_name in ["mountain_car_imitation_learning"]:

        env = GymEnvDiscretizer(MountainCarEnv(), dim_bins=20)

        # Load distribution to imitate.
        with open(DATA_FOLDER_PATH + MOUNTAINCAR_D_BETA + "/train_data.json", 'r') as f:
            ql_data = json.load(f)
            ql_data = json.loads(ql_data)
        f.close()

        imitation_learning_d_beta = ql_data["estimated_d_pi"]

        def obj_f(x):
            return np.sum((x - imitation_learning_d_beta)**2)
        obj = obj_f

        # Normalize.
        if normalize_obj:
            f_min = 0.0
            f_max = 2.0
            def obj_f_normalized(x):
                aux = obj_f(x)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized

        return env, obj
    
    else:
        raise ValueError("Unknown environment.")

    """ if env_name in ["pendulum_entropy"]:


        # Define objective function for entropy maximization.
        d_lower_bounds_eps = 1e-08
        SA = env.num_states * env.num_actions
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x))
        obj = obj_f
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
        
        return env, obj
    
    elif env_name in ["pendulum_imitation_learning"]:

        env = env_discretizer.wrap_env(PendulumEnv)(dim_bins=20) # TODO: Check this value.

        # Load distribution to imitate.
        with open(DATA_FOLDER_PATH + "pendulum_entropy_q_learning_2025-04-12-18-14-03" + "/train_data.json", 'r') as f:
            ql_data = json.load(f)
            ql_data = json.loads(ql_data)
        f.close()

        imitation_learning_d_beta = ql_data["estimated_d_pi"]

        def obj_f(x):
            return np.sum((x - imitation_learning_d_beta)**2)
        obj = obj_f

        # Normalize.
        if normalize_obj:
            f_min = 0.0
            f_max = 2.0
            def obj_f_normalized(x):
                aux = obj_f(x)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized

        return env, obj """


class Occupancy_MDP_Gym:

    def __init__(self, gym_env, gamma, H, obj_f):

        self.gym_env = gym_env
        self.gamma = gamma
        self.H = H
        self.obj_f = obj_f

    def available_actions(self, state):
        return list(range(self.gym_env.num_actions))

    def sample_initial_state(self):
        # Sample initial state.
        state, _ = self.gym_env.reset()
        extended_state = {"state": state,
                          "occupancy": np.zeros((self.gym_env.num_states, self.gym_env.num_actions)),
                          "t": 0} # (state, running occupancy, timestep).
        return extended_state
    
    def step(self, extended_state, a):

        # Simulate a step of the finite-horizon, occupancy MDP.
        state_t, occupancy_t, timestep_t = extended_state["state"], extended_state["occupancy"], extended_state["t"]
        next_occupancy = np.copy(occupancy_t)
        next_occupancy[state_t][a] += self.gamma**timestep_t
        next_state, _, _, _, _ = self.gym_env.step(state_t, a)
        next_timestep = timestep_t + 1
        next_extended_state = {"state": next_state,
                               "occupancy": next_occupancy,
                               "t": next_timestep}

        if next_timestep >= self.H:
            normalized_occupancy = ((1 - self.gamma)/(1 - self.gamma**self.H)) * occupancy_t
            cost = (-1.0) * self.obj_f(normalized_occupancy) # Multiply by -1 to convert from "minimization" to "maximization" because the MCTS considers rewards.
            terminated = True
        else:
            cost = 0.0
            terminated = False

        return next_extended_state, cost, terminated


class Occupancy_MDP_Gym_v2:

    def __init__(self, gym_env, gamma, H, obj_f):

        self.gym_env = gym_env
        self.gamma = gamma
        self.H = H
        self.obj_f = obj_f

    def available_actions(self, state):
        return list(range(self.gym_env.num_actions))

    def sample_initial_state(self):
        # Sample initial state.
        discrete_state, continuous_state = self.gym_env.reset()
        extended_state = {"continuous_state": continuous_state,
                          "discrete_state": discrete_state,
                          "occupancy": np.zeros((self.gym_env.num_states, self.gym_env.num_actions)),
                          "t": 0} # (state, running occupancy, timestep).
        return extended_state
    
    def step(self, extended_state, a):

        # Simulate a step of the finite-horizon, occupancy MDP.
        discrete_state_t, continuous_state_t, occupancy_t, timestep_t = extended_state["discrete_state"], extended_state["continuous_state"], extended_state["occupancy"], extended_state["t"]
        next_occupancy = np.copy(occupancy_t)
        next_occupancy[discrete_state_t][a] += self.gamma**timestep_t
        next_discrete_state, _, _, _, next_continuous_state = self.gym_env.step(continuous_state_t, a)
        next_timestep = timestep_t + 1
        next_extended_state = {"continuous_state": next_continuous_state,
                               "discrete_state": next_discrete_state,
                               "occupancy": next_occupancy,
                               "t": next_timestep}

        if next_timestep >= self.H:
            normalized_occupancy = ((1 - self.gamma)/(1 - self.gamma**self.H)) * occupancy_t
            cost = self.obj_f(normalized_occupancy)
            terminated = True
        else:
            cost = 0.0
            terminated = False

        return next_extended_state, cost, terminated
