import numpy as np

from utils import NumpyEncoder, compute_d_pi
from envs.create_grid_env import get_grid_env_transition_matrix

ENVS = {
    "entropy_mdp_00": { # Deterministic transitions.
        "states": [0,1,2],
        "actions": [0,1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0],
        "P": np.array([[[0,1,0],[0,1,0],[1,0,0]],[[0,0,1],[1,0,0],[0,0,1]]]),
    },
    "entropy_mdp_10": { # Stochastic transitions.
        "states": [0,1,2],
        "actions": [0,1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0],
        "P": np.array([[[0.05,0.9,0.05],[0.05,0.9,0.05],[0.9,0.05,0.05]],[[0.05,0.05,0.9],[0.9,0.05,0.05],[0.05,0.05,0.9]]]),
    },
    "imitation_learning_mdp_00": { # Deterministic transitions.
        "states": [0,1],
        "actions": [0,1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0,0.0],
        "P": np.array([[[1.0,0.0],[1.0,0.0]],[[0.0,1.0],[0.0,1.0]]]),
    },
    "imitation_learning_mdp_10": { # Stochastic transitions.
        "states": [0,1],
        "actions": [0,1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0,0.0],
        "P": np.array([[[0.9,0.1],[0.9,0.1]],[[0.1,0.9],[0.1,0.9]]]),
    },
    "risk-averse_mdp_00": { # Deterministic transitions.
        "states": [0,1],
        "actions": [0,1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0,0.0],
        "P": np.array([[[1.0,0.0],[1.0,0.0]],[[0.0,1.0],[0.0,1.0]]]),
    },
    "entropy_grid_mdp_10": { # Stochastic transitions (eps=0.1).
        "states": list(range(25)),
        "actions": [0,1,2,3], # Action 0 = Up, action 1 = Down, action 2 = Left, action 3 = Right.
        "gamma": 0.99,
        "p_0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "P": get_grid_env_transition_matrix(eps=0.1),
    },
    "imitation_learning_grid_mdp_10": { # Stochastic transitions (eps=0.1).
        "states": list(range(25)),
        "actions": [0,1,2,3], # Action 0 = Up, action 1 = Down, action 2 = Left, action 3 = Right.
        "gamma": 0.99,
        "p_0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "P": get_grid_env_transition_matrix(eps=0.1),
    },
    "entropy_grid_mdp_00": { # Determinisitic transitions (eps=0.0).
        "states": list(range(25)),
        "actions": [0,1,2,3], # Action 0 = Up, action 1 = Down, action 2 = Left, action 3 = Right.
        "gamma": 0.99,
        "p_0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "P": get_grid_env_transition_matrix(eps=0.0),
    },
    "imitation_learning_grid_mdp_00": { # Determinisitic transitions (eps=0.0).
        "states": list(range(25)),
        "actions": [0,1,2,3], # Action 0 = Up, action 1 = Down, action 2 = Left, action 3 = Right.
        "gamma": 0.99,
        "p_0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "P": get_grid_env_transition_matrix(eps=0.0),
    },
    "adversarial_mdp_10": { # Stochastic transitions.
        "states": [0,1,2],
        "actions": [0,1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0],
        "P": np.array([[[0.05,0.9,0.05],[0.05,0.9,0.05],[0.9,0.05,0.05]],[[0.05,0.05,0.9],[0.9,0.05,0.05],[0.05,0.05,0.9]]]),
        "cost_f1": np.array([[0.5,1.0], [1.0,0.5], [1.5,2.0]]),
        "cost_f2": np.array([[1.5,2.0], [0.5,1.0], [0.5,1.0]]),
        "cost_f3": np.array([[1.0,0.5], [1.5,2.0], [1.0,0.5]]),
    },
    "mdp_10": { # Standard MDP with stochastic transitions.
        "states": [0,1,2],
        "actions": [0,1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0],
        "P": np.array([[[0.05,0.9,0.05],[0.05,0.9,0.05],[0.9,0.05,0.05]],[[0.05,0.05,0.9],[0.9,0.05,0.05],[0.05,0.05,0.9]]]),
    },
    "mean_variance_mdp": {
        "states": [0,1,2,3],
        "actions": [0,1], # Action 0 and action 1.
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0, 0.0],
        "P": np.array([[[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]],[[0.0,0.0,0.9,0.1],[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]]]),
        "cost_f": np.array([[0.0,0.0], [1.0,1.0], [0.0,0.0], [8.0,8.0]]),
    },
}

def get_env(env_name, normalize_obj = True):

    env = ENVS[env_name]

    if env_name in ["entropy_mdp_00", "entropy_mdp_10", "entropy_grid_mdp_10", "entropy_grid_mdp_00"]:
        # Define objective function for entropy maximization MDP.
        d_lower_bounds_eps = 1e-08
        SA = len(env["states"]) * len(env["actions"])
        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = x.flatten()
            x = (1 - eps * sa) * x + eps
            return np.dot(x, np.log(x))
        
        obj = obj_f
        L_const = np.abs(np.log(d_lower_bounds_eps) + 1)
        
        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0
            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa, eps=eps)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
            L_const *= 1.0 / (f_max - f_min)

        env["f"] = obj
        env["lipschitz_constant"] = L_const

    elif env_name in ["imitation_learning_mdp_00", "imitation_learning_mdp_10"]:
        # Define objective function for imitation learning MDP.

        beta = np.array([[0.8,0.2], [0.2,0.8]]) # Define behaviour policy.
        imitation_learning_d_beta = compute_d_pi(env, beta)

        def obj_f(x):
            return np.sum((x - imitation_learning_d_beta)**2)
        obj = obj_f
        L_const = 4.0

        # Normalize.
        if normalize_obj:
            f_min = 0.0
            f_max = 2.0
            def obj_f_normalized(x):
                aux = obj_f(x)
                return (aux - f_min) / (f_max - f_min)
            obj = obj_f_normalized
            L_const *= 1.0 / (f_max - f_min)

        env["d_beta"] = imitation_learning_d_beta
        env["lipschitz_constant"] = L_const
        env["f"] = obj

    elif env_name in ["mean_variance_mdp"]:

        cost_f = ENVS["mean_variance_mdp"]["cost_f"]

        def obj_f(x):
            cost = np.dot(x.flatten(), cost_f.flatten())
            return cost - np.dot(x.flatten(), (cost_f.flatten() - cost)**2)
        obj = obj_f
        L_const = 1.0 # TODO.

        env["f"] = obj
        env["lipschitz_constant"] = L_const

    elif env_name in ["adversarial_mdp_10"]:

        cost_f_1 = ENVS["adversarial_mdp_10"]["cost_f1"]
        cost_f_2 = ENVS["adversarial_mdp_10"]["cost_f2"]
        cost_f_3 = ENVS["adversarial_mdp_10"]["cost_f3"]

        # Define objective function for the adversarial MDP.
        def obj_f(x):
            J_1 = np.dot(x.flatten(), cost_f_1.flatten())
            J_2 = np.dot(x.flatten(), cost_f_2.flatten())
            J_3 = np.dot(x.flatten(), cost_f_3.flatten())
            return max(J_1, J_2, J_3)
        obj = obj_f
        L_const = 1.0 # TODO.

        env["f"] = obj
        env["lipschitz_constant"] = L_const

    elif env_name in ["mdp_10"]:

        cost_f_1 = np.array([[1.0,1.0], [0.0,0.0], [2.0,2.0]])

        # Define objective function for the adversarial MDP.
        def obj_f(x):
            return np.dot(x.flatten(), cost_f_1.flatten())
        obj = obj_f
        L_const = 1.0 # TODO.

        env["f"] = obj
        env["lipschitz_constant"] = L_const


    # elif env_name in ["imitation_learning_grid_mdp_10", "imitation_learning_grid_mdp_00"]:
    #     # Define objective function for imitation learning MDP.
    #     beta = np.array([[0.05,0.05,0.05,0.85], # 0
    #                      [0.05,0.05,0.05,0.85], # 1
    #                      [0.05,0.05,0.05,0.85], # 2
    #                      [0.05,0.05,0.05,0.85], # 3
    #                      [0.05,0.85,0.05,0.05], # 4
    #                      [0.05,0.05,0.05,0.85], # 5
    #                      [0.05,0.05,0.05,0.85], # 6
    #                      [0.05,0.05,0.05,0.85], # 7
    #                      [0.05,0.05,0.05,0.85], # 8
    #                      [0.05,0.85,0.05,0.05], # 9
    #                      [0.05,0.05,0.05,0.85], # 10
    #                      [0.05,0.05,0.05,0.85], # 11
    #                      [0.05,0.05,0.05,0.85], # 12
    #                      [0.05,0.05,0.05,0.85], # 13
    #                      [0.05,0.85,0.05,0.05], # 14
    #                      [0.05,0.05,0.05,0.85], # 15
    #                      [0.05,0.05,0.05,0.85], # 16
    #                      [0.05,0.05,0.05,0.85], # 17
    #                      [0.05,0.85,0.05,0.05], # 18
    #                      [0.05,0.85,0.05,0.05], # 19
    #                      [0.05,0.05,0.05,0.85], # 20
    #                      [0.05,0.05,0.05,0.85], # 21
    #                      [0.05,0.05,0.05,0.85], # 22
    #                      [0.25,0.25,0.25,0.25], # 23
    #                      [0.05,0.05,0.85,0.05], # 24
    #                     ]) # Define behaviour policy.
    #     imitation_learning_d_beta = compute_d_pi(env, beta)

    #     def obj_f(x):
    #         return np.sum((x - imitation_learning_d_beta)**2)
    #     obj = obj_f
    #     L_const = 4.0

    #     # Normalize.
    #     if normalize_obj:
    #         f_min = 0.0
    #         f_max = 2.0
    #         def obj_f_normalized(x):
    #             aux = obj_f(x)
    #             return (aux - f_min) / (f_max - f_min)
    #         obj = obj_f_normalized
    #         L_const *= 1.0 / (f_max - f_min)

    #     env["d_beta"] = imitation_learning_d_beta
    #     env["lipschitz_constant"] = L_const
    #     env["f"] = obj

    return env

class Occupancy_MDP:

    def __init__(self, mdp, H):

        self.mdp = mdp
        self.H = H

    def available_actions(self, state):
        return self.mdp["actions"]

    def sample_initial_state(self):
        # Sample initial state.
        state = np.random.choice(self.mdp["states"], p=self.mdp["p_0"])
        extended_state = {"state": state,
                          "occupancy": np.zeros((len(self.mdp["states"]), len(self.mdp["actions"]))),
                          "t": 0} # (state, running occupancy, timestep).
        return extended_state
    
    def step(self, extended_state, a):

        # Simulate a step of the finite-horizon, occupancy MDP.
        state_t, occupancy_t, timestep_t = extended_state["state"], extended_state["occupancy"], extended_state["t"]
        next_occupancy = np.copy(occupancy_t)
        next_occupancy[state_t][a] += self.mdp["gamma"]**timestep_t
        next_state = np.random.choice(self.mdp["states"], p=self.mdp["P"][a,state_t,:])
        next_timestep = timestep_t + 1
        next_extended_state = {"state": next_state,
                               "occupancy": next_occupancy,
                               "t": next_timestep}

        if next_timestep >= self.H:
            normalized_occupancy = ((1 - self.mdp["gamma"])/(1 - self.mdp["gamma"]**self.H)) * occupancy_t
            cost = (-1.0) * self.mdp["f"](normalized_occupancy) # Multiply by -1 to convert from "minimization" to "maximization" because the MCTS considers rewards.
            terminated = True
        else:
            cost = 0.0
            terminated = False

        return next_extended_state, cost, terminated
    
    def next_possible_states(self, extended_state, a):

        state_t, occupancy_t, timestep_t = extended_state["state"], extended_state["occupancy"], extended_state["t"]
        next_occupancy = np.copy(occupancy_t)
        next_occupancy[state_t][a] += self.mdp["gamma"]**timestep_t
        next_timestep = timestep_t + 1

        if next_timestep >= self.H:
            is_final = True
        else:
            is_final = False

        possible_next_states = np.nonzero(self.mdp["P"][a,state_t,:])[0]
        next_extended_states_list = []
        for next_state in possible_next_states:
            next_extended_state = {"state": next_state,
                                 "occupancy": next_occupancy,
                                 "t": next_timestep}
            
            next_extended_states_list.append((next_extended_state, self.mdp["P"][a,state_t,next_state], is_final))
    
        return next_extended_states_list
