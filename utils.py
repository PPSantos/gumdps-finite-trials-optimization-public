import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def compute_d_pi(mdp, policy):
    # Args:
    # mdp - MDP specification.
    # policy - policy specification (np.array).

    nS = len(mdp["states"])
    nA = len(mdp["actions"])

    P_pi = np.zeros((nS, nS))
    
    for a in range(nA):
        P_pi += np.dot(np.diag(policy[:,a]), mdp["P"][a])

    d_S = (1 - mdp["gamma"]) * np.dot(mdp["p_0"], np.linalg.inv(np.eye(nS) - mdp["gamma"] * P_pi))
    d_pi = np.dot(np.diag(d_S), policy)
    
    return d_pi