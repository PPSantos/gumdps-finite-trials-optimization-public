import numpy as np

class GymEnvDiscretizer:

    def __init__(self, gym_env, dim_bins):

        self.gym_env = gym_env

        self.dim_bins = dim_bins
        self.num_states = dim_bins**self.gym_env.observation_space.shape[0]
        self.num_actions = self.gym_env.action_space.n
        self.state_dims = self.gym_env.observation_space.shape[0]

        self.bins = []
        self.bins_size = []
        for s_dim in range(self.state_dims):
            self.bins.append(np.linspace(self.gym_env.observation_space.low[s_dim],
                                    self.gym_env.observation_space.high[s_dim],
                                    dim_bins+1))
            self.bins_size.append((self.gym_env.observation_space.high[s_dim] - \
                                self.gym_env.observation_space.low[s_dim]) / dim_bins)

    def _get_discrete_state(self, state):
        discretized_state = 0
        for s_dim in range(self.state_dims):
            trimmed_dim_state_value = np.minimum(state[s_dim], \
                            self.gym_env.observation_space.high[s_dim] - 1e-05)
            s_dim_idx = np.digitize(trimmed_dim_state_value, bins=self.bins[s_dim]) - 1
            discretized_state += s_dim_idx * self.dim_bins**(self.state_dims - 1 - s_dim)
        return discretized_state
    
    def _get_continuous_state(self, state):
        continuous_state = ()
        remainder = state
        for s_dim in range(self.state_dims):
            dim_state = remainder // self.dim_bins**(self.state_dims - 1 - s_dim)
            remainder -= dim_state * self.dim_bins**(self.state_dims - 1 - s_dim)
            dim_continuous_state = (self.gym_env.observation_space.low[s_dim] + \
                dim_state*self.bins_size[s_dim]) + np.random.rand()*self.bins_size[s_dim]
            continuous_state += (dim_continuous_state,)
        return continuous_state
            
    def reset(self):
        new_state, _ = self.gym_env.reset()
        return self._get_discrete_state(new_state), None
    
    def step(self, state, action):
        continuous_state = self._get_continuous_state(state)
        new_state, reward, terminated, _, _ = self.gym_env.step(continuous_state, action)
        return self._get_discrete_state(new_state), reward, terminated, None, None


class GymEnvDiscretizer_v2:

    def __init__(self, gym_env, dim_bins):

        self.gym_env = gym_env

        self.dim_bins = dim_bins
        self.num_states = dim_bins**self.gym_env.observation_space.shape[0]
        self.num_actions = self.gym_env.action_space.n
        self.state_dims = self.gym_env.observation_space.shape[0]

        self.bins = []
        self.bins_size = []
        for s_dim in range(self.state_dims):
            self.bins.append(np.linspace(self.gym_env.observation_space.low[s_dim],
                                    self.gym_env.observation_space.high[s_dim],
                                    dim_bins+1))
            self.bins_size.append((self.gym_env.observation_space.high[s_dim] - \
                                self.gym_env.observation_space.low[s_dim]) / dim_bins)

    def _get_discrete_state(self, state):
        discretized_state = 0
        for s_dim in range(self.state_dims):
            trimmed_dim_state_value = np.minimum(state[s_dim], \
                            self.gym_env.observation_space.high[s_dim] - 1e-05)
            s_dim_idx = np.digitize(trimmed_dim_state_value, bins=self.bins[s_dim]) - 1
            discretized_state += s_dim_idx * self.dim_bins**(self.state_dims - 1 - s_dim)
        return discretized_state
    
    def _get_continuous_state(self, state):
        continuous_state = ()
        remainder = state
        for s_dim in range(self.state_dims):
            dim_state = remainder // self.dim_bins**(self.state_dims - 1 - s_dim)
            remainder -= dim_state * self.dim_bins**(self.state_dims - 1 - s_dim)
            dim_continuous_state = (self.gym_env.observation_space.low[s_dim] + \
                dim_state*self.bins_size[s_dim]) + np.random.rand()*self.bins_size[s_dim]
            continuous_state += (dim_continuous_state,)
        return continuous_state
    
    def reset(self):
        new_continuous_state, _ = self.gym_env.reset()
        return self._get_discrete_state(new_continuous_state), new_continuous_state
    
    def step(self, continuous_state, action):
        new_continuous_state, reward, terminated, _, _ = self.gym_env.step(continuous_state, action)
        return self._get_discrete_state(new_continuous_state), reward, terminated, None, new_continuous_state
