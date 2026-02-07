import random
import numpy as np

def all_eq(values):
    # Returns True if every element of 'values' is the same.
    return all(np.isnan(values)) or max(values) - min(values) < 1e-6

def choice_eps_greedy(values, epsilon):
    if np.random.rand() <= epsilon or all_eq(values):
        return np.random.choice(len(values))
    else:
        return np.argmax(values)

class NumpyReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, info):
        data = (obs_t, action, reward, obs_tp1, done, info)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, info = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            infos.append(info)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(infos)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        infos: np.array
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

def sample_trajectory(env, Q_vals, epsilon, traj_length):

    states = []
    actions = []

    s, _ = env.reset()

    for t in range(traj_length):
        states.append(s)

        a = choice_eps_greedy(Q_vals[s], epsilon=epsilon)
        actions.append(a)

        s, _, _, _, _ = env.step(s, a)

    return states, actions


def sample_trajectory_from_policy(env, policy, traj_length):

    states = []
    actions = []

    s, _ = env.reset()

    for t in range(traj_length):
        states.append(s)

        a = np.random.choice(range(env.num_actions), p=policy[s,:])
        actions.append(a)

        s, _, _, _, _ = env.step(s, a)

    return states, actions


def estimate_d_pi_from_trajectory(env, trajectory_states, trajectory_actions, gamma):

    nS = env.num_states
    nA = env.num_actions

    d_hat = np.zeros((nS,nA))
    for t in range(len(trajectory_states)):
        s_t = trajectory_states[t]
        a_t = trajectory_actions[t]
        
        d_hat[s_t,a_t] += gamma**t * 1

    return ((1 - gamma)/(1 - gamma**(t+1))) * d_hat


def estimate_dpi(env, Q_vals, gamma, epsilon, num_episodes, episode_length=100):

    d_pi = np.zeros((env.num_states, env.num_actions))

    for k in range(num_episodes):
        traj_states, traj_actions = sample_trajectory(env, Q_vals, epsilon, traj_length=episode_length)
        d_pi += estimate_d_pi_from_trajectory(env, traj_states, traj_actions, gamma)
    
    return d_pi / num_episodes


def estimate_dpi_from_policy(env, policy, gamma, num_episodes, episode_length=100):

    d_pi = np.zeros((env.num_states, env.num_actions))

    for k in range(num_episodes):
        traj_states, traj_actions = sample_trajectory_from_policy(env, policy, traj_length=episode_length)
        d_pi += estimate_d_pi_from_trajectory(env, traj_states, traj_actions, gamma)
    
    return d_pi / num_episodes
