import numpy as np
from tqdm import tqdm

from algos.utils import choice_eps_greedy, NumpyReplayBuffer


class QLearning(object):

    def __init__(self, env, qlearning_args):
        self.env = env
        self.alpha_init = qlearning_args['alpha_init']
        self.alpha_final = qlearning_args['alpha_final']
        self.alpha_steps = qlearning_args['alpha_steps']
        self.gamma = qlearning_args['gamma']

        self.replay_buffer = NumpyReplayBuffer(size=qlearning_args['replay_buffer_size'])
        self.batch_size = qlearning_args['replay_buffer_batch_size']

    def train(self, learning_steps, episode_length=100, custom_reward=None):

        Q = np.zeros((self.env.num_states, self.env.num_actions))
        Q_old = np.copy(Q)
        rollouts_rewards = []
        rollouts_steps = []

        state, _ = self.env.reset()

        for step in tqdm(range(learning_steps)):

            # Calculate learning rate.
            fraction = np.minimum(step / self.alpha_steps, 1.0)
            alpha = self.alpha_init + fraction * (self.alpha_final - self.alpha_init)

            action = choice_eps_greedy(Q[state], epsilon=0.1)

            next_state, reward, terminated, _, _ = self.env.step(state, action)

            if custom_reward:
                reward = custom_reward(state, action)

            if terminated:
                # Reset.
                Q[state][action] += alpha * (reward - Q[state][action])
                state, _ = self.env.reset()
                
            elif step % episode_length == 0:
                # Reset.
                Q[state][action] += alpha * \
                        (reward + self.gamma * np.max(Q[next_state,:]) - Q[state][action])
                state, _ = self.env.reset()

            else:
                # Do not reset.
                Q[state][action] += alpha * \
                        (reward + self.gamma * np.max(Q[next_state,:]) - Q[state][action])
                state = next_state

            if step % 1_000 == 0:
                print('Alpha:', alpha)
                print('Q tab error:', np.sum(np.abs(Q-Q_old)))
                rollout_reward = self._execute_rollout(Q, episode_length, custom_reward)
                rollouts_rewards.append(rollout_reward)
                rollouts_steps.append(step)
                Q_old = np.copy(Q)

        data = {}
        data['Q_vals'] = Q
        data['rollouts_rewards'] = rollouts_rewards
        data['rollouts_steps'] = rollouts_steps

        return data
    
    def train_offline(self, learning_steps, num_samples_per_sa_pair, episode_length=100, custom_reward=None):

        print("num_samples_per_sa_pair", num_samples_per_sa_pair)
        # Prefill replay buffer.
        print('Pre-filling replay buffer.')
        count_samples = 0
        for _ in range(num_samples_per_sa_pair):
            for state in range(self.env.num_states):
                for action in range(self.env.num_actions):
                    s_t1, r_t1, terminated, _, _ = self.env.step(state, action)
                    if custom_reward:
                        r_t1  = custom_reward(state, action)
                    self.replay_buffer.add(state, action, r_t1, s_t1, done=terminated, info={})
                    count_samples += 1
        print(f"Inserted to replay buffer {count_samples} samples.")

        Q = np.zeros((self.env.num_states, self.env.num_actions))
        Q_old = np.copy(Q)
        rollouts_rewards = []
        rollouts_steps = []

        for step in tqdm(range(learning_steps)):

            # Calculate learning rate.
            fraction = np.minimum(step / self.alpha_steps, 1.0)
            alpha = self.alpha_init + fraction * (self.alpha_final - self.alpha_init)

            # Update.
            states, actions, rewards, next_states, dones, infos = self.replay_buffer.sample(self.batch_size)

            for i in range(self.batch_size):
                state, action, reward, next_state, done, info = \
                    states[i], actions[i], rewards[i], next_states[i], dones[i], infos[i]

                is_truncated = info.get('TimeLimit.truncated', False)
                # Q-learning update.
                if done and (not is_truncated):
                    Q[state][action] += alpha * (reward - Q[state][action])
                else:
                    Q[state][action] += alpha * \
                            (reward + self.gamma * np.max(Q[next_state,:]) - Q[state][action])

            if step % 1_000 == 0:
                print('Alpha:', alpha)
                print('Q tab error:', np.sum(np.abs(Q-Q_old)))
                rollout_reward = self._execute_rollout(Q, episode_length, custom_reward)
                rollouts_rewards.append(rollout_reward)
                rollouts_steps.append(step)
                Q_old = np.copy(Q)

        data = {}
        data['Q_vals'] = Q
        data['rollouts_rewards'] = rollouts_rewards
        data['rollouts_steps'] = rollouts_steps

        return data

    def _execute_rollout(self, Q_vals, episode_length, custom_reward):

        s_t, _ = self.env.reset()
        episode_cumulative_reward = 0

        for _ in range(episode_length):

            # Pick action.
            a_t = choice_eps_greedy(Q_vals[s_t], epsilon=0.0)

            # Env step.
            s_t1, r_t1, terminated, _, _ = self.env.step(s_t, a_t)

            #if terminated:
            #    break

            if custom_reward:
                r_t1 = custom_reward(s_t, a_t)

            episode_cumulative_reward += r_t1

            s_t = s_t1

        print('Rollout episode reward:', episode_cumulative_reward)
        return episode_cumulative_reward
