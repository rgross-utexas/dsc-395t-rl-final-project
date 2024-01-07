import gymnasium as gym
from gymnasium import Env
import numpy as np
from tqdm import tqdm

from utils.function_approximation import StateActionFeatureVectorWithTile


def sarsa_lambda(
        env: Env,  # openai gym environment
        gamma: float,  # discount factor
        lam: float,  # decay rate
        alpha: float,  # step size
        X: StateActionFeatureVectorWithTile,
        num_episodes: int,
) -> np.array:

    # id = "tb/4_eps_05_normal_3_0"
    # if TB:
    #     tb_logger = tb.SummaryWriter(id, flush_secs=5)

    def epsilon_greedy_policy(_s, _done, _w, _epsilon=.0):

        num_actions = env.action_space.n
        if np.random.rand() < _epsilon:
            return np.random.randint(num_actions)
        else:
            q = [np.dot(w, X(_s, _a, _done)) for _a in range(num_actions)]
            return np.argmax(q)

    # initialize weight vector
    w = np.reshape(np.zeros((X.feature_vector_len())), [1, -1])

    counter = 0
    for e in tqdm(range(num_episodes)):

        # if DEBUG:
        #     print(f'Processing episode {e + 1}...')

        s = env.reset()
        done = False
        a = epsilon_greedy_policy(_s=s, _done=done,
                                  _w=w, _epsilon=0.05)
        x = X(s, a, done)

        z = np.zeros_like(w)
        q_old = 0

        t = 0
        while True:

            t += 1

            s_prime, r, done, _, _ = env.step(a)
            a_prime = epsilon_greedy_policy(_s=s_prime, _done=done,
                                            _w=w, _epsilon=0.05)
            x_prime = X(s_prime, a_prime, done)

            q = float(np.dot(w, x))
            q_prime = float(np.dot(w, x_prime))

            delta = r + gamma * q_prime - q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x.T
            w = w + alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x.T

            q_old = q_prime
            x = x_prime
            a = a_prime

            counter += 1

            if done:
                break

        # if TB:
        #     tb_logger.add_scalar('Q', Q, global_step=counter)
        #     tb_logger.add_scalar('delta', delta, global_step=counter)
        #     tb_logger.add_scalar('e_length', e_length, global_step=counter)
        #
        # if DEBUG:
        #     print(f'Processed episode of length {e_length}.')

    # if DEBUG:
    #     print(f'id: {id}')

    return w


if __name__ == '__main__':

    env = gym.make("MountainCar-v0", render_mode="human")
    env.metadata['render_fps'] = 120

    gamma = 1.

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        num_tile_parts=4
    )

    w = sarsa_lambda(env, gamma, 0.8, 0.01, X, 2000)

    def greedy_policy(s, done):
        Q = [np.dot(w, X(s, done, a)) for a in range(env.action_space.n)]
        return np.argmax(Q)


    def _eval(render=True):
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s, done)
            s, r, done, _ = env.step(a)
            if render: env.render()

            G += r

        return G


    Gs = [_eval() for _ in range(100)]
    G_rendered = _eval(True)
