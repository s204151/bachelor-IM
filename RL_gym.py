import gymnasium as gym
import numpy as np
import MoldFlowY
import random
import GetProjectData
import pandas as pd
import ell_check
import matplotlib.pyplot as plt
from scipy import linalg as LA

from sklearn.preprocessing import StandardScaler
import ml_formulation

# https://gymnasium.farama.org/api/env/#gymnasium.Env.step
class PlasticMaker(gym.Env):

    def f_exact(self, x):
        # return y # compute using linear interpolation.
        return MoldFlowY.predict(x, pd.concat([self.X_exact, self.y_exact], axis=1))

    def __init__(self, X_exact, y_exact, noise=0.1, noise_drift=0.1, y_scale=(0.1, 0.5), apply_noise_to_y=True, max_steps=1000, L=10):
        self.n = X_exact.shape[1]
        self.m = y_exact.shape[1]
        self.apply_noise_to_y = apply_noise_to_y
        self.X_exact = X_exact
        self.y_exact = y_exact

        self.drift = np.zeros((self.m, ))
        self.noise_drift = noise_drift
        self.noise = noise
        self.y_scale = np.asarray(y_scale)
        self.max_steps = max_steps

        self.all_xs = []
        self.all_ys = []
        self.L = L

        self.cur_step = 0

        # Set observation space
        obs_space_lower = np.ones((self.L, self.n + self.m))
        obs_space_upper = np.ones((self.L, self.n + self.m))

        x0_min = np.min(X_exact.iloc[:, 0])
        x1_min = np.min(X_exact.iloc[:, 1])
        x2_min = np.min(X_exact.iloc[:, 2])

        x0_max = np.max(X_exact.iloc[:, 0])
        x1_max = np.max(X_exact.iloc[:, 1])
        x2_max = np.max(X_exact.iloc[:, 2])

        obs_lower_limits = [x0_min, x1_min, x2_min, -np.inf, -np.inf]
        obs_opper_limits = [x0_max, x1_max, x2_max, np.inf, np.inf]
        for i in range(len(obs_lower_limits)):
            obs_space_lower[:, i] *= obs_lower_limits[i]
            obs_space_upper[:, i] *= obs_opper_limits[i]

        self.observation_space = gym.spaces.Box(low=obs_space_lower,
                                                high=obs_space_upper,
                                                dtype=np.int64)

        # Set action space
        act_space_lower = np.ones(self.n)
        act_space_upper = np.ones(self.n)
        act_lower_limits = [x0_min, x1_min, x2_min]
        act_opper_limits = [x0_max, x1_max, x2_max]
        for i in range(len(act_space_lower)):
            act_space_lower[i] *= act_lower_limits[i]
            act_space_upper[i] *= act_opper_limits[i]

        self.action_space = gym.spaces.Box(low=act_space_lower,
                                           high=act_space_upper,
                                           dtype=np.int64)

    def reset(self):
        self.drift = self.drift * 0
        self.cur_step = 0
        dictionary = {}
        self.all_xs = []
        self.all_ys = []

        X_exact_np = self.X_exact.to_numpy()
        for l in range(self.L + 1):
            # Generate a random x
            x = [self._get_random_value(X_exact_np[:, 0]),
                 self._get_random_value(X_exact_np[:, 1]),
                 self._get_random_value(X_exact_np[:, 2])]
            y = self.f_exact(x)

            self.all_xs.append(x)
            self.all_ys.append(y)

        return self._make_observation(), dictionary

    # Get a random value between the minimum and the maximum value in the input array
    def _get_random_value(self, data_arr):
        return random.uniform(np.min(data_arr), np.max(data_arr))

    def _make_observation(self):
        all_zs = []
        for l in range(self.L + 1):
            # giver 5 dim. vector.
            all_zs.append(np.append(self.all_xs[-l], self.all_ys[-l]))
        # return self.L x 5 dimensionel matrix
        return np.stack(all_zs)

    def step(self, action):
        # action is a 3d vector.
        self.cur_step = self.cur_step + 1

        if self.apply_noise_to_y:
            x0 = self.f_exact(action)
            self.drift = self.drift + np.random.randn(self.m) * self.y_scale*self.noise_drift
            x1 = self.drift
            tau = 0.01
            self.drift = self.drift * (1-tau)
            x2 = np.random.randn(self.m) * self.noise
            # noise paa y.
            next_y = x0 + x1 + x2
        else:
            next_y = self.f_exact(action)

        # https://gymnasium.farama.org/api/env/
        # rl:
        truncated = False
        terminated = self.cur_step >= self.max_steps
        meta_information = {'drift': self.drift}

        self.all_xs.append(action)
        self.all_ys.append(next_y)

        next_observation = self._make_observation()
        next_y_target = next_observation[-1, -2:]*0
        # reward: evt. -| next_y - next_y_target|.
        # 0, 1 afhaengigt af om vi opfylder betingelser.
        # reward = 0
        reward = - np.abs(next_y - next_y_target)[0]
        return next_observation, reward, terminated, truncated, meta_information


def manual_evaluation(data_gen, env, L=10):
    # Train the model f_theta (in the LaTeX document)
    f_theta = ml_formulation.f_theta(data_gen.to_numpy(), L)

    state, _ = env.reset()  # s0 er en L x 5-matrix.
    rewards = []
    for t in range(200):
        # Get the last machine configuration x_old from state.
        state_flatten = state.flatten()
        # generate an action, i.e. x, using equation 1.
        epsilon_star = f_theta.epsilon_star(state_flatten[:-2])
        x_old = state_flatten[-5:-2]
        x = epsilon_star + x_old  # Nye handling.
        state, reward, terminated, _, _ = env.step(x)
        rewards.append(reward)

    plt.plot(rewards)
    plt.show()

    # reward_means = np.mean(rewards)
    # np.save("ml_rewards_newdata", rewards)
    # plt.plot(rewards)  # sammenlign med RL paa PlasticMaker miljoeet.


def plot_A(data, L=2):
    env = PlasticMaker(data.iloc[:, 0:3], data.iloc[:, 3:5], L=L, apply_noise_to_y=True)
    state, _ = env.reset()
    data_pandas = data
    data = data.to_numpy()
    f_theta = ml_formulation.f_theta(data, L)
    y_star = [np.mean(data[:, 3]), np.mean(data[:, 4])]

    reward_theta_all = []
    reward_surro_all = []

    start = np.min(data[:, 0]) - 5
    stop = np.max(data[:, 0]) + 5
    epsilons = np.linspace(start, stop, num=100)
    for e in epsilons:
        state_flatten = state.flatten()
        Z = state_flatten[:-2]

        reward_theta = np.abs(y_star - f_theta.predict(Z, np.array([e, 0, 0])))
        reward_theta = LA.norm(reward_theta, ord=2)
        reward_theta_all.append(reward_theta)

        x = state_flatten[-5:-2]
        reward_surro = np.abs(y_star - env.f_exact(x + np.array([e, 0, 0])))
        reward_surro = LA.norm(reward_surro, ord=2)
        reward_surro_all.append(reward_surro)

    fig, ax = plt.subplots()
    plt.xlabel("epsilon")
    plt.ylabel("loss")
    # plt.title("Plot A")
    ax.plot(epsilons, reward_theta_all, label="f_theta")
    ax.plot(epsilons, reward_surro_all, label="f_surrogate")
    ax.legend()
    plt.show()


def plot_B(data, L = 2):
    # Train the model f_theta (in the LaTeX document)
    f_theta = ml_formulation.f_theta(data.to_numpy(), L)

    # 2 different environments
    # baseline environment
    env_base = PlasticMaker(data.iloc[:, 0:3], data.iloc[:, 3:5], L=L, apply_noise_to_y=True)
    env_base.reset()

    # f_theta environment
    env_theta = PlasticMaker(data.iloc[:, 0:3], data.iloc[:, 3:5], L=L, apply_noise_to_y=True)
    state, _ = env_theta.reset()  # s0 er en L x 5-matrix.

    rewards_theta = []
    rewards_baseline = []
    for t in range(100):
        state_flatten = state.flatten()
        Z = state_flatten[:-2]

        e_star = f_theta.epsilon_star(Z)
        state, reward_t, _, _, _ = env_theta.step(e_star)
        reward_theta = LA.norm(reward_t, ord=2)
        rewards_theta.append(reward_theta)

        _, reward_b, _, _, _ = env_base.step([0, 0, 0])
        reward_base = LA.norm(reward_b, ord=2)
        # reward_base = np.sqrt(reward[0]**2 + reward[1]**2)
        rewards_baseline.append(reward_base)


    fig, ax = plt.subplots()
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.title("Plot B")
    ax.plot(np.linspace(0, 100, 100), np.asarray(rewards_baseline), label="baseline")
    ax.plot(np.linspace(0, 100, 100), rewards_theta, label="f_theta")
    ax.legend()
    plt.show()
    np.save("baseline_rewards", rewards_baseline)


if __name__ == "__main__":
    random.seed(1)

    # material index definitions:
    mat_abs = 0
    mat_ldpe = 1
    mat_moplen = 2

    material_to_use = "moplen"

    if material_to_use == "abs":
        data_material = GetProjectData.get_data()
        data = data_material[mat_abs].iloc[:, 0:5]
    elif material_to_use == "ldpe":
        data_material = GetProjectData.get_data()
        data = data_material[mat_ldpe].iloc[:, 0:5]
    elif material_to_use == "moplen":
        data_material = GetProjectData.get_data()
        data = data_material[mat_moplen].iloc[:, 0:5]
    elif material_to_use == "all":
        data = GetProjectData.get_concat_data().iloc[:, :-1]

    # test = (data - np.mean(data)) / np.std(data)

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    L = 2
    plot_A(data, L=L)
    plot_B(data, L=L)
