# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

# from irlc.ex11.feature_encoder import DirectEncoder

# from irlc import train, main_plot
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from RL_gym import PlasticMaker

import GetProjectData

from sklearn.preprocessing import StandardScaler
import ml_formulation
import pandas as pd


# material index definitions:
mat_abs = 0
mat_ldpe = 1
mat_moplen = 2
data_material = GetProjectData.get_data()
data = data_material[mat_abs].iloc[:, 0:5]
# data = GetProjectData.get_concat_data().iloc[:, :-1]
# data.iloc[:,1:3] = np.zeros((data.shape[0], 2)) #Replaces feature 2 and 3 with zeros
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
ell = 2
def mkplastic():
    env = PlasticMaker(data.iloc[:, 0:3], data.iloc[:, 3:5], L=ell, apply_noise_to_y=True)
    from gymnasium.wrappers import TimeLimit
    env = TimeLimit(env, max_episode_steps=10000)
    return env

# import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: mkplastic()])
# env = mkplastic()
# Create environment
# env = gym.make('LunarLander-v2')

# Instantiate the agent
model = PPO('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=0.5e6)
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("dqn_lunar")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, mkplastic(), n_eval_episodes=3)

# Enjoy trained agent
env = mkplastic()
obs, _  = env.reset()
rewards = []
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, dones, _, info = env.step(action)
    rewards.append(-reward)
    if dones:
        break

import matplotlib.pyplot as plt
import numpy as np
plt.plot(rewards, label="Plot of cost for RL PPO model")
np.save("RL_rewards_500000", rewards)
plt.xlabel("time")
plt.ylabel("loss")
plt.legend()
plt.show()


#
# from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
# agent = LinearSemiGradQAgent(env, gamma=0.8, alpha=0.01, q_encoder=DirectEncoder(env))
#
# expq = 'experiments/direct_q_gent'
#
# res, _ = train(env, agent, expq, num_episodes=100, return_trajectory=False)
# x = "Episode"
#
# # train(env, agent, experiment_q, num_episodes=episodes, max_runs=10)
# main_plot(experiments=[expq], x_key=x, y_key='Accumulated Reward', smoothing_window=30, resample_ticks=100)
# # savepdf("semigrad_q")
# import matplotlib.pyplot as plt
# plt.show()
#
#
#
#
