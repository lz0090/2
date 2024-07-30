import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from scipy.special import jv
from datetime import datetime
from get_RandomPoint_InCircle import getRandomPointInCircle  #导入在园内随机取点函数
from SplitNumber import split_number # 将数随机分成n份
from scipy import io

def AGV_trace(x):
    y = x**0.9+1/10*x
    return y



# a = np.load("E:/project/PPO/loc_delay/loc_delay_dynamic_psi.npy", allow_pickle=True)
# b = np.load("E:/project/PPO/loc_delay/loc_delay_static.npy", allow_pickle=True)
# c = np.load("E:/project/PPO/loc_delay/loc_delay_dynamic_psi.npy", allow_pickle=True)
# d = np.load("E:/project/PPO/data_train/PPO_continuous_Gaussian_env_MyEnv-v0_number_1_seed_10.npy", allow_pickle=True)
# e = np.load("E:/project/PPO/loc_action/loc_action_dynamic_all.npy", allow_pickle=True)
# io.savemat('evaluate_average_T_3e3.mat', {'evaluate_average_3e3': d})
# print(d)
