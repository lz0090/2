# coding:utf-8
import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from scipy.special import jv
from get_RandomPoint import generate_random_coordinates  # 随机生成设备节点的二维坐标
from safety_distance import safe_distance_judgment  # 判断边缘服务器的位置是否在设备节点的安全范围内
from Connect_server import connect_server_judgment  # 判断设备节点是否具备连通性并求出每个设备节点所连接的边缘服务器

' 系统环境变量 '
# ****************************************************************************************************
global point_flag  # 设置全局设备节点分配变量
global current_max  # 设置全局当前最大簇内时延
global last_max  # 设置全局上一次最大簇内时延

point_flag = 0
num_PM = 25  # 网络中的设备节点数
num_server = 3  # 网络中的边缘服务器数目
coverage_server = 30  # 边缘服务器的覆盖范围为30m

if point_flag == 0:
    PM_coordinate = generate_random_coordinates(coverage_server, num_PM)  # 根据
    point_flag = 1  # 全局变量置1，代表每次程序运行只执行一次

power_tx_server = 35  # 边缘服务器发射功率为35dBm
power_tx_PM = 10  # 设备节点发射功率为10dBm
delta_coverage = 1  # 设备节点的安全距离，在此安全距离内不允许布置RSU
data_trans = 50 * np.power(10, 6)  # 所有设备节点的上行传输的数据量为50M    传输任务量
data_comp = 50 * np.power(10, 6)  # 计算任务量为50M
N_0 = -100  # 噪声功率为-100dBm
n_loss = 4  # 路径损耗指数
d_0 = 1  # 路径损耗参考距离1m
c_0 = 3 * np.power(10, 8)  # 光速
f_0 = 2.4 * np.power(10, 9)  # 理应是每个子载波中心频率都不一样，但中心频率远大于可分配的带宽，因此差异可以忽略不计
xigema_shadow = 3.39  # 阴影衰落3.39dB
t_0 = 0.01  # 认证时延设为10ms
bandwith = 20 * np.power(10, 6)  # 边缘服务器给每个设备节点分配的带宽为20MHz
load_eta = 2  # 负载均衡约束阈值
C_max = 12 * 50 * np.power(10, 6)  # IES最大计算能力

delay_max = num_PM * t_0  # 设置一个可能的最大簇内时延常量


#  ****************************************************************************************************


class MyEnv(gym.Env):
    def __init__(self):

        self.viewer = None
        self.num_PM = num_PM
        self.num_server = num_server
        self.PM_coordinate = PM_coordinate
        self.coverage_server = coverage_server
        self.N_0 = N_0
        self.delta_coverage = delta_coverage
        self.power_tx_server = power_tx_server
        self.power_tx_PM = power_tx_PM
        self.bandwith = bandwith  # 初始化赋值

        # 以边缘服务器在整个100x100m中的部署坐标为动作空间的取值，动作空间是一个(num_server * 2)*1维的空间 将原本的2维空间变为1维空间，但长度要变为原来的2倍
        self.low = coverage_server * np.ones((num_server * 2, 1), dtype=np.float16)  # 动作空间的下限，取值为(coverage_server,coverage_server)
        self.high = (100 - coverage_server) * np.ones((num_server * 2, 1), dtype=np.float16)  # 动作空间的上限，取值为最远点(100-coverage_server,100-coverage_server)
        self.action_space = spaces.Box(self.low, self.high, dtype=np.float16)

    def step(self, action):

        state = self.state  # 获取当前状态
        reward = 0  # 定义初始奖励值为0
        done = False  # 判断一个episode是否结束
        flag_safety = 0  # 定义判断边缘服务器是否在设备节点安全范围内的变量，一旦有个边缘服务器部署在了某个设备节点的安全范围内flag_safety的值加1，其最大值为num_server
        server_load = []  # 创建边缘服务器负载列表，用于存储每个边缘服务器的负载值
        PM_delay = np.zeros((num_PM, 1), dtype=np.float16)  # 创建设备时延数组，存储所有设备节点的上行时延
        server_delay = np.zeros((num_server, 1), dtype=np.float16)  # 创建簇内总时延数组，用于存储每个簇内总时延

        flag_eta = 0  # 定义负载均衡约束满足变量，0代表不满足，1代表满足
        flag_compu = 0  # 定义IES不过载约束满足变量，0代表不过载，1代表计算过载

        flag_connect, PM_allocation = connect_server_judgment(PM_coordinate, num_PM, num_server, coverage_server, action)  # 由action值判断此时的设备节点连通性,flag_connect值代表具备连通性的设备节点的数目，PM_allocation是节点分配数组，里面的值表示设备节点连接的边缘服务器如0,1,2...

        ######   首先针对action的值进行设备节点的连通性判断，所有设备都具备连通性具有最高优先级,flag_connect不等于num_PM说明存在设备节点不具备连通性
        if flag_connect != num_PM:
            next_state = state
            reward += - (num_PM - flag_connect) * 100  # 根据num_PM和flag_connect之间的差值设置惩罚
            self.state = next_state
            info = {}
            return next_state, reward, done, info

        ######   然后在所有设备都具备连通性的前提下判断是否有边缘服务器部署在某个设备节点的安全距离范围内
        for i in range(num_server):
            flag_safety += safe_distance_judgment(PM_coordinate, num_PM, delta_coverage, action[2 * i], action[2 * i + 1])

        if flag_safety != 0:
            next_state = state
            reward += - flag_safety * 70  # 根据flag_safety的值设置惩罚
            self.state = next_state
            info = {}
            return next_state, reward, done, info

        ######  计算负载
        for i in range(num_server):
            index = [k for k, val in enumerate(PM_allocation) if val == i]  # 找出PM_allocation中哪些位置值为i（i即为对应连接的边缘服务器）
            tmp_load = 0  # 中间存储变量
            for l in range(len(index)):
                tmp_load += data_comp  # 计算边缘服务器i的负载
            server_load.append(tmp_load)  # 将边缘服务器i的负载数据添加到列表中

        ######  计算时延
        for i in range(num_server):
            index = [k for k, val in enumerate(PM_allocation) if val == i]  # 找出PM_allocation中哪些位置值为i（i即为对应连接的边缘服务器）
            tmp_delay = []  # 中间存储列表，用于存储时延数据

            for l in range(len(index)):  # 求出每个设备节点到对应边缘服务器的传输时延和每个簇内总时延
                PM_delay[index[l]] = data_trans / (bandwith * mt.log(1 + 10 ** ((power_tx_PM - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + 10 * n_loss * mt.log(mt.sqrt(mt.pow(PM_coordinate[index[l]][0] - action[2 * i], 2) + mt.pow(PM_coordinate[index[l]][1] - action[2 * i + 1], 2)) / d_0, 10) + xigema_shadow) - N_0) / 10), 2))
                tmp_delay.append(PM_delay[index[l]])  # 将簇i内的设备节点的上行时延存储到tmp_delay中
                server_delay[i] += t_0  # 先将簇i内的认证时间累计求和
            server_delay[i] += max(tmp_delay)  # 加上簇i内的最大上行时延得到簇i的总时延

        ###### 计算负载均衡约束
        sum_differ = 0  # 存储请求量差的平方和
        for i in range(num_server):
            sum_differ += (server_load[i] - sum(server_load) / num_server) ** 2  # 计算请求量差的平方和计算公式

        if sum_differ <= load_eta:
            flag_eta = 1

        ###### 计算IES不过载约束
        for i in range(num_server):
            if server_load[i] > C_max:  # 只要有IES的计算负载大于其最大计算能力，flag_compu就置1
                flag_compu = 1

        ###### 判断是否满足负载均衡约束和IES不过载约束
        if flag_eta == 1 and flag_compu == 0:
            current_max = max(server_delay)  # 当前的最大簇内时延
            reward = (delay_max / current_max) * 200
            next_state = state
            info = {}

        else:
            reward = flag_eta * 20 + (1 - flag_compu) * 20
            next_state = state
            info = {}
        return next_state, reward, done, info

    def reset(self):
        self.state = np.array([1])

        return self.state

    def render(self, mode="human"):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# # coding:utf-8
# import numpy as np
# import math as mt
# import random
# import matplotlib.pyplot as plt
# import gym
# from gym import spaces
# from scipy.special import jv
# from get_RandomPoint import generate_random_coordinates    # 随机生成设备节点的二维坐标
# from safety_distance import safe_distance_judgment         # 判断边缘服务器的位置是否在设备节点的安全范围内
# from Connect_server import connect_server_judgment         # 判断设备节点是否具备连通性并求出每个设备节点所连接的边缘服务器
#
#
#
# ' 系统环境变量 '
# # ****************************************************************************************************
# global point_flag         # 设置全局设备节点分配变量
# global current_max        # 设置全局当前最大簇内时延
# global last_max           # 设置全局上一次最大簇内时延
#
# point_flag = 0
# num_PM = 25               # 网络中的设备节点数
# num_server = 3            # 网络中的边缘服务器数目
# coverage_server = 30      # 边缘服务器的覆盖范围为30m
#
# if point_flag == 0:
#     PM_coordinate = generate_random_coordinates(coverage_server, num_PM)  # 根据
#     point_flag = 1        # 全局变量置1，代表每次程序运行只执行一次
#
# power_tx_server = 35      # 边缘服务器发射功率为35dBm
# power_tx_PM = 10          # 设备节点发射功率为10dBm
# delta_coverage = 1        # 设备节点的安全距离，在此安全距离内不允许布置RSU
# data_trans = 50 * np.power(10, 6)                                           # 所有设备节点的上行传输的数据量为50M    传输任务量
# data_comp = 50 * np.power(10, 6)                                            # 计算任务量为50M
# N_0 = -100                # 噪声功率为-100dBm
# n_loss = 4                # 路径损耗指数
# d_0 = 1                   # 路径损耗参考距离1m
# c_0 = 3 * np.power(10, 8) # 光速
# f_0 = 2.4 * np.power(10, 9)   # 理应是每个子载波中心频率都不一样，但中心频率远大于可分配的带宽，因此差异可以忽略不计
# xigema_shadow = 3.39      # 阴影衰落3.39dB
# t_0 = 0.01                # 认证时延设为10ms
# bandwith = 20 * np.power(10, 6)  # 边缘服务器给每个设备节点分配的带宽为20MHz
# load_eta = 2              # 负载均衡约束阈值
# C_max = 12 * 50 * np.power(10, 6)  # IES最大计算能力
#
# delay_max = num_PM * t_0  # 设置一个可能的最大簇内时延常量
#
#
# #  ****************************************************************************************************
#
#
# class MyEnv(gym.Env):
#     def __init__(self):
#
#         self.viewer = None
#         self.num_PM = num_PM
#         self.num_server = num_server
#         self.PM_coordinate = PM_coordinate
#         self.coverage_server = coverage_server
#         self.N_0 = N_0
#         self.delta_coverage = delta_coverage
#         self.power_tx_server = power_tx_server
#         self.power_tx_PM = power_tx_PM
#         self.bandwith = bandwith                   # 初始化赋值
#
#         # 以边缘服务器在整个100x100m中的部署坐标为动作空间的取值，动作空间是一个num_server*2维的空间
#         self.low = coverage_server * np.ones((num_server,2), dtype=np.float16)                             # 动作空间的下限，取值为(coverage_server,coverage_server)
#         self.high = (100 - coverage_server) * np.ones((num_server, 2), dtype=np.float16)                   # 动作空间的上限，取值为最远点(100-coverage_server,100-coverage_server)
#         self.action_space = spaces.Box(self.low, self.high, dtype=np.float16)
#
#
#
#
#
#     def step(self, action):
#
#         state = self.state   # 获取当前状态
#         reward = 0           # 定义初始奖励值为0
#         done = False         # 判断一个episode是否结束
#         flag_safety = 0      # 定义判断边缘服务器是否在设备节点安全范围内的变量，一旦有个边缘服务器部署在了某个设备节点的安全范围内flag_safety的值加1，其最大值为num_server
#         server_load = []     # 创建边缘服务器负载列表，用于存储每个边缘服务器的负载值
#         PM_delay = np.zeros((num_PM, 1), dtype=np.float16)        # 创建设备时延数组，存储所有设备节点的上行时延
#         server_delay = np.zeros((num_server, 1), dtype=np.float16)    # 创建簇内总时延数组，用于存储每个簇内总时延
#
#         flag_eta = 0         # 定义负载均衡约束满足变量，0代表不满足，1代表满足
#         flag_compu = 0       # 定义IES不过载约束满足变量，0代表不过载，1代表计算过载
#
#         flag_connect, PM_allocation = connect_server_judgment(PM_coordinate, num_PM, num_server, coverage_server, action)  # 由action值判断此时的设备节点连通性,flag_connect值代表具备连通性的设备节点的数目，PM_allocation是节点分配数组，里面的值表示设备节点连接的边缘服务器如0,1,2...
#
# ######   首先针对action的值进行设备节点的连通性判断，所有设备都具备连通性具有最高优先级,flag_connect不等于num_PM说明存在设备节点不具备连通性
#         if flag_connect != num_PM :
#             next_state = state
#             reward += - (num_PM - flag_connect) * 100    # 根据num_PM和flag_connect之间的差值设置惩罚
#             self.state = next_state
#             info = {}
#             return next_state, reward, done, info
#
# ######   然后在所有设备都具备连通性的前提下判断是否有边缘服务器部署在某个设备节点的安全距离范围内
#         for i in range(num_server):
#             flag_safety += safe_distance_judgment(PM_coordinate, num_PM, delta_coverage, action[2 * i], action[2 * i + 1])
#
#         if flag_safety != 0:
#             next_state = state
#             reward += - flag_safety * 70                # 根据flag_safety的值设置惩罚
#             self.state = next_state
#             info = {}
#             return next_state, reward, done, info
#
# ######  计算负载
#         for i in range(num_server):
#             index = [k for k, val in enumerate(PM_allocation) if val == i]  # 找出PM_allocation中哪些位置值为i（i即为对应连接的边缘服务器）
#             tmp_load = 0    # 中间存储变量
#             for l in range(len(index)):
#                 tmp_load += data_comp             # 计算边缘服务器i的负载
#             server_load.append(tmp_load)          # 将边缘服务器i的负载数据添加到列表中
#
# ######  计算时延
#         for i in range (num_server):
#             index = [k for k, val in enumerate(PM_allocation) if val == i]  # 找出PM_allocation中哪些位置值为i（i即为对应连接的边缘服务器）
#             tmp_delay = []                              # 中间存储列表，用于存储时延数据
#
#             for l in range(len(index)):                 # 求出每个设备节点到对应边缘服务器的传输时延和每个簇内总时延
#                 PM_delay[index[l]] = data_trans / (bandwith * mt.log(1 + 10 ** ((power_tx_PM - (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + 10 * n_loss * mt.log(mt.sqrt(mt.pow(PM_coordinate[index[l]][0] - action[num_server][0], 2) + mt.pow(PM_coordinate[index[l]][1] - action[num_server][1], 2)) / d_0, 10) + xigema_shadow) - N_0) / 10), 2))
#                 tmp_delay.append( PM_delay[index[l]] )  # 将簇i内的设备节点的上行时延存储到tmp_delay中
#                 server_delay[i] += t_0                  # 先将簇i内的认证时间累计求和
#             server_delay[i] += max(tmp_delay)           # 加上簇i内的最大上行时延得到簇i的总时延
#
# ###### 计算负载均衡约束
#         sum_differ = 0  # 存储请求量差的平方和
#         for i in range(num_server):
#             sum_differ += (server_load[i] - sum(server_load) / num_server) **2 # 计算请求量差的平方和计算公式
#
#         if sum_differ <= load_eta:
#             flag_eta = 1
#
# ###### 计算IES不过载约束
#         for i in range(num_server):
#             if server_load[i] > C_max:           # 只要有IES的计算负载大于其最大计算能力，flag_compu就置1
#                 flag_compu = 1
#
# ###### 判断是否满足负载均衡约束和IES不过载约束
#         if flag_eta == 1 and flag_compu == 0:
#             current_max = max(server_delay)      # 当前的最大簇内时延
#             reward = (delay_max / current_max) * 200
#             next_state = state
#             info = {}
#
#         else:
#             reward = flag_eta * 20 + (1 - flag_compu) * 20
#             next_state = state
#             info = {}
#         return next_state, reward, done, info
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     def reset(self):
#         self.state = np.array([[1],[1]])
#
#
#         return self.state
#
#     def render(self, mode="human"):
#         return None
#
#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None
#
#
#
#
#
#
#
#
#
#
#
#
#
# # locx = dict()
# # tom = MyEnv()
# # tom.reset()
# # tom.step([5, 5, 5, 5, 5, 5, 5, 5, 5, 5,     3, 3,   3, 3, 3, 3, 2, 2,     2, 2, 2, 2,   19])
#
#
#
#
#
#
#
#
#
#











