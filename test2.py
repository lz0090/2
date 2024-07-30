
# coding:utf-8
import math

import numpy as np
import math as mt
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces

from safety_distance import safe_distance_judgment  # 判断边缘服务器的位置是否在设备节点的安全范围内
from Connect_server import connect_server_judgment  # 判断设备节点是否具备连通性并求出每个设备节点所连接的边缘服务器
from load_constraint import load_judgment           # 求出每个服务器的负载总量
import config


' 系统环境变量 '
# ****************************************************************************************************
global point_flag  # 设置全局设备节点分配变量
global current_max  # 设置全局当前最大簇内时延
global last_max  # 设置全局上一次最大簇内时延


num_PM = config.glo_num_PM  # 网络中的设备节点数
num_server = config.glo_num_server  # 网络中的边缘服务器数目
coverage_server = config.glo_coverage_server  # 边缘服务器的覆盖范围

# PM_coordinate = config.glo_PM_coordinate # 边缘节点的坐标
PM_coordinate = np.load('D:/桌面/新仿真记录/PM坐标/PM_20/PM_20_server_3_coverage_30_coordinate.npy')

power_tx_server = 35  # 边缘服务器发射功率为35dBm
power_tx_PM = 10  # 设备节点发射功率为10dBm
delta_coverage = 1  # 设备节点的安全距离，在此安全距离内不允许布置RSU
data_trans = 10 * np.power(10, 6)  # 所有设备节点的上行传输的数据量为10M    传输任务量
data_comp = 10 * np.power(10, 6)  # 计算任务量为10M
N_0 = -174  # 噪声功率为-174dBm/Hz
n_loss = 4  # 路径损耗指数
d_0 = 1  # 路径损耗参考距离1m
c_0 = 3 * np.power(10, 8)  # 光速
f_0 = 2.4 * np.power(10, 9)  # 理应是每个子载波中心频率都不一样，但中心频率远大于可分配的带宽，因此差异可以忽略不计
xigema_shadow = 3.39  # 阴影衰落3.39dB
t_0 = 0.1  # 测试时延设为100ms  现在已经没用了
t_pre = 0.01565  #  第一部分认证时延为15.65ms
t_auth = 0.003733  # 第二部分认证时延，随连接设备数增大而增大

bandwith = 50 * np.power(10, 6)  # 边缘服务器给每个设备节点分配的带宽为50MHz
C_max = 15 * 10 * np.power(10, 6)  # IES最大计算能力  我们仿真时设每一个边缘服务器的计算能力相同
load_eta  = np.float64(num_server * ((num_PM * (data_comp / (1 * np.power(10, 6))) ) / num_server) ** 2)  # 负载均衡约束阈值
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

        # # 以边缘服务器在整个100x100m中的部署坐标为动作空间的取值，动作空间是一个(num_server * 2)*1维的空间 将原本的2维空间变为1维空间，但长度要变为原来的2倍
        # self.low = coverage_server * np.ones((num_server * 2, 1), dtype=np.float16)  # 动作空间的下限，取值为(coverage_server,coverage_server)
        # self.high = (100 - coverage_server) * np.ones((num_server * 2, 1), dtype=np.float16)  # 动作空间的上限，取值为最远点(100-coverage_server,100-coverage_server)
        # self.action_space = spaces.Box(self.low, self.high, dtype=np.float16)

        # 一个100x100m空间作为动作空间的取值，动作空间是一个(num_server * 2)*1维的空间 将原本的2维空间变为1维空间，但长度要变为原来的2倍
        self.low = coverage_server * np.ones((num_server * 2, 1), dtype=np.float16)  # 动作空间的下限，取值为(coverage_server,coverage_server)
        self.high = (100 + coverage_server) * np.ones((num_server * 2, 1), dtype=np.float16)  # 动作空间的上限，取值为最远点(200+coverage_server,200+coverage_server)
        self.action_space = spaces.Box(self.low, self.high, dtype=np.float16)

    def step(self, action):

        state = self.state  # 获取当前状态
        reward = 0  # 定义初始奖励值为0
        done = False  # 判断一个episode是否结束
        flag_safety = 0  # 定义判断边缘服务器是否在设备节点安全范围内的变量，一旦有个边缘服务器部署在了某个设备节点的安全范围内flag_safety的值加1，其最大值为num_server
        server_load = []  # 创建边缘服务器负载列表，用于存储每个边缘服务器的负载值
        PM_delay = np.zeros((num_PM, 1), dtype=np.float16)  # 创建设备时延数组，存储所有设备节点的上行时延
        server_delay = np.zeros((num_server, 1), dtype=np.float16)  # 创建簇内总时延数组，用于存储每个簇内总时延


        flag_server = 0 # 定义边缘服务器坐标位置满足变量，若有一个边缘服务器的xy坐标满足要求，变量值加1，正常情况flag_server的值应该等于
        flag_eta = 0  # 定义负载均衡约束满足变量，0代表不满足，1代表满足
        flag_compu = 0  # 定义IES不过载约束满足变量，0代表不过载，1代表计算过载

        for i in range(num_server): # 判断action（即边缘服务器位置）是否部署在适宜位置
            if coverage_server <= action[2 * i] and action[2 * i] <= (200 + coverage_server) and coverage_server <= action[2 * i + 1] and action[2 * i + 1] <= (200 + coverage_server):
                flag_server += 1

        if flag_server != num_server: # 判断边缘服务器位置是否都正确
            next_state = state

            reward += -(num_server - flag_server) * 1000  # 改动前的奖励函数
            # reward +=  flag_server * 1       # 改动后的奖励函数

            self.state = next_state
            info = {'error':2}
            return next_state, reward, done, info


        flag_connect, PM_allocation = connect_server_judgment(PM_coordinate, num_PM, num_server, coverage_server, action)  # 由action值判断此时的设备节点连通性,flag_connect值代表具备连通性的设备节点的数目，PM_allocation是节点分配数组，里面的值表示设备节点连接的边缘服务器如0,1,2...

        ######   首先针对action的值进行设备节点的连通性判断，所有设备都具备连通性具有最高优先级,flag_connect不等于num_PM说明存在设备节点不具备连通性
        if flag_connect != num_PM:
            next_state = state

            reward += - (num_PM - flag_connect) * 500  # 根据num_PM和flag_connect之间的差值设置惩罚    改动前的奖励函数
            # reward +=   flag_connect * 4    # 改动后的奖励函数

            self.state = next_state
            info = {'error':[flag_connect,PM_allocation]}
            return next_state, reward, done, info

        np.save('./PM_allocation/PM_{}_server_{}_coverage_{}_allocation.npy'.format(config.glo_num_PM, config.glo_num_server, config.glo_coverage_server), PM_allocation)

        server_load = load_judgment(num_server, PM_allocation, data_comp, C_max, load_eta)   #  计算出每个边缘服务器的负载值
        flag_load = 0 # 负载超出变量，存储边缘服务器超出最大计算量的部分，用于进行奖励反馈
        for i in range(num_server):
            if server_load[i] > C_max:
                flag_load += (server_load[i] - C_max)    #   如果边缘服务器i的计算量大于最大计算量C_max，将超过量加入flag_load中

        #####   对是否过载进行判断，当flag_load不等于0时说明存在服务器过载，对过载量进行加法惩罚
        if flag_load == 0:
            flag_compu = 0
        else:
            flag_compu = 1
            next_state = state

            reward += - (flag_load / data_comp) * 100  # 对过载量进行加法惩罚    改动前的奖励函数
            # reward += - (flag_load / data_comp) * 7  # 改动前奖励函数

            self.state = next_state
            info = {'error': server_load}
            return next_state, reward, done, info


        ######   然后在所有设备都具备连通性的前提下判断是否有边缘服务器部署在某个设备节点的安全距离范围内
        for i in range(num_server):
            flag_safety += safe_distance_judgment(PM_coordinate, num_PM, delta_coverage, action[2 * i], action[2 * i + 1])

        if flag_safety != 0 and flag_compu == 0:
            next_state = state

            reward += - flag_safety * 1  # 根据flag_safety的值设置惩罚  改动前的奖励函数
            # reward += - flag_safety * 5   # 改动后的奖励函数

            self.state = next_state
            info = {'error': flag_safety}
            return next_state, reward, done, info

        ######  计算时延
        for i in range(num_server):
            index = [k for k, val in enumerate(PM_allocation) if val == i]  # 找出PM_allocation中哪些位置值为i（i即为对应连接的边缘服务器）
            tmp_delay = []  # 中间存储列表，用于存储时延数据

            for l in range(len(index)):  # 求出每个设备节点到对应边缘服务器的传输时延和每个簇内总时延
                PM_delay[index[l]] = data_trans / (   bandwith * mt.log(1 + 10 **
                                                                        ((power_tx_PM -  (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + 10 * n_loss * mt.log(mt.sqrt(mt.pow(PM_coordinate[index[l]][0] - action[2 * i], 2) + mt.pow(PM_coordinate[index[l]][1] - action[2 * i + 1], 2)) / d_0, 10) + xigema_shadow)
                                                                          - (N_0 + 10 * mt.log(bandwith, 10)) - ((1.25 ** (num_server - 1)) * (num_server - 1))   ) / 10)
                                                                        , 2))
################ 测试用
                # tmp1 = (20 * mt.log(((4 * mt.pi * f_0 * d_0) / c_0), 10) + 10 * n_loss * mt.log(mt.sqrt(mt.pow(PM_coordinate[index[l]][0] - action[2 * i], 2) + mt.pow(PM_coordinate[index[l]][1] - action[2 * i + 1], 2)) / d_0, 10) + xigema_shadow)
                # tmp2 = (N_0 + 10 * mt.log(bandwith, 10))
                # tmp_IF =
                # tmp3 = power_tx_PM - tmp1 - tmp2
                # tmp4 = 10 ** (tmp3 / 10)
                # tmp5 = mt.log(1 + tmp4, 2)
                # tmp6 = bandwith * tmp5
                # tmp7 = data_trans / tmp6



                tmp_delay.append(float(PM_delay[index[l]]))  # 将簇i内的设备节点的上行时延存储到tmp_delay中
                server_delay[i] += t_auth  # 先将簇i内的认证时间累计求和
            if index:
                server_delay[i] += max(tmp_delay) + t_pre + (0.040125 * len(index) - 0.055875 ) # 加上簇i内的最大上行时延和准备时延得到簇i的总时延


        ###### 计算负载均衡约束
        sum_differ = 0  # 存储请求量差的平方和
        server_load_tmp = []  #  存储负载的中间变量，单位为M
        for i in range(len(server_load)):
            server_load_tmp.append(server_load[i] / (1 * np.power(10, 6)))
        for i in range(num_server):
            sum_differ += ((server_load[i] / (1 * np.power(10, 6))) - (sum(server_load_tmp) / num_server)) ** 2  # 计算请求量差的平方和计算公式

        if sum_differ <= load_eta:
            flag_eta = 1

        ###### 判断是否满足负载均衡约束和IES不过载约束
        if flag_eta == 1 and flag_compu == 0:
            current_max = max(server_delay)  # 当前的最大簇内时延

            # reward = (delay_max / current_max) * 200 * (math.exp(- sum_differ) + 1)
            reward =  1000 * math.exp( - current_max )    # 改动后的奖励函数
            # reward =  num_PM * 5 * (delay_max / current_max) ** 2  # 改动后的奖励函数
            next_state = state
            done = True
            info = {'correct': current_max}

        else:
            # reward = -10
            reward = 0
            next_state = state
            info = {'error':flag_eta}
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