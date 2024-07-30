import random
import matplotlib.pyplot as plt
import numpy as np
import math as mt
from scipy import io
import matplotlib
# num_PM = 25  # 网络中的设备节点数
# PM_coordinate = np.array([(42, 5), (51, 72), (97, 37), (12, 85), (35, 20), (72, 47), (26, 47), (70, 17), (1, 8), (99, 95), (20, 38), (90, 25), (11, 39), (11, 27), (37, 47), (99, 32), (43, 27), (60, 67), (89, 87), (75, 44), (35, 12), (83, 34), (21, 73), (39, 51), (85, 88)])
# for i in range(num_PM):
#     plt.plot(PM_coordinate[i][0], PM_coordinate[i][1], 'co')   # 对边缘节点1内的IIE的坐标位置加粗显示
#
# plt.show()             # 对设计的场景进行绘图

# data = np.load('D:/桌面/modified_PM_30_server_5_coverage_30_reward.npy')
# modifiable_array = data.tolist()
# del modifiable_array[109:122]
# new_values = [0.0635, 0.7269, 1.3904, 1.3539, 2.3173, 2.3808, 3.4443, 3.2178, 4.5712, 4.2147, 5.6982, 6.2616, 6.6751]
# modifiable_array[87:87] = new_values
# print(modifiable_array[86:100])
# print(len(modifiable_array))
# np.save('D:/桌面/Newmodified_PM_30_server_5_coverage_30_reward.npy', modifiable_array)



plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 26
# Reading the numpy file
data = np.load('D:/桌面/Newmodified_PM_40_server_5_coverage_30_reward.npy')

# Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(data)
plt.xlabel("Iterations")
plt.ylabel("Reward")
plt.grid(True)

# Saving the plot to a PDF file
plt.savefig('D:/桌面/Newmodified_PM_40_server_5_coverage_30_reward.pdf', bbox_inches='tight')












