import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib的全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 34
# 更新区域大小为 140x140
width, height = 140, 140

## 生产设备的坐标

device_coords = np.array([
    [101, 123], [93, 36], [102, 61], [82, 62], [77, 39], [72, 101], [117, 43], [99, 110], [35, 124],
    [46, 65], [44, 82], [120, 55], [65, 104], [32, 73], [60, 96], [43, 91], [106, 94], [112, 103],
    [37, 99], [111, 48], [92, 113], [40, 109], [68, 107], [55, 46], [129, 78], [48, 31], [80, 125],
    [115, 115], [47, 118], [113, 74], [63, 111], [118, 30], [105, 53], [125, 112], [66, 75], [130, 44],
    [95, 105], [51, 37], [124, 121], [84, 59], [90, 71], [30, 97], [64, 117], [56, 119], [67, 51],
    [58, 40], [107, 100], [94, 67], [70, 87], [62, 86]
])  # PM=50

# device_coords = np.load('D:/桌面/新仿真记录/PM坐标/PM_30/PM_30_server_4_coverage_30_coordinate.npy')  # PM = 30
# device_coords = np.load('D:/桌面/新仿真记录/PM坐标/PM_40/PM_40_server_4_coverage_30_coordinate.npy')    # PM = 40
# device_coords = np.load('D:/桌面/新仿真记录/PM坐标/PM_20/PM_20_server_3_coverage_30_coordinate.npy')    # PM = 20

## 边缘服务器的坐标

server_coords = np.array([
    [50.875275, 110.39109], [102.54313, 117.15244], [69.3675, 41.13763],
    [112.84, 56.54827], [49.15271, 82.796005]
])  # PM = 50时IES坐标

# server_coords = np.array([
#     [122.68829, 48.295547], [47.34478, 107.9806], [112.79568, 97.137566],
#     [49.156425, 49.298428], [79.32416, 103.59723]
# ])  # PM = 30时IES坐标

# server_coords = np.array([
#     [109.73516, 113.41782], [108.41943, 66.09323], [43.676716, 79.19477],
#     [74.25159, 49.631165], [69.86078, 103.73297]
# ])  # PM = 40时IES坐标

# server_coords = np.array([
#     [70.4949,  102.836655], [49.16765, 59.437004], [94.808914, 58.248077],
#     [120.373535, 88.80694]
# ])  # PM = 20时IES坐标

## 分配矩阵
allocation = np.array([
    1, 2, 3, 2, 2, 0, 3, 1, 0, 4, 4, 3, 0, 4, 0, 4, 1, 1, 0, 3, 1, 0, 0, 2, 3,
    2, 1, 1, 0, 3, 0, 3, 3, 1, 4, 3, 1, 2, 1, 2, 3, 4, 0, 0, 2, 2, 1, 3, 4, 4
])  # PM = 50时IES分配矩阵

# allocation = np.load('D:/桌面/新仿真记录/分配方案/PM_30_server_5_coverage_30_allocation.npy')  # PM = 30时IES分配矩阵

# allocation = np.load('D:/桌面/新仿真记录/分配方案/PM_40_server_5_coverage_30_allocation.npy')  # PM = 40时IES分配矩阵

# allocation = np.load('D:/桌面/新仿真记录/分配方案/PM_20_server_4_coverage_30_allocation.npy')  # PM = 20时IES分配矩阵


## 设备颜色映射，更新黄色为紫色
# colors = ['green', 'blue', 'black', 'red', 'orange'] # thistle PM=30
# colors = ['green', 'orange', 'blue', 'red', 'blue'] # thistle PM=20
# colors = ['blue', 'green', 'black', 'orange', 'red']  # thistle PM=40
colors = ['orange', 'black', 'red', 'green', 'blue']  # thistle PM=40


# 创建图形
plt.figure(figsize=(10, 10))
ax = plt.gca()

# 画设备
for i, coord in enumerate(device_coords):
    ax.plot(coord[0], coord[1], 'o', color=colors[allocation[i]], markersize=12)

# 画服务器和覆盖范围
for coord in server_coords:
    ax.plot(coord[0], coord[1], '*', markersize=24, color='black')
    circle = plt.Circle((coord[0], coord[1]), 30, color='black', fill=False, linestyle=(0, (5, 10)))
    ax.add_artist(circle)

# 设置坐标轴，包括整个矩形边界
ax.set_xlim([0, width])
ax.set_ylim([0, height])
x = range(0, 141, 20)
plt.xticks(x)


ax.set_xlabel('X Coordinate', fontname='Times New Roman')
ax.set_ylabel('Y Coordinate', fontname='Times New Roman')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# 设置图片标题
# ax.set_title('Edge Server Coverage for Production Devices in 140x140 Area')

# 显示整个矩形边界
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# 保存图形为SVG文件
plt.savefig('D:/cooperation/231008 Second Version/231008 Second Version/figures/PM50IES5.pdf', bbox_inches='tight')

# 显示图形
plt.show()
