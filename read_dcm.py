import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np

file_path = r"C:\Users\26522\Desktop\valid.h5"  # 'valid.h5'
with h5py.File(file_path, 'r') as file:
    print("Keys: %s" % list(file.keys()))
    # Keys: ['S', 'center', 'imgname', 'index', 'normalize', 'part', 'person', 'scale', 'torsoangle', 'visible', 'zind']
    # for k in file.keys():
    #     print(file[k])

    S = file['S'][:] # 对字典中的value使用[:]获取数据
    print(S[1])

    S_200 = S[:200]
    # 保存到txt文件
    np.savetxt('keypoints_200_frames.txt', S_200.reshape(-1, S_200.shape[-1]), fmt='%f')

    # 设置3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 设置坐标轴范围
    ax.set_xlim([np.min(S[:, :, 0]), np.max(S[:, :, 0])])
    ax.set_ylim([np.min(S[:, :, 1]), np.max(S[:, :, 1])])
    ax.set_zlim([np.min(S[:, :, 2]), np.max(S[:, :, 2])])

    # 初始化散点图
    sc = ax.scatter(S[0, :, 0], S[0, :, 1], S[0, :, 2], c='r', marker='o')

    # 循环更新关键点位置
    for i in range(len(S)):
        # 更新数据
        sc._offsets3d = (S[i, :, 0], S[i, :, 1], S[i, :, 2])

        # 暂停以创建动画效果
        plt.pause(0.05)  # 暂停0.5秒

        # 清除绘图
        ax.cla()

        # 重新绘制散点图
        sc = ax.scatter(S[i, :, 0], S[i, :, 1], S[i, :, 2], c='r', marker='o')

        # 重新设置坐标轴标签和范围
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim([np.min(S[:, :, 0]), np.max(S[:, :, 0])])
        ax.set_ylim([np.min(S[:, :, 1]), np.max(S[:, :, 1])])
        ax.set_zlim([np.min(S[:, :, 2]), np.max(S[:, :, 2])])

        ax.view_init(elev=-90, azim=-90)  # 使Y轴朝下
    # 显示图像
    plt.show()