from copy import deepcopy

import cv2 as cv
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from camera_pose import *
from loguru import logger
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


@logger.catch()
def main():
    RATE = 1
    SKIP_FRAME = 2
    R1_RATIO = 2
    R2_RATIO = 1
    # 3组标定
    # K = np.array([[840.37179811, 0., 980.16114332],
    #               [0., 630.19285897, 556.82334397],
    #               [0., 0., 1.]]
    #              )
    # 3.5cm标定
    K = np.array([[840.7340148, 0., 979.83540414],
                  [0., 630.42165924, 556.86959546],
                  [0., 0., 1.]]
                 )
    cap = cv.VideoCapture(r'..\..\Homework2\Dataset\GoPro_video\GOPR0110.MP4')

    # 角点检测参数
    feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)

    # KLT光流参数
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))

    # 随机颜色
    color = np.random.randint(0, 255, (100, 3))

    # 读取第一帧
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params, useHarrisDetector=False, k=0.04)

    # 绘图准备
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', 1960, 1080)
    fig = plt.figure()
    ax = Axes3D(fig)

    # 运动位置初始化
    Translation = np.zeros((3, 1))
    Rotation = np.eye(3)
    poses = deepcopy(Translation.T)

    # 光流跟踪初始化
    mask = np.zeros_like(old_frame)

    # 图片记录矩阵
    ret_old = []
    skip = 0

    while True:
        # 读取视频帧
        ret, frame = cap.read()
        # 根据图片记录矩阵判断是否结束
        if ret is False and any(ret_old[-10:]) is False:
            break
        elif ret is False:
            ret_old.append(ret)
            continue
        else:
            ret_old.append(ret)
        # 视频帧数改变
        if skip != 0:
            skip += 1
            skip %= SKIP_FRAME
            continue
        else:
            skip += 1
            skip %= SKIP_FRAME

        # 光流估计
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 计算光流
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 根据状态选择
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 绘制跟踪线
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # 相机位置估计
        # 计算基础矩阵F
        F, mask_F = cv.findFundamentalMat(good_old, good_new, cv.FM_RANSAC)
        if F is not None:
            F = F[:3]
            # 计算本质矩阵E
            E = K.T.dot(F).dot(K)
            # 本质矩阵分解
            R1, R2, t = cv.decomposeEssentialMat(E)
            # 求解旋转矩阵与平移向量
            R_t_candidate = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
            flag = 0
            for p in range(4):
                R, t = R_t_candidate[p]
                Z = extract_Rot_and_Trans(R, t, good_old, good_new, K)
                if flag < Z: flag, reg = Z, p
            R_1, t_1 = R_t_candidate[reg]

            # opencv求解旋转矩阵与平移向量
            _, R_2, t_2, _ = cv.recoverPose(E, good_old, good_new, K, mask=mask_F.copy())

            # 相机矩阵取平均
            R = (R1_RATIO * R_1 + R2_RATIO * R_2) / (R1_RATIO + R2_RATIO)
            t = (R1_RATIO * t_1 - R2_RATIO * t_2) / (R1_RATIO + R2_RATIO)

            # 位置更新
            Rotation = R.dot(Rotation)
            Rotation = Rotation / np.linalg.norm(Rotation, axis=1)[:, np.newaxis].repeat(3, axis=1)  # 方向矩阵模归一化
            Translation += Rotation.dot(t)
            poses = np.r_[poses, Translation.T]
        else:
            print("None")

        # 位置相机位置运动图绘制
        ax.plot(poses[:, 0], poses[:, 1], poses[:, 2])
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # 合并视频图片、跟踪线、相机位置运动图
        image = cv.add(frame, mask)
        image[:480, :640] = img
        cv.imshow('frame', image)

        # 跟踪线渐变
        mask = (mask / 1.1).astype(np.uint8)

        # 图片等待
        k = cv.waitKey(RATE) & 0xff
        if k == 27:
            cv.imwrite("flow.jpg", cv.add(frame, mask))
            break

        # 更新
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # 重新追踪，避免前面无追踪点导致下一次迭代出错
        if good_new.shape[0] < 20:
            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
