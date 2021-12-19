import glob

import cv2
import numpy as np


# from tqdm import tqdm


def find_chess_point(size):
    print(f"开始寻找{size}cm棋盘点")
    # 找棋盘格角点
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 阈值
    # 棋盘格模板规格
    w = 8
    h = 11
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp = objp * size * 10  # 18.1 mm

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    # 加载pic文件夹下所有的jpg图像
    images = glob.glob(rf"..\..\Homework2\Dataset\GoPro_figure_Resized\{size}cm\*.jpg")  # 拍摄的十几张棋盘图片所在目录

    i = 0
    for fname in images:
        img = cv2.imread(fname)
        # 获取画面中心点
        # 获取图像的长宽
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # 如果找到足够点对，将其存储起来
        if ret == True:
            i = i + 1
            # 在原角点的基础上寻找亚像素角点
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            # 追加进入世界三维点和平面二维点中
            objpoints.append(objp)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 640, 480)
            cv2.imshow('findCorners', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    return objpoints, imgpoints, gray.shape


if __name__ == '__main__':
    objpoint1, imgpoint1, shape = find_chess_point(3)
    objpoint2, imgpoint2, _ = find_chess_point(3.5)
    objpoint3, imgpoint3, _ = find_chess_point(4.5)
    objpoints = objpoint1 + objpoint2 + objpoint3
    imgpoints = imgpoint1 + imgpoint2 + imgpoint3

    # %% 标定
    print('正在计算')
    # 标定
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, shape[::-1], None, None)

    print("ret:", ret)
    
    print("mtx:\n", mtx)  # 内参数矩阵
    print("dist畸变值:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs旋转（向量）外参:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs平移（向量）外参:\n", tvecs)  # 平移向量  # 外参数
