import cv2 as cv
import numpy as np


def linear_triangulation(K: np.array, C1: np.array, R1: np.array, C2: np.array, R2: np.array, pt: np.array,
                         pt_: np.array) -> list:
    P1 = K @ np.hstack((R1, -R1 @ C1))
    P2 = K @ np.hstack((R2, -R2 @ C2))
    X = []
    for i in range(len(pt)):
        x1 = pt[i]
        x2 = pt_[i]
        A1 = x1[0] * P1[2, :] - P1[0, :]
        A2 = x1[1] * P1[2, :] - P1[1, :]
        A3 = x2[0] * P2[2, :] - P2[0, :]
        A4 = x2[1] * P2[2, :] - P2[1, :]
        A = [A1, A2, A3, A4]
        U, S, V = np.linalg.svd(A)
        V = V[3]
        V = V / V[-1]
        X.append(V)
    return X


def extract_Rot_and_Trans(R1: np.array, t: np.array, pt: np.array, pt_: np.array, K: np.array):
    C = [[0], [0], [0]]
    R = np.eye(3, 3)
    X1 = linear_triangulation(K, C, R, t, R1, pt, pt_)
    X1 = np.array(X1)
    count = 0
    for i in range(X1.shape[0]):
        x = X1[i, :].reshape(-1, 1)
        if R1[2] @ np.subtract(x[0:3], t) > 0 and x[2] > 0: count += 1
    return count
