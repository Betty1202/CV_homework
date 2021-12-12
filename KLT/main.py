import numpy as np
import cv2 as cv

KEEP_MASK = 20
RATE = 1

cap = cv.VideoCapture(r'cvpr2021_相机运动估计数据\GoPro视频数据\GOPR0110.MP4')

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
good_ini = p0.copy()

mask = np.zeros_like(old_frame)
ret_old = []
# 光流跟踪
while True:
    ret, frame = cap.read()
    if ret is False and any(ret_old[-10:]) is False:
        break
    elif ret is False:
        ret_old.append(ret)
        continue
    else:
        ret_old.append(ret)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 计算光流
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 根据状态选择
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # 删除静止点
    k = 0
    for i, (new0, old0) in enumerate(zip(good_new, good_old)):
        a0, b0 = new0.ravel()
        c0, d0 = old0.ravel()
        dist = abs(a0 - c0) + abs(b0 - d0)
        if dist > -1:  # 不删除静止点
            good_new[k] = good_new[i]
            good_old[k] = good_old[i]
            good_ini[k] = good_ini[i]
            k = k + 1

    # 提取动态点
    good_ini = good_ini[:k]
    good_new = good_new[:k]
    good_old = good_old[:k]

    # 绘制跟踪线
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    cv.imshow('frame', cv.add(frame, mask))
    mask = (mask / 1.1).astype(np.uint8)
    k = cv.waitKey(RATE) & 0xff
    if k == 27:
        cv.imwrite("flow.jpg", cv.add(frame, mask))
        break

    # 更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 重新追踪，避免前面无追踪点导致下一次迭代出错
    if good_ini.shape[0] < 10:
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        good_ini = p0.copy()

cv.destroyAllWindows()
cap.release()
