import cv2
import numpy as np
import matplotlib.pyplot as plt
# 2) 以灰度图的形式读入图片

psd_img_1 = cv2.imread('01.png', cv2.IMREAD_GRAYSCALE)
psd_img_2 = cv2.imread('02.png', cv2.IMREAD_GRAYSCALE)

# 3) SIFT特征计算
sift = cv2.SIFT_create()

psd_kp1, psd_des1 = sift.detectAndCompute(psd_img_1, None)
psd_kp2, psd_des2 = sift.detectAndCompute(psd_img_2, None)

# 4) Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(psd_des1, psd_des2, k=2)
goodMatch = []
for m, n in matches:
    # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)
# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
print(len(goodMatch))

img_out = cv2.drawMatchesKnn(psd_img_1, psd_kp1, psd_img_2, psd_kp2, goodMatch, None, flags=2)

cv2.imshow('image', img_out)  # 展示图片
plt.figure()
plt.imshow(img_out)
plt.savefig('imgOut2.png', dpi=300)
