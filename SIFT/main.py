import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

MATCH_RATIO = 0.5

logger = logging.getLogger("SIFT")
image1 = cv2.imread("01.png")
image2 = cv2.imread("02.png")

logger.info("Calculate Keypoints and features with SIFT")
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

kp_image1 = cv2.drawKeypoints(image1, kp1, None)
kp_image2 = cv2.drawKeypoints(image2, kp2, None)

plt.figure()
plt.imshow(kp_image1)
plt.savefig('kp_image1.png', dpi=300)

plt.figure()
plt.imshow(kp_image2)
plt.savefig('kp_image2.png', dpi=300)

logger.info(f"Key points number: {len(kp1)}, {len(kp2)}")
logger.info(f"feature shape: {des1.shape}, {des2.shape}")

logger.info("Match keypoints with KNN")
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k=2)
good_matches = []
for m1, m2 in raw_matches:
    if m1.distance < MATCH_RATIO * m2.distance:
        good_matches.append([m1])

matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good_matches, None, flags=2)

plt.figure()
plt.imshow(matches)
plt.savefig('matches.png', dpi=300)

logger.info(f"Number of match pairs: {len(good_matches)}")

logger.info("RANSAC")
if len(good_matches) > 4:
    ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4

    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
    logger.info(f"Homography: {H}")
    imgOut = cv2.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    plt.figure()
    plt.imshow(imgOut)
    plt.savefig('imgOut.png', dpi=300)
