import cv2
import numpy as np


def main():
    # Reading files
    home = cv2.imread('data/lab7/lab7.png')
    ghost = cv2.imread('data/lab7/candy_ghost.png')

    # findig key points with sift
    sift = cv2.SIFT.create()
    kp_ghost, des_ghost = sift.detectAndCompute(ghost, None)
    kp_home, des_home = sift.detectAndCompute(home, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    # matching key points using flan
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ghost, des_home, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # finding ghost and drawing circle around it via matrix
    src_pts = np.float32([kp_ghost[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_home[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w, _ = ghost.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, matrix)
    home = cv2.polylines(home, [np.int32(dst)], True, (255, 255, 0), 3, cv2.LINE_AA)
    home = scale(home)
    cv2.imshow('res', home)
    cv2.waitKey(0)


def find_features(img):
    orb = cv2.ORB.create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return kp, des


def scale(img):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
