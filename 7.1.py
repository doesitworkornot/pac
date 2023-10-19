import cv2
import numpy as np
import os
import random


def rotate(img, mask, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    result2 = cv2.warpAffine(mask, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result, result2


def mirror(img, mask):
    return cv2.flip(img, 1), cv2.flip(mask, 1)


def crop(img, mask):
    h, w, _ = img.shape
    delta_h = int(h/10)
    delta_w = int(w / 10)
    return img[delta_h:h-delta_h, delta_w:w-delta_w], mask[delta_h:h-delta_h, delta_w:w-delta_w]


def blur(img, mask):
    ksize = (30, 30)
    return cv2.blur(img, ksize, cv2.BORDER_DEFAULT), cv2.blur(mask, ksize, cv2.BORDER_DEFAULT)


def main():
    path = 'data/lab6/images/'
    path2 = 'data/lab6/labels/'
    files = os.listdir(path)
    n = 4
    files = random.sample(files, n)
    for elem in files:
        fp = path+elem
        fp2 = path2+elem
        img = cv2.imread(fp)
        mask = cv2.imread(fp2)
        angle = np.random.randint(1, 360)
        funcs = [rotate(img, mask, angle), mirror(img, mask), crop(img, mask), blur(img, mask)]
        i = 0
        for fun in funcs:

            res, res_m = fun
            image_gray = cv2.cvtColor(res_m, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_out = cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
            cv2.imshow(str(i), image_out)
            i += 1
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == "__main__":
    main()
