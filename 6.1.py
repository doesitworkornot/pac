import os
import cv2


def first():
    path = 'data/lab6/images/'
    path2 = 'data/lab6/labels/'
    files = os.listdir(path)
    for elem in files:
        fp = path+elem
        fp2 = path2+elem
        img = cv2.imread(fp)
        mask = cv2.imread(fp2)
        image_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_out = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        cv2.imshow(path, image_out)
        cv2.waitKey(0)


def sec():
    cap = cv2.VideoCapture('data/lab6/rabbit.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('HAAHHAHA HES SO FUNNY', gray)
        cv2.waitKey(int(1000/fps))
    cap.release()
    cv2.destroyAllWindows()


def main():
    first()
    sec()


if __name__ == "__main__":
    main()
