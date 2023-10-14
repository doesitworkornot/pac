import cv2
import time
import numpy as np


def main():
    cap = cv2.VideoCapture('data/lab6/childgame.mp4')
    time_start = time.time()
    state_len = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    while cap.isOpened():
        if frame2 is None:
            break
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        time_past = time.time()
        time_diff = time_past - time_start
        if time_diff % (2 * state_len) > state_len:
            color = (0, 0, 255)
            text = 'Moving is prohibited'
        else:
            color = (0, 255, 0)
            text = 'Alles gut'
        cv2.drawContours(frame1, сontours, -1, color, 2)
        cv2.putText(frame1, text, (1, 30), font, 1, color, 2)
        cv2.imshow('full grown man', frame1)

        if cv2.waitKey(int(1000 / fps)) == ord('q'):
            break
        frame1 = frame2
        ret, frame2 = cap.read()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
