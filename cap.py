import cv2
import time
import os
SAVE_PATH="./img_src_2"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
cap = cv2.VideoCapture(0)
tag = 0
while(cap.isOpened() and cv2.waitKey(2)!=ord("q")):
    (flag,frame) = cap.read()
    cv2.imwrite(SAVE_PATH+"/img_%02d.jpg"%tag,frame)
    tag = tag+1
    print(tag)
    cv2.imshow("camera",frame)

cap.release()