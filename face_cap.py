import cv2
import time
import os
PATH_PREFFIX = "faces/"
PATH_SUFFIX = "_img"
PATH_LABLE = 0
PATH = PATH_PREFFIX+str(PATH_LABLE)+PATH_SUFFIX    

classifer = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    if os.path.exists(PATH_PREFFIX+str(PATH_LABLE)+PATH_SUFFIX):
        PATH_LABLE+=1
    else:
        PATH = PATH_PREFFIX+str(PATH_LABLE)+PATH_SUFFIX    
        os.makedirs(PATH)
        print("path:=======",PATH)
        break
# if not os.path.exists(SAVE_PATH):
#     os.mkdir(SAVE_PATH)
cap = cv2.VideoCapture(0)
tag = 0
while(cap.isOpened() and cv2.waitKey(2)!=ord("q")):
    (flag,frame) = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces = classifer.detectMultiScale(frame_gray,1.3,5)
    if len(faces) == 1:
        (x,y,w,h) = faces[0]
        face = frame[y:y+h,x:x+w]
        cv2.imwrite(PATH+"/img_%02d.jpg"%tag,face)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        tag = tag+1
        print(tag)
    cv2.imshow("camera",frame)

cap.release()