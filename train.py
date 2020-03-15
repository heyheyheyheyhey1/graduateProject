import cv2
import tensorflow
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

PATH_PREFFIX = "faces/"
imgs = []
labels = []
def read_data():
    if not os.path.exists(PATH_PREFFIX):
        print("img path not found")
        exit()
    else:
        len_ = len(os.listdir(PATH_PREFFIX))
        for i in os.listdir(PATH_PREFFIX):
            print("found path %s"%i)
            cur_index = int(i.split("_")[0])
            y_ = np.zeros([len_])
            y_[cur_index]=1.0
            for j in os.listdir(PATH_PREFFIX+i):
                path = PATH_PREFFIX+i+"/"+j
                print("reading %s"%path)
                imgs.append(cv2.imread(path,1))
                labels.append(y_)

read_data()
print("total img count %s"%len(imgs))
print("total label count %s"%len(labels))
 
train_x,test_x,train_y,test_y = train_test_split(imgs,labels,test_size=0.2, random_state=random.randint(0,20))
print("leng test img : %d"%len(test_y))
print("leng train img : %d"%len(train_y))
