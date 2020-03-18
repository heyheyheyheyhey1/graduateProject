import cv2
import tensorflow as tf
import os
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split

PATH_PREFFIX = "./faces/"
MODEL_PATH = "./model/"
imgs = []
labels = []
names = ["jt","nxw","lfl","ty"]
size = 64 #规范图片长度
out_size = None
batch_size = 100
batch_num = None
classifer = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def read_data():
    if not os.path.exists(PATH_PREFFIX):
        print("img path not found")
        exit()
    else:
        len_ = len(os.listdir(PATH_PREFFIX))
        global out_size
        out_size = len_
        for i in os.listdir(PATH_PREFFIX):
            print("found path %s"%i)
            cur_index = int(i.split("_")[0])
            y_ = np.zeros([len_])
            y_[cur_index]=1.0
            for j in os.listdir(PATH_PREFFIX+i):
                path = PATH_PREFFIX+i+"/"+j
                print("reading %s"%path)
                imgs.append(cv2.resize(cv2.imread(path,1),(size,size)))
                labels.append(y_)

def weight_var(shape):
    w = tf.random_normal(shape,stddev=0.1) #标准差0.1
    return tf.Variable(w)

def bias_var(shape):
    b = tf.random_normal(shape) #正态分布取值
    return tf.Variable(b)

def conv2d(x,w):
    #1 4位置固定数据1  2是水平方向抽取数据跨度 3是竖直方向跨度
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME") 

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def drop_out(x,keep):
    return tf.nn.dropout(x,keep) #drop out 放弃一些权重的变化，避免over fitting


def add_cnn_layer ():
    # 第一层卷积 核心大小3x3 输入通道3(RGB) 输出 通道32 
    w1 = weight_var([3,3,3,32]) 
    b1 = bias_var([32])
    conv1 = tf.nn.relu(conv2d(x_holder,w1)+b1)
    # 第一层池化
    pool_1 = max_pool(conv1)
    # 第一层 drop out 
    out1 = drop_out(pool_1,1) 

    # 第二层卷积
    w2 = weight_var([3,3,32,64])
    b2 = bias_var([64])
    conv2 = tf.nn.relu(conv2d(out1,w2)+b2)
    pool_2 = max_pool(conv2)
    out2 = drop_out(pool_2,1)

    # 第三层卷积
    w3 = weight_var([3,3,64,64])
    b3 = bias_var([64])
    conv3 = tf.nn.relu(conv2d(out2,w3)+b3)
    pool_3 = max_pool(conv3)
    out3 = drop_out(pool_3,1)

    # 全连接 
    wf = weight_var([8*8*64,512])
    bf = bias_var([512])
    out3_flat = tf.reshape(out3,[-1,8*8*64])
    flatw_plus_b = tf.nn.relu(tf.matmul(out3_flat,wf)+bf)
    fout = drop_out(flatw_plus_b,1)

    # 输出
    w_out = weight_var([512,out_size])
    b_out = bias_var([out_size])
    out = tf.add(tf.matmul(fout,w_out),b_out)
    return out

def recog_face(face):
    face = cv2.resize(face,(size,size))
    result = sess.run(pred,feed_dict={x_holder:[face/255.0]})
    return names[result[0]]

def face_with_name(frame,face,pos):
    img = cv2.putText(frame,recog_face(face),pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    return img

read_data()


print("total img count %s"%len(imgs))
print("total label count %s"%len(labels))

imgs = np.array(imgs)
labels = np.array(labels)

train_x,test_x,train_y,test_y = train_test_split(imgs,labels,test_size=0.1, random_state=random.randint(0,20))

#为卷积拉平
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

batch_num = train_x.shape[0] // batch_size

#RGB小于256 转换为 0~1的浮点
train_x = train_x.astype("float32")/255.0
test_x = test_x.astype("float32")/255.0


#定义 holder
x_holder = tf.placeholder(tf.float32,[None,size,size,3]) 
y_holder = tf.placeholder(tf.float32,[None,out_size])

out = add_cnn_layer()
print(out.shape)
pred = tf.argmax(out,1)
print(pred.shape)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,tf.train.latest_checkpoint(MODEL_PATH))
print("模型载入完毕")
cap = cv2.VideoCapture(0)
while(cap.isOpened() and cv2.waitKey(2)!=ord("q")):
    flag,frame = cap.read()
    img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = classifer.detectMultiScale(img_gray,1.1,3,minSize=(64,64))
    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        print(recog_face(face))
        frame = face_with_name(frame,face,(x,y))
    cv2.imshow("face",frame)

