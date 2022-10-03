# 導入相關函式庫與模組
import os
import random
import numpy as np
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import  Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D 
from keras.models import load_model
from keras.utils import np_utils
from matplotlib import pyplot as plt

def data_x_y_preprocess(datapath):  # 資料前置處理函式
    img_row,img_col = 28,28  # 圖片高寬
    datapath = datapath  # 資料路徑
    data_x = np.zeros((28,28)).reshape(1,28,28) # 儲存圖片
    pictureCount = 0  # 記錄圖片張數
    data_y = []  # 記錄label
    num_class=10  # 圖片種類共10種
    for root, dir, file in os.walk(datapath):  # 讀取資料夾所有圖片
        for f in file: 
            label=int(root.split("\\")[2])  # 取得圖片label
            data_y.append(label)  # 記錄label
            fullpath=os.path.join(root,f)  # 取得圖片路徑
            img=Image.open(fullpath)  # 開啟圖片
            img=(np.array(img)/255).reshape(1,28,28)  # 資料正規化&reshape
            data_x=np.vstack((data_x,img))
            pictureCount += 1  # 圖片計數
    data_x=np.delete(data_x,[0],0)  # 刪除原宣告的np.zeros
    data_x=data_x.reshape(pictureCount,img_row,img_col,1)
    data_y=np_utils.to_categorical(data_y,num_class)  # label轉為one-hot encoding
    return data_x,data_y

model = Sequential()  # 建立線性模型
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))  # 卷積層
model.add(MaxPooling2D(pool_size=(2,2)))  # 池化層
model.add(Conv2D(64,(3,3),activation='relu'))  # 卷積層
model.add(MaxPooling2D(pool_size=(2,2)))  # 池化層
model.add(Dropout(0.1))  # Dropout層-隨機斷開以防過擬合
model.add(Flatten())  # Flatten層-多維輸入一維化
model.add(Dropout(0.1))  # Dropout層-隨機斷開以防過擬合
model.add(Dense(128,activation='relu'))  # 全連接層-128個output
model.add(Dropout(0.25))  # Dropout層-隨機斷開以防過擬合
model.add(Dense(units=10,activation='softmax'))  # 將結果分10類


(data_train_X,data_train_Y) = data_x_y_preprocess("train_image") # 載入訓練資料
(data_test_X,data_test_Y) = data_x_y_preprocess("test_image") # 載入測試資料

# 選擇損失函數,優化方法及成效衡量方式
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
# 進行訓練
train_history = model.fit(data_train_X,data_train_Y,batch_size=32,epochs=16,verbose=1,validation_split=0.1)

# 顯示損失函數及訓練成果百分比
score = model.evaluate(data_test_X,data_test_Y,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
