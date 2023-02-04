import numpy as np
from tensorflow import keras
import cv2

def draw(event,x,y,flags,param):
    if (flags==1):
        if [x,y] in drawings or x>27 or y>27:
            pass
        else:
            drawings.append([x,y])


new_model = keras.models.load_model("my_model.h5")

cap=cv2.VideoCapture(0)

cv2.namedWindow("img",cv2.WINDOW_NORMAL)



drawings=[]

while True:
    img=np.zeros((28,28),np.uint8)

    cv2.setMouseCallback("img",draw)

    for x,y in drawings:
        img[y,x]=255

    cv2.resizeWindow("img",560,560)

    cv2.imshow("img",img)
    




    key=cv2.waitKey(1)

    if key==ord('q'):
        break
    elif key==ord('e'):
        drawings.clear()
    elif key==ord('p'):
        img=img.reshape(1,784)

        img=img.astype("float32")

        img /=255
        print(np.argmax(new_model.predict(img)))