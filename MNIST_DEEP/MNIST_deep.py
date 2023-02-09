import numpy as np              #calculations library for matrix
from tensorflow import keras    #machine learning library for uploading pre-trained model
import cv2                      #OpenCV library for image processing

def draw(event,x,y,flags,param):                #drawing function
    if (flags==1):                              #check if user is pressing left button or holding it
        if [x,y] in drawings or x>27 or y>27:   #if current pixels are in list or pixels are bigger than the window, pass 
            pass
        else:
            drawings.append([x,y])              #append pixels to drawings list

model=keras.models.load_model("deeplearning.h5")   #loading pre-trained model

cv2.namedWindow("img",cv2.WINDOW_NORMAL)           #creating an image window

drawings=[]                                        #empty list for pixels

while True:
    img=np.zeros((28,28),np.uint8)      #creating a blank image of 28x28 pixels

    cv2.setMouseCallback("img",draw)    #do mouse events

    for x,y in drawings:                #loop for every item in drawings list
        img[y,x]=255                    #draw every pixel value at drawings list

    cv2.resizeWindow("img",560,560)     #resizing window for better drawing

    cv2.imshow("img",img)               #shows image
    

    key=cv2.waitKey(1)      #key is for keyboard inputs

    if key==ord('q'):       #pressing 'q' will quit the program
        break
    elif key==ord('e'):     #pressing 'e' will clear the image
        drawings.clear()
    elif key==ord('p'):     #pressing 'p' will predict the number

        img=img.reshape(1,28,28,1)      #reshaping image to 2 dim and 784 len matrix data
        img=img.astype("float32")   #setting data values as floating number
        img /=255                   #dividing every item in data to make values between 0-255


        print(np.argmax(model.predict(img)))  #prediction of model
