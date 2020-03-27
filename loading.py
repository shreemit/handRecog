import numpy as np
import matplotlib.pyplot as plt
import os
import cv2.cv2 as cv2

DATADIR="C:/ML/Hand Gestures/train"
CATEGORIES =["1L","2L","3L","4L","5L"]

for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        plt.show()

        print(img_array)

        print("edfefew", img_array.shape)

        break

    break

IMG_SIZE = 50
new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))



