import cv2
import glob2
import numpy as np
import csv

X_pix=np.zeros((38,18278))
images=glob2.glob('*.jpg')
i=0
for image in images:
    img=cv2.imread(image,0)
    img_re=cv2.resize(img,(38,38))
    X_pix[:,38*i:38*(i+1)]=img_re[:,:]
    i+=1
X_pix=np.reshape(X_pix,(481,1444))
y=np.ones((481,1))

#saving the pixel intensities and corresponding labels to a csv file

np.savetxt('car0.txt',X_pix)
np.savetxt('car1.txt',y)