import cv2
import glob2
import numpy as np
import csv

X_pix=np.zeros((38,8664))
images=glob2.glob('*.jpg')
i=0
for image in images:
    img=cv2.imread(image,0)
    img_re=cv2.resize(img,(38,38))
    X_pix[:,38*i:38*(i+1)]=img_re[:,:]
    i+=1
X_pix=np.reshape(X_pix,(228,1444))
y=np.zeros((228,1))

#saving the pixel intensities and corresponding labels to a text file

np.savetxt('bicycle0.txt',X_pix)
np.savetxt('bicycle1.txt',y)
