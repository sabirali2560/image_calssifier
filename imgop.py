import cv2
import glob2
import numpy as np

def pixel_val():
    X_pix=np.zeros((1920,128,3))  #creating a numpy array to store the pixel values of the 15 images used as the training images
    images=glob2.glob('*.jpg')
    i=0
    for image in images:
        img=cv2.imread(image,1)
        img_re=cv2.resize(img,(128,128))      #resizing the images to 128*128 resolution
        X_pix[128*i:128*(i+1),:,:]=img_re[:,:]     #storing the pixel intensity values of the resized images in X_pix
        i+=1
    img1=cv2.imread('bird_small.png',1)     #storing the pixel intensity values of the 15th training image with a different extension
    X_pix[1792:1920,:,:]=img1[:,:]
    m,n,p=np.shape(X_pix)
    return(X_pix)
