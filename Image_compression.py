import scipy.io as sio
import numpy as np
import tensorflow as tf
import random

K=30 #the number of clusters to be formed


#loading data

filename0='bird_small.mat'
datadict=sio.loadmat(filename0)
X_sep=np.array(datadict['A'],np.float32)
X=np.zeros((384,128))
for i in range(0,3):
   X[128*i:128*(i+1),:]=X_sep[:,:,i]
m,n=np.shape(X)


#randomly choosing the centroids

centroids=np.zeros((K,n),np.float32)
for i in range(0,K):
    w=random.randint(0,m)
    centroids[i,:]=X[w,:]
X=tf.constant(X,tf.float32)    
centroids=tf.Variable(centroids,tf.float32)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


#forming clusters

p=0
while(1):
    centroids_mat=tf.reshape(tf.tile(centroids,[m,1]),[m,K,n])
    X_mat=tf.reshape(tf.tile(X,[1,K]),[m,K,n])
    distances=tf.reduce_sum(tf.square(X_mat-centroids_mat),reduction_indices=2)
    centroids_index=tf.argmin(distances,1)
    total_sum=tf.unsorted_segment_sum(X,centroids_index,K)
    num_total=tf.unsorted_segment_sum(tf.ones_like(X),centroids_index,K)
    c0=sess.run(centroids)
    centroids=total_sum/num_total
    c1=sess.run(centroids)
    l=0
    p=p+1
    print('iteration', p)
    for i in range(0,K):
        for j in range(0,n):
            if(c1[i][j]==c0[i][j]):
                l=l+1
    if(l==K*n):
       break
Dict={'centroids':centroids,'centroids_index':centroids_index}
filename1='ex7data1.mat'
sio.savemat(filename1,Dict)




       
