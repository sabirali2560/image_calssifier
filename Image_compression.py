import csv
import numpy as np
import tensorflow as tf
import random
import imgop

#loading data
X_sep0=imgop.pixel_val()
X_sep0=X_sep0/255

for q in range(0,15):
    X_sep1=X_sep0[128*q:128*(q+1),:,:]
    X=np.zeros((16384,3))
    for i in range(0,128):
        for j in range(0,128):
            X[i*j,:]=X_sep1[i,j,:]
    m,n=np.shape(X)

    K=10    #no. of clusters to be formed
    max_iter=80 #maximum iterations to be used

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
    for i in range(0,max_iter):
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
    x0=np.zeros((K,1))
    centroids_index=sess.run(centroids_index)
    q=0
    for i in range(0,m):
        w=centroids_index[i,0]
        if(i>1):
            if(x0[q,0]!=w):
                q+=1
                x0[q,0]=w
    X=sess.run(X)
    c=1.0
    e=0.0
    X_maxmin=np.zeros((K,2,n))
    for i in range(0,n):
        for j in range(0,K):
             for k in range(0,m):
                 if(centroids_index[k,0]==x0[j,0]):
                     if(X[k,i]>e):
                         e=X[k,i]
                     if(X[k,i]<c):
                         c=X[k,i]
             X_maxmin[j,:,i]=np.array([e,c])







































'''with open('centroids.csv', 'w') as file0:
    csvwrite = csv.writer(file0)
    csvwrite.writerow(['centroids', 'centroids_index'])
    for i in range(0,348):
        csvwrite.writerow([centroids[i,], centroids_index[i, 0]])'''


