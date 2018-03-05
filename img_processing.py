import numpy as np
import cv2
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# loading data

X=np.zeros((927,1444))
Y_=np.zeros((927,1))
j=0

X[0:228,:]=np.loadtxt('bicycle0.txt')
Y_[0:228,0]=np.loadtxt('bicycle1.txt')

X[228:709,:]=np.loadtxt('car0.txt')
Y_[228:709,0]=np.loadtxt('car1.txt')

X[709:927,:]=np.loadtxt('motorbike0.txt')
Y_[709:927,0]=np.loadtxt('motorbike1.txt')

m,n=np.shape(X)


# structuring the neural network


hid_layer0 = 1600
hid_layer1 = 1700
num_out = 3  # number of output units
y0=np.reshape(Y_,(927))
y=np.zeros((m,num_out),dtype=np.float32)
for i in range(0,num_out):
    y[:,i]=np.array(y0==i,dtype=np.float32)


# randomly initializing the parameters

t0 = np.random.random((hid_layer0, n))
t1 = np.random.random(( hid_layer1,hid_layer0))
t2 = np.random.random((num_out,hid_layer1))
t0 = (np.array(t0, dtype=np.float32) * (0.6) - np.ones((hid_layer0, n), dtype=np.float32) * 0.3).T
t1 = (np.array(t1, dtype=np.float32) * (0.6) - np.ones((hid_layer1,hid_layer0), dtype=np.float32) * 0.3).T
t2 = (np.array(t2, dtype=np.float32) * (0.6) - np.ones((num_out,hid_layer1), dtype=np.float32) * 0.3).T

# defining important parameters and variables to use as tensors


x = tf.placeholder(tf.float32, [None, n])
y_ = tf.placeholder(tf.float32, [None, num_out])
w1 = tf.Variable(t0, tf.float32)
w2 = tf.Variable(t1, tf.float32)
w3 = tf.Variable(t2, tf.float32)


# finding the activation values


def activation(x0, w01, w02, w03):
    a1 = tf.nn.sigmoid(tf.matmul(x0, w01))
    a2 = tf.nn.sigmoid(tf.matmul(a1, w02))
    a3 = tf.matmul(a2,w03)
    return a3


# optimizing the parameters

max_iter = 1500
c = np.zeros((max_iter, 1))
Y = activation(x, w1, w2, w3)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y, labels=y_))
train = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
costf = np.zeros((max_iter, 1))
for i in range(0, max_iter):
    print('iteration', i + 1)
    sess.run(train, {x: X, y_: y})
    costf[i][0] = sess.run(cost, {x: X, y_: y})
print('The value of cost function after optimizing the parameters is :', costf[max_iter - 1, 0])
p = np.array(range(1, max_iter + 1))
plt.plot(p, costf)
plt.xlabel('Number of iterations')
plt.ylabel('Value of Cost Function')
plt.title('The variation of the cost function with the number of iterations')
plt.show()


#saving the learned parameters to a text file

w1=sess.run((w1)).T
np.savetxt('t0.txt',w1)
w2=sess.run((w2)).T
np.savetxt('t1.txt',w2)
w3=sess.run((w3)).T
np.savetxt('t2.txt',w3)


# calculating the accuracy

z0 = np.array(sess.run(Y, feed_dict={x: X, y_: y}))
z = np.array([np.max(z0, axis=1)]).T
pred = np.zeros((np.shape(z0)[0], 1), dtype=np.int32)
for i in range(0, np.shape(z0)[0]):
    for j in range(0, np.shape(z0)[1]):
        if (z[i][0] == z0[i][j]):
            pred[i, 0] = j + 1
pred = tf.Variable(pred, tf.int32)
init = tf.global_variables_initializer()
sess.run(init)
y_tc = tf.placeholder(tf.int32, [None, 1])
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y_tc), tf.float32))
print('The accuracy on the test set is:', sess.run(accuracy, feed_dict={x: X, y_: y, y_tc: Y_}))










