import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros(10))
y=tf.nn.softmax(tf.matmul(x,W)+b)

y_=tf.placeholder(tf.float32,[None,10])

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.InteractiveSession()
init=tf.global_variables_initializer()
sess.run(init)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
for i in range(100):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train,feed_dict={x:batch_xs,y_:batch_ys})
    sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys})
#print '--------------------'
#print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    w1=sess.run((W[:,:]))
w1=np.reshape(w1,(28,28,10))

#plt.imshow(w1[:,:,0],cmap='gray')

plt.subplot(221)
plt.imshow(w1[:,:,0],cmap='gray')
plt.axis('off')
plt.subplot(222)
plt.imshow(w1[:,:,1],cmap='gray')
plt.axis('off')
plt.subplot(223)
plt.imshow(w1[:,:,2],cmap='gray')
plt.axis('off')
plt.subplot(224)
plt.imshow(w1[:,:,3],cmap='gray')
plt.axis('off')
plt.show()

#plt.save
