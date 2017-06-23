from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorlayer as tl
import numpy as np
import tensorflow as tf
from PIL import Image as Img
import matplotlib.pyplot as plt
import random

sess = tf.InteractiveSession()


#training data
trainX = np.loadtxt(open("data/trainX.csv", "rb"), delimiter=",", skiprows=0)
trainY = np.loadtxt(open("data/trainY.csv", "rb"), delimiter=",", skiprows=0)
testX = np.loadtxt(open("data/testX.csv", "rb"), delimiter=",", skiprows=0)

def weight_tensor(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_tensor(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#placeholders and variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
weight = weight_tensor([5, 5, 1, 32])
bias = bias_tensor([32])


#First convolutional layer
W_conv1 = weight_tensor([5, 5, 1, 32])
b_conv1 = bias_tensor([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer
W_conv2 = weight_tensor([5, 5, 32, 64])
b_conv2 = bias_tensor([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely connected layer
W_fc1 = weight_tensor([7 * 7 * 64, 1024])
b_fc1 = bias_tensor([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout implementation for dcl
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Densely connected layer #2

#Densely connected layer
W_fc2 = weight_tensor([1024,1024])
b_fc2 = bias_tensor([1024])
h_fc1_flat = tf.reshape(h_fc1_drop, [-1, 1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)

#dropout implementation for dcl2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


#readout layer
W_fc3 = weight_tensor([1024, 10])
b_fc3 = bias_tensor([10])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3




cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))




train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

print('Training neural network...')
print('training iterations: 20,000...')
for i in range(20000):
    subset = np.random.randint(4000, size=100)
    batchX = trainX[subset, :]
    batchY = trainY[subset, :]
    index = 0
    for j in batchX:
        newimage = j.reshape(28,28)
        rand = np.random.binomial(1, .5, 1)

        #plt.imshow(newimage)
        #plt.show()

        if rand == 1:
            newimage = tl.prepro.elastic_transform(newimage, alpha=28 * 3, sigma=28 * 0.32)

        img = Img.fromarray(newimage)
        rand = random.gauss(0,20)
        rotated = Img.Image.rotate(img,rand)

        #plt.imshow(np.array(rotated))
        #plt.show()

        batchX[index] = np.array(rotated).flatten()
        index = index +1;


    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batchX, y_: batchY, keep_prob: 1.0})

    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: batchX, y_: batchY, keep_prob: 1.0}))
print("running test set...")
prediction=tf.argmax(y_conv,1)
pred = np.array(prediction.eval(feed_dict={x: testX, keep_prob: 1.0}))

print("writing to file...")
file = open('pred.txt', 'w')
file.write("%s\n" % "id,digit")
for i in range(0,800):
    item = str(i) + ',' + str(pred[i])
    file.write("%s\n" % item)

print("done")