from __future__ import print_function
import csv
from collections import defaultdict
import tensorflow as tf
import numpy
import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader
import pandas

rng = np.random

DATA_PATH = '/Users/natali/Desktop/tensorflow/sahibinden/Blocket_3variable.csv'
COLUMNS = ['size', 'room', 'price']

learning_rate = 0.01
training_epochs = 1000
display_step = 10

columns = defaultdict(list)

data = pandas.read_csv(tf.gfile.Open(DATA_PATH), names=COLUMNS, delimiter=',', header=0)
#data = pandas.read_csv("blocket_scrape.csv", names=COLUMNS, delimiter=',', header=0)  

x1 = np.array(columns['size'], dtype=np.float)
x2 = np.array(columns['room'], dtype=np.float)
x = tf.Variable(np.row_stack((x1, x2)).astype(np.float32))
y = np.array(columns['price'], dtype=np.float)

train_X = np.asarray([i[1] for i in data.loc[:,'size':'room'].to_records()],dtype="float")
train_Y = np.asarray([i[1] for i in data.loc[:,['price']].to_records()],dtype="float")

print (train_X.shape)
print (train_Y.shape)


n_samples = train_X.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(rng.randn(), [1,2])
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
 
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
 
        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"W=", sess.run(W), "b=", sess.run(b))
 
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    
    print ("W=",sess.run(W))
    print ("b=",sess.run(b))
 
    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
