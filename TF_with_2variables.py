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

x1 = tf.Variable(np.array(columns['size']).astype(np.float32))
x2 = tf.Variable(np.array(columns['room']).astype(np.float32))
y = tf.Variable(np.array(columns['price']).astype(np.float32))

train_X1 = np.asarray([i[1] for i in data.loc[:,['size']].to_records()],dtype="float")
train_X2 = np.asarray([i[1] for i in data.loc[:,['room']].to_records()],dtype="float")
train_X = np.asarray([i[1] for i in data.loc[:,'size':'room'].to_records()],dtype="float")
train_Y = np.asarray([i[1] for i in data.loc[:,['price']].to_records()],dtype="float")

n_samples = train_X.shape[0]

X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
Y = tf.placeholder("float")

W1 = tf.Variable(rng.randn(), name="weight1")
W2 = tf.Variable(rng.randn(), name="weight2")
b = tf.Variable(rng.randn(), name="bias")

#Construct Linear Model
sum_list = [tf.multiply(X1,W1),tf.multiply(X2,W2)] #pred = W1*X1+W2*X2+W3*X3+b
pred_X = tf.add_n(sum_list)
pred = tf.add(pred_X,b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
 
    # Fit all training data
    for epoch in range(training_epochs):
        for (x1, x2, y) in zip(train_X1, train_X2, train_Y):
            sess.run(optimizer, feed_dict={X1: x1, X2:x2, Y: y})
 
        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X1:train_X1 , X2:train_X2, Y: train_Y})
            print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b))
 
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X1:train_X1 , X2:train_X2, Y: train_Y})
    print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b), '\n')
    
    print ("W1=",sess.run(W1))
    print ("W2=",sess.run(W2))
    print ("b=",sess.run(b))

    #Graphic display
    #plt.plot(train_X1, train_Y, 'ro', label='Original X1', color='green')
    #plt.plot(train_X2, train_Y, 'ro', label='Original X2', color='blue')
    plt.plot(train_X, train_Y, 'ro', label='Original X2', color='blue')
    plt.plot(train_X, sess.run(W1) * train_X1 + sess.run(W2) * train_X2 + sess.run(b), 
             label='Fitted line',color='orange' )
    plt.legend()
    plt.show()



