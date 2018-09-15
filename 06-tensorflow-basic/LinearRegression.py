# Import libraries (Numpy, Tensorflow, matplotlib)
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf

import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Create 1000 points following a function y=0.1 * x + 0.4 (i.e. y = W * x + b) with some normal random distribution
num_points = 1000
vectors_set = []
for i in range(num_points):
    W = 0.1  # W
    b = 0.4  # b
    x1 = np.random.normal(0.0, 1.0)
    nd = np.random.normal(0.0, 0.05)
    y1 = W * x1 + b
    # Add some impurity with the some normal distribution -i.e. nd
    y1 = y1 + nd
    # Append them and create a combined vector set
    vectors_set.append([x1, y1])

# Seperate the data point across axixes
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
x_train = np.array(x_data, dtype=np.float32).reshape(num_points, 1)
y_train = np.array(y_data, dtype=np.float32).reshape(num_points, 1)

# Plot and show the data points on a 2D space
plot.plot(x_data, y_data, 'ro', label='Original data')
plot.legend()
plot.show()

# tf.name_scope organize things on the tensorboard graph view
with tf.name_scope("LinearRegression") as scope:
    X = tf.placeholder(x_train.dtype, [None, 1], name='X')  # takes any number of rows but n_dim columns
    Y = tf.placeholder(y_train.dtype, [None, 1], name='Y')  # #takes any number of rows but only 1 continuous column
    W = tf.Variable(tf.zeros([1]), name='W')
    b = tf.Variable(tf.zeros([1]), name='b')
    y = W * X + b

# Define a loss function that take into account the distance between the prediction and our dataset
with tf.name_scope("LossFunction") as scope:
    loss = tf.reduce_mean(tf.square(y - Y))

optimizer = tf.train.GradientDescentOptimizer(0.6)
train = optimizer.minimize(loss)

# Annotate loss, weights and bias (Needed for tensorboard)
loss_summary = tf.summary.scalar("loss", loss)
w_ = tf.summary.histogram("W", W)
b_ = tf.summary.histogram("b", b)

# Merge all the summaries
merged_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Writer for tensorboard (Directory)
writer_tensorboard = tf.summary.FileWriter('logs/', tf.get_default_graph())

for i in range(6):
    _, summary = sess.run([train, merged_op], feed_dict={X: x_train, Y: y_train})
    writer_tensorboard.add_summary(summary, i)
    print(
        i,
        sess.run(W, feed_dict={X: x_train, Y: y_train}),
        sess.run(b, feed_dict={X: x_train, Y: y_train}),
        sess.run(loss, feed_dict={X: x_train, Y: y_train})
    )
    plot.plot(x_data, y_data, 'ro', label='Original data')
    plot.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plot.xlabel('X')
    plot.xlim(-2, 2)
    plot.ylim(0.1, 0.6)
    plot.ylabel('Y')
    plot.legend()
    plot.show()
