#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import datetime

# create some noisy data
x_data = np.random.rand(1000, 2).astype(np.float32)
correct_W = [[1, 2, 3], [4, 5, 6]]
correct_b = [11, 12, 13]
correct_W, correct_b = map(lambda l: np.array(l, dtype=np.float32), (correct_W, correct_b))
noise_level = 0.01
y_data = np.dot(x_data, correct_W) + correct_b + np.random.normal(size=(1000, 3))


# define the symbolic variables
W = tf.Variable(tf.random_uniform(correct_W.shape, -1.0, 1.0))
b = tf.Variable(tf.zeros(correct_b.shape))

# define the data placeholders
batch_size = 10
x_ph = tf.placeholder(tf.float32, shape=(batch_size, 2))
y_ph = tf.placeholder(tf.float32, shape=(batch_size, 3))

# define the model (using placeholders)
y_hat = ...

# define the (stochastic!) loss
loss = ...
tf.scalar_summary('log loss', tf.log(1.0 + loss))  # attention: this is the stochastic loss, i.e. it will be noisy

# define the optimizer
step_size = 0.1
optimizer = tf.train.GradientDescentOptimizer(step_size)
train_op = optimizer.minimize(loss)

# initialize the tensorflow session
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter("train/ex2_{}".format(datetime.datetime.now().strftime("%s")), sess.graph)

# call the train_op many times, each time it will update the variables W and b according to their gradients
    for step in range(201):
        # determine the minibatch
        start_index = (batch_size * step) % x_data.shape[0]
        stop_index = start_index + batch_size

        # get the minibatch data
        x_minibatch = x_data[start_index:stop_index]
        y_minibatch = y_data[start_index:stop_index]

        feed_dict = {
                x_ph: x_minibatch,
                y_ph: y_minibatch
                }

        _, loss_value, summary_str = sess.run([train_op, loss, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        print("iteration:", step, "loss:", loss_value)

    print("learned W:\n{}".format(sess.run(W)))
    print("learned b:\n{}".format(sess.run(b)))
