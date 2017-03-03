import tensorflow as tf
import numpy as np

X_pl = tf.placeholder(tf.float32, [None,1])
Y_pl = tf.placeholder(tf.float32, [None,1])

w1 = tf.get_variable('w1', [1,10], tf.float32)
w2 = tf.get_variable('w2', [10,1], tf.float32)
b1 = tf.get_variable('b1', [10], tf.float32)
b2 = tf.get_variable('b2', [1], tf.float32)
out_FC1 = tf.sigmoid(tf.nn.bias_add(tf.matmul(X_pl, w1), b1))
out = tf.nn.bias_add(tf.matmul(out_FC1, w2), b2)

cost = tf.reduce_mean(tf.square(Y_pl-out))

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



batch_size = 1000
n_iter = 400
for it in range(n_iter):
    X = np.random.uniform(0.1,0.9,[batch_size,1])
    Y = X**3 - 2*X**2 + 10*X - 4
    print('cost', sess.run(cost, {X_pl:X, Y_pl:Y}))
    sess.run(train, {X_pl:X, Y_pl:Y})
    




