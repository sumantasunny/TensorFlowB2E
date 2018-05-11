"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
"""
import tensorflow as tf

x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
z1 = tf.add(x1, y1) #or x1+y1
x3 = tf.Variable(dtype=tf.float32, initial_value=1)
z3 = tf.assign(x3, z1)

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
	# when only one operation to run
	z1_value = sess.run(z1, feed_dict={x1: 1.0, y1: 2.0})
	print(sess.run(z3, feed_dict={x1: 1.0, y1: 2.0}))
	# when run multiple operations
	z1_value, z2_value = sess.run([z1, z2], feed_dict={x1: 1., y1: 2.,x2: [[2.], [2.]], y2: [[3., 3.]]}) #Run together
	print(z1_value)
	print(z2_value)