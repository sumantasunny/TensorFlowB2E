import tensorflow as tf
from random import randint
import random


tf.set_random_seed(100)
random.seed(100)

num_of_input_units = 2
num_of_output_units = 1

x = []
y = []

def create_inputs():
	global x, y, num_of_input_units
	for i in range(100):
		xi = [float(randint(1,9)) for j in range(10)]
		yi = [float(randint(0,1)) for j in range(1)]
		print xi, yi
		x.append(xi)
		y.append(yi)
	num_of_input_units = len(x[0])

v_input_x = None
v_input_y = None
v_weights = None
v_bias = None
def declare_variables():
	global v_input_x, v_input_y, v_weights, v_bias
	v_input_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_input_units])
	v_input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_of_output_units])

	v_weights = tf.get_variable(name="Weights_1", shape=[num_of_input_units, num_of_output_units], initializer=tf.random_normal_initializer())
	v_bias = tf.ones(shape=[num_of_output_units])

v_score = None
v_cost = None
v_optimizer = None
v_output = None
def train():
	global v_score, v_cost, v_optimizer, v_output
	v_score = tf.add(tf.matmul(v_input_x, v_weights), v_bias)
	v_output = tf.sigmoid(v_score)
	#v_print = tf.Print(v_output, [v_output, v_score])
	v_cost = tf.sqrt(tf.reduce_mean(tf.pow((v_output - v_input_y), 2)))
	#v_cost = tf.abs(v_output-v_input_y)
	v_optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(v_cost)

with tf.Session() as sess:
	create_inputs()
	declare_variables()
	train()
	v_init = tf.global_variables_initializer()
	sess.run(v_init)
	print(sess.run(v_weights))
	print(sess.run(v_bias))
	cost = 0
	''' for i in range(100):
		for j in range(0, len(x)):
			xj = [x[j]]
			yj = [y[j]]
			out, c, _ = sess.run([v_output, v_cost, v_optimizer], feed_dict={v_input_x: xj, v_input_y: yj})
			cost += c
			#print ">>", xj, out, y[j], c
		print(cost/len(x)) '''
	for i in range(100):
		out, c, _ = sess.run([v_output, v_cost, v_optimizer], feed_dict={v_input_x: x, v_input_y: y})
		cost = c
		print cost
		#reset
		#print ">>", out, y