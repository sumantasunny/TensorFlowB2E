import tensorflow as tf

tf.set_random_seed(100)

num_of_input_units = 2
num_of_output_units = 1

x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
y = [[1], [1]]
num_of_input_units = len(x[0])

v_input_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_input_units])
v_input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_of_output_units])

v_weights = tf.get_variable(name="Weights_1", shape=[num_of_input_units, num_of_output_units], initializer=tf.random_normal_initializer())
v_bias = tf.ones(shape=[num_of_output_units])

v_score = tf.add(tf.matmul(v_input_x, v_weights), v_bias)

v_output = tf.sigmoid(v_score)

v_cost = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(v_score, v_input_y), 2)))

v_optimzer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(v_cost)

v_init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(v_init)
	print(sess.run(v_weights))
	print(sess.run(v_bias))
	#output, cost = sess.run([v_output, v_cost], feed_dict={v_input_x:x, v_input_y: y})
	cost = 0
	for i in range(1000):
		for j in range(0, len(x)):
			xj = [x[j]]
			yj = [y[j]]
			out, c, _ = sess.run([v_output, v_cost, v_optimzer], feed_dict={v_input_x: xj, v_input_y: yj})
			cost += c
			print ">>", out
		print(cost/len(x))