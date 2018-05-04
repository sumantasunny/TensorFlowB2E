import tensorflow as tf

input_layer_size = 100
output_layer_size = 10
hidden_layer1_size = 100

input_x = tf.placeholder(dtype=tf.float32, shape=[None, input_layer_size])
actual_y = tf.placeholder(dtype=tf.float32, shape=[None, output_layer_size])

ih1w = tf.get_variable(name="ih1w", shape=[input_layer_size, hidden_layer1_size])
ih1b = tf.get_variable(name="ih1b", shape=[hidden_layer1_size])
h1ow = tf.get_variable(name="h1ow", shape=[hidden_layer1_size, output_layer_size])
h1ob = tf.get_variable(name="h1ob", shape=[output_layer_size])

class customCell():
	def __init__(self):
		self.internal_weights1 = tf.get_variable(name="internal_weights1", shape=[1, input_layer_size])
		self.internal_bias1 = tf.ones(name="internal_bias1", shape=[1])

	def train(self, X, Y):
		score = tf.matmul(X, self.internal_weights1) + self.internal_bias1
		output = tf.sigmoid(score)
		return output


init_vars = tf.global_variables_initializer()


_cell = customCell()

output = _cell.train([], [])

with tf.Session as sess:
	sess.run(init_vars)