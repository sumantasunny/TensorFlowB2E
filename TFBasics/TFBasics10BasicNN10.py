import tensorflow as tf
from random import randint
import random

tf.set_random_seed(100)
random.seed(100)

def create_inputs():
	global x, y, num_of_input_units
	for i in range(100):
		xi = [float(randint(1,9)) for j in range(100)]
		yi = [float(randint(0,1)) for j in range(10)]
		print xi, yi
		x.append(xi)
		y.append(yi)
	num_of_input_units = len(x[0])

#declare hyper params here
lst_layers_size = [100, 50, 50, 10] #including input and output
number_of_layers = len(lst_layers_size) - 1 #ommiting the input layer

dict_weights = {}
dict_biases = {}

lst_weights = []
lst_biases = []

input_x = None
input_y = None

def declare_all_weights_and_biases():
	for i in range(number_of_layers):
		str_layer_weight_key = str("weights_from_"+str(i)+"_to_"+str(i+1))
		str_layer_bias_key = str("biases_from_"+str(i)+"_to_"+str(i+1))
		
		layer_weights_mat = tf.get_variable(name=str_layer_weight_key, shape=[lst_layers_size[i], lst_layers_size[i+1]], dtype=tf.float32, initializer=tf.random_normal_initializer())
		layer_biases_mat = tf.get_variable(name=str_layer_bias_key, shape=[lst_layers_size[i+1]], dtype=tf.float32, initializer=1.0)

		dict_weights[str_layer_weight_key] = layer_weights_mat
		dict_biases[str_layer_bias_key] = layer_biases_mat

		lst_weights.append(layer_weights_mat)
		lst_biases.append(layer_biases_mat)

declare_all_weights_and_biases()

X = input_x
Y = input_y

i_layer_out = X

for i in range(number_of_layers):
	i_layer_socres = tf.matmul(i_layer_out, lst_weights[i]) + lst_biases[i]
	i_layer_out = tf.sigmoid(i_layer_socres)

output_print = tf.Print(i_layer_out, [i_layer_out])
cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow((output_print - Y),2), 1, keepdims=True)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init_var = tf.global_variables_initializer()

with tf.Session as sess:
	sess.run(init_var)
	for i in range(10):
		_, c = sess.run([optimizer, cost], feed_dict={X:input_x, Y:input_y})
		print c