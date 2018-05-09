import tensorflow as tf
from random import randint
import random

tf.set_random_seed(100)
random.seed(100)

input_x = []
input_y = []

#declare hyper params here
epoch = 100
learning_rate = 0.0001
num_input_units = 100
num_output_units = 10
lst_layers_size = [num_input_units, 90, 80, 70, 60, 50, 40, 30, 20, num_output_units] #including input and output
number_of_layers = len(lst_layers_size) - 1 #omitting the input layer

dict_weights = {}
dict_biases = {}

lst_weights = []
lst_biases = []

def create_inputs():
	global input_x, input_y
	for i in range(100):
		xi = [float(randint(1,9)) for j in range(lst_layers_size[0])]
		yi = [float(randint(0,1)) for j in range(lst_layers_size[len(lst_layers_size)-1])]
		print len(xi), len(yi)
		input_x.append(xi)
		input_y.append(yi)

create_inputs()

def declare_all_weights_and_biases():
	for i in range(number_of_layers):
		str_layer_weight_key = str("weights_from_"+str(i)+"_to_"+str(i+1))
		str_layer_bias_key = str("biases_from_"+str(i)+"_to_"+str(i+1))
		
		layer_weights_mat = tf.get_variable(name=str_layer_weight_key, shape=[lst_layers_size[i], lst_layers_size[i+1]], dtype=tf.float32, initializer=tf.random_normal_initializer())
		layer_biases_mat = tf.ones(name=str_layer_bias_key, shape=[lst_layers_size[i+1]], dtype=tf.float32)

		dict_weights[str_layer_weight_key] = layer_weights_mat
		dict_biases[str_layer_bias_key] = layer_biases_mat

		lst_weights.append(layer_weights_mat)
		lst_biases.append(layer_biases_mat)

declare_all_weights_and_biases()

#X = input_x
#Y = input_y

v_input_x = tf.placeholder(dtype=tf.float32, shape=[None, lst_layers_size[0]])
v_input_y = tf.placeholder(dtype=tf.float32, shape=[None, lst_layers_size[len(lst_layers_size)-1]])

i_layer_out = v_input_x

for i in range(number_of_layers):
	i_layer_socres = tf.matmul(i_layer_out, lst_weights[i]) + lst_biases[i]
	i_layer_out = tf.sigmoid(i_layer_socres)

output_print = tf.Print(i_layer_out, [i_layer_out, v_input_y])
cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow((output_print - v_input_y),2), 1, keepdims=True)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init_var = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_var)
	for i in range(epoch):
		_, c = sess.run([optimizer, cost], feed_dict={v_input_x:input_x, v_input_y:input_y})
		print c