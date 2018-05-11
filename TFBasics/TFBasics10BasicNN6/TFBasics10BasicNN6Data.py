from random import randint
import random

input_x = []
input_y = []

def create_inputs(input_size, output_size):
	global input_x, input_y
	random.seed(100)
	for i in range(100):
		xi = [float(randint(1,9)) for j in range(input_size)]
		yi = [float(randint(0,1)) for j in range(output_size)]
		print len(xi), len(yi)
		input_x.append(xi)
		input_y.append(yi)

def create_batches(batch_size):
	global input_x, input_y
	batch_x = []
	batch_y = []
	for i in range(batch_size):
		ch = randint(0, len(input_x))
		batch_x[i] = input_x[ch]
		batch_y[i] = input_y[ch]