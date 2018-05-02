import tensorflow as tf

class MyNN():
	def __init__(self, p_shape, p_scope="Train"):
		with tf.variable_scope(p_scope, reuse=tf.AUTO_REUSE):
			self.w = tf.get_variable(name="weights_1", initializer=tf.random_normal(shape=p_shape))
			self.b = tf.ones(name="bias_1", shape=p_shape[0])
			self.cost = None

	def train(self, p_x, p_y):
			self.x = p_x
			self.y = p_y
			mwx = tf.matmul(self.w, self.x)
			mwxab = tf.add(mwx, self.b)
			v_sum = tf.reduce_sum(mwxab)
			self.out = tf.sigmoid(v_sum)
			self.cost = tf.pow(tf.subtract(self.y, self.out), 2.0)
			return self.cost

nn = MyNN([2,1])
#optm = tf.train.AdamOptimizer().minimize(nn.cost)

tfinit = tf.global_variables_initializer()

with tf.Session() as sess:
	x1 = [[2.,3.]]
	y1 = [1.]
	sess.run(tfinit)
	cost = nn.train(x1, y1)
	print sess.run(cost)
	#print cost
	#print(sess.run(optm))