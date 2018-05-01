import tensorflow as tf

tf.set_random_seed(100)
with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
	x1 = tf.Variable([1,2,3,4], dtype=tf.float64)
	x2 = tf.get_variable(name="var_x1", shape=[1,4], dtype=tf.float64)
	x5 = tf.get_variable("var_x1", dtype=tf.float64)
	x3 = tf.Variable(initial_value=tf.random_normal(mean=10.0, stddev=5.0, shape=[1,4], dtype=tf.float64), name="var_x1", dtype=tf.float64)
	x4 = tf.matmul(x2, x3, transpose_a=True)


with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
	x6 = tf.get_variable("var_x1", dtype=tf.float64)

with tf.variable_scope("", reuse=tf.AUTO_REUSE):
	x7 = tf.get_variable("foo/var_x1", dtype=tf.float64)

x10 = tf.Variable([1,2,3,4], dtype=tf.float64, name="var_x10")

x8 = [var for var in tf.global_variables() if var.op.name=="var_x10"][0]

x9 = bar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="var_x10")[0]

print x10 is x8
print x10 is x9

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print(sess.run(x1))
print(sess.run(x2))
print(sess.run(x5))
print(sess.run(x3))
print(sess.run(x4))
sess.close()