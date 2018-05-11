"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
"""
import tensorflow as tf

var = tf.Variable(0)    # our first variable in the "global_variable" set
g_var_1 = tf.get_variable(name="g_var_1", shape=[1], dtype=tf.float32, initializer=tf.random_normal_initializer())


add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)

with tf.Session() as sess:
    # once define variables, you have to initialize them by doing this
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
		sess.run(update_operation)
		print(sess.run(g_var_1))
		print(sess.run(var))