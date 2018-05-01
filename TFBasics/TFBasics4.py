import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

x3 = tf.reshape(x1, [1,-1], name="re_x3")

shape3 = tf.shape(x3)

tf.global_variables_initializer()

sess = tf.Session()

output = sess.run(x3)

print output

output2 = sess.run(shape3)

print output2

sess.close()