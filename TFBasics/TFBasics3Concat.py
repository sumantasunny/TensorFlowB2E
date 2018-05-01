import tensorflow as tf

x1 = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
x2 = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

x3 = tf.concat([x1, x2], 1)

sess = tf.Session()

output = sess.run(x3)

print output

sess.close()