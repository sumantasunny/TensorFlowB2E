import tensorflow as tf
import matplotlib as plt


x1 = tf.constant([1,2,3,4])
x2 = tf.constant([4,3,2,1])

x3 = tf.multiply(x1, x2)
#x4 = [x3]
tfprt = tf.Print(x3, [x3, tf.shape(x3), "OK !! : "])
#x4 = tf.multiply(tfprt, x3)
#print x4

sess = tf.Session()
sess.run(tfprt)