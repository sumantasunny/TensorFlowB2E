import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([[4,3,2,1],[1,2,3,4],[4,3,2,1],[1,2,3,4]])

x3 = tf.multiply(tf.transpose(x1), x2)
#x4 = [x3]
tfprt = tf.Print(x3, [x3, tf.shape(x3), "OK !! : "])
#x4 = tf.multiply(tfprt, x3)
#print x4
x4 = [10, 10, 10, 10]

tfadd = tf.add(tfprt, x4)
tfprt2 = tf.Print(tfadd, [tfadd])

with tf.Session() as sess:
    sess.run(tfprt2)
    sess.run(x3)
    print sess.run(tf.add(x1, x2))
    output = sess.run(tf.transpose(x1)*x2)
    print output
