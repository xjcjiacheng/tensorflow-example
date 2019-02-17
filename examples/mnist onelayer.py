import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
max_epoch = 25
batch_size = 64
dir = "E/tmp/data"
data = input_data.read_data_sets(dir,one_hot=True)
X = tf.placeholder(tf.float32,shape=[None,784])
Y = tf.placeholder(tf.float32,shape=[None,10])
W = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))
b = tf.Variable(tf.zeros([10]))
Y_= tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=Y_))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1)),tf.float32))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(max_epoch):
        t = int(data.train.num_examples/batch_size)
        ave_loss = 0
        for i in range(t):
            train_x,train_y = data.train.next_batch(batch_size)
            _,loss = sess.run([train_op,cost],feed_dict={X:train_x,Y:train_y})
            ave_loss += loss
        ave_loss /= t
        if (epoch+1)%1==0:
            print("epoch:{0:04d},cost:{1:6.4f}".format(epoch+1,ave_loss))
    print("finish")
    print("test...")
    accuracy1 = sess.run(accuracy,feed_dict={X:data.test.images,Y:data.test.labels})
    print("accuracy:{:6.4f}%".format(accuracy1*100))