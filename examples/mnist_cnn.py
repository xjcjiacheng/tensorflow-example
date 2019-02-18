import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
dir = "E/tmp/data"
data = input_data.read_data_sets(dir,one_hot=True)
num_step = 2
batch_size = 128
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias(shape):
    return tf.Variable(tf.zeros(shape))
def con2d(X,shape,activition=True):
    W = weight(shape)
    b = bias([shape[-1]])
    P = tf.nn.conv2d(X,W,strides=[1,1,1,1],padding="SAME") +b
    if activition:
        P = tf.nn.relu(P)
    return P
def max_pool(X):
    return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
X_ = tf.placeholder(tf.float32,shape=[None,784])
Y = tf.placeholder(tf.float32,shape=[None,10])
X = tf.reshape(X_,shape=[-1,28,28,1])
P1 = con2d(X,[5,5,1,32])
P1 = max_pool(P1)
P2 = con2d(P1,[5,5,32,64])
P2 = max_pool(P2)
P3 = tf.reshape(P2,[-1,7*7*64])
P3 = tf.matmul(P3,weight([7*7*64,1024]))+bias([1024])
P4 = tf.matmul(P3,weight([1024,10]))+bias([10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=P4,labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
cross_acc = tf.equal(tf.argmax(Y,1),tf.argmax(P4,1))
acc = tf.reduce_mean(tf.cast(cross_acc,tf.float32))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(num_step):
        ace_lost = 0
        pre_batch = 0
        t = int(data.train.num_examples/batch_size)
        for i in range(t):
            batch_x,batch_y = data.train.next_batch(batch_size)
            _,cost,pre= sess.run([train_op,loss,acc],feed_dict={X_:batch_x,Y:batch_y})
            ace_lost+=cost
            pre_batch+=pre
        ace_lost/=t
        pre_batch/=t
        if (epoch+1)%1==0 or epoch==0:
            print("epoch:{0},cost:{1:6.4f},accuracy:{2:6.4f}".format(epoch+1,ace_lost,pre_batch))
    print("finish")
    print("test...")
    ac = sess.run(acc,feed_dict={X_:data.test.images,Y:data.test.labels})
    print("accuracy:{:6.4f}".format(ac))