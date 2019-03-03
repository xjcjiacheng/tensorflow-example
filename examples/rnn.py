import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
dir = "E/tmp/data"
data = input_data.read_data_sets(dir,one_hot=True)
num_step = 1000
batch_size = 128
num_hidden = 28
num_class = 10
num_input = 28
time_step = 28
X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])
x = tf.reshape(X,[-1,time_step,num_input])
w1 = tf.Variable(tf.truncated_normal([num_hidden,num_class],stddev=0.1))
b1 = tf.Variable(tf.zeros([num_class]))
def rnn(x,num_hidden):
    rnn = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)
    output,state = tf.nn.dynamic_rnn(rnn,x,dtype=tf.float32)
    return output,state
output,state = rnn(x,num_hidden)
output = tf.transpose(output,[1,0,2])
Y_ = tf.matmul(output[-1],w1) + b1
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_,labels=Y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
cross_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
acurrracy = tf.reduce_mean(tf.cast(cross_correct,tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_step):
        num = int(data.train.num_examples/batch_size)
        tmp_accuracy = 0
        tmp_cost = 0
        for i in range(num):
            batch_x,batch_y = data.train.next_batch(batch_size)
            _,cost1,accuracy1 = sess.run([train_step,cost,acurrracy],feed_dict={X:batch_x,Y:batch_y})
            tmp_cost += cost1
            tmp_accuracy += accuracy1
        tmp_accuracy /=num
        tmp_cost /= num
        if epoch%1 == 0:
            print(epoch,":cost",tmp_cost,"acc:",tmp_accuracy)
    print(sess.run(acurrracy,feed_dict={X:data.test.images,Y:data.test.labels}))