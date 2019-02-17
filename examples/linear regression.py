import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
max_epoch =50
x = np.random.rand(100)*10
y = 3*x + np.random.rand(100)+8
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.constant(0,tf.float32))
b = tf.Variable(tf.constant(0,tf.float32))
Y_ = tf.multiply(W,X) + b
cost = tf.reduce_mean(tf.square(Y-Y_))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(max_epoch):
        for (x_mini,y_mini) in zip(x,y):
            sess.run([train_op],feed_dict={X:x_mini,Y:y_mini})
        if (epoch+1)%10 == 0:
            loss = sess.run(cost,feed_dict={X:x,Y:y})
            print("epoch:",epoch+1,"cost;","{0:6.4f}".format(loss),"W",sess.run(W),"b",sess.run(b))
    print("finish")
    train_loss ,Y_train= sess.run([cost,Y_],feed_dict={X:x,Y:y})
    print("train_loss:{0:6.4f}".format(train_loss))
    plt.plot(x,y,"or",label = "original")
    plt.plot(x,Y_train,label = "prediction")
    plt.legend()
    plt.show()
    x_test = np.random.rand(50)*10
    y_test = 3*x_test +np.random.rand(50)+8
    Y_test = sess.run(Y_,feed_dict={X:x_test,Y:y_test})
    plt.plot(x_test,y_test,"ob",label = "test_original")
    plt.plot(x_test,Y_test,label = "prediction")
    plt.legend()
    plt.show()
