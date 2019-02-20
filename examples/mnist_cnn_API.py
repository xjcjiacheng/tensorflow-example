import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
dir = "E/tmp/data"
num_epoch = 2000
batch_size = 64
data = input_data.read_data_sets(dir,one_hot=True)
def cnn(X_,reuse,is_train):
    with tf.variable_scope("cnn",reuse = reuse):
        X = X_["images"]
        X_image = tf.reshape(X,[-1,28,28,1])
        P1 = tf.layers.conv2d(X_image,32,5,1,"SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          activation=tf.nn.relu)
        Z1 = tf.layers.max_pooling2d(P1,2,2,"SAME")
        P2 = tf.layers.conv2d(Z1,64,5,1,"SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          activation=tf.nn.relu)
        Z2 = tf.layers.max_pooling2d(P2,2,2,"SAME")
        flatten_Z = tf.contrib.layers.flatten(Z2)
        Z3 = tf.layers.dropout(flatten_Z,0.7,)
        Z4 = tf.layers.dense(Z3,1024,activation = tf.nn.relu)
        Z5 = tf.layers.dense(Z4,10)
        return Z5

def model_fn(mode,features,labels):
    train_logits = cnn(features,reuse=False,is_train=True)
    test_logits = cnn(features,reuse=True,is_train=False)
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=tf.nn.softmax(test_logits))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits,
                                                                  labels=labels))
    accuracy = tf.metrics.accuracy(tf.argmax(test_logits,1),tf.argmax(labels,1))
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost,global_step=tf.train.get_or_create_global_step())
    estimator = tf.estimator.EstimatorSpec(mode,loss = cost,train_op=train_op,eval_metric_ops={"accuracy":accuracy})
    return estimator
model = tf.estimator.Estimator(model_fn)
input_op = tf.estimator.inputs.numpy_input_fn(x={"images":data.train.images},y=data.train.labels,num_epochs=None,
                                              shuffle=True,batch_size=batch_size)
model.train(input_op,steps = num_epoch)
input_op = tf.estimator.inputs.numpy_input_fn(x={"images":data.test.images},y=data.test.labels,shuffle=False)
e = model.evaluate(input_op)
print("accuracy:{}".format(e["accuracy"]))