import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
dir = "E/tmp/data"
num_step = 1000
batch_size = 128
data = input_data.read_data_sets(dir,one_hot=True)
def layers(X):
    z1 = tf.layers.dense(X["x"],256,activation = tf.nn.relu)
    z2 = tf.layers.dense(z1,256,activation=tf.nn.relu)
    z3 = tf.layers.dense(z2,10)
    return z3
def model_fn(mode,features,labels):
    logits = layers(features)
    prediction = tf.nn.softmax(logits)
    if model == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=prediction)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    acc = tf.metrics.accuracy(labels = tf.argmax(labels,1),predictions=tf.argmax(prediction,1))
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss,global_step=tf.train.get_global_step())
    estimator = tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op,eval_metric_ops={"acc":acc})
    return estimator
model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":data.train.images},y=data.train.labels,batch_size=batch_size,num_epochs=None,
                                              shuffle=True)
model.train(input_fn,steps = num_step)
input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":data.test.images},y = data.test.labels,batch_size=batch_size,
                                              shuffle=False)
e = model.evaluate(input_fn)
print("Accuracy:",e["acc"])