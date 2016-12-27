import tensorflow as tf
import numpy as np

def tmodel(x,hiddenNeuros):
    '''
    construct a R^2 ---> R^1 model
    @hidden1 :int. number of neurons in hidden layer.
    @x : R^(n x 2).
    '''
    # hidden layer 1
    with tf.name_scope('hidden'):
        weights = tf.Variable(tf.ones([2,hiddenNeuros]), name='weights')
        hidden = tf.nn.sigmoid(tf.matmul(x, weights))
    # output layer
    with tf.name_scope('linear_output'):
        weights = tf.Variable(tf.ones([hiddenNeuros, 1]), name='weights')
        out = tf.matmul(hidden,weights)
    return out


def tloss(model, y):
    '''
    @y, R^(n,1)
    '''
    diff = tf.square(tf.subtract(model, y))
    loss = tf.reduce_mean(diff)
    return loss

def ttrain(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(loss)


if '__main__' == __name__:

    with tf.Session() as session:
        x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        model = tmodel(x,10)
        loss = tloss(model, y)
        train = ttrain(loss, 0.01)
        init = tf.global_variables_initializer()
        session.run(init)
        batch = 10
        x1 = np.ones((batch,2))
        y1 = np.ones((batch,1))
        for i in xrange(100):
            session.run(train, feed_dict={x: x1, y: y1})
        print(session.run(model, feed_dict={x:x1}))
