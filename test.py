import tensorflow as tf
import tflearn
observation = tflearn.input_data(shape=[None, 4])
net = tflearn.fully_connected(observation, 256, activation="relu")
net = tflearn.fully_connected(net, 256, activation="relu")
net = tflearn.fully_connected(net, 256, activation="relu")
out = tflearn.fully_connected(net, 2, activation="softmax")
tf.keras.Input(
    shape=None,
    batch_size=None,
    dtype=None,
    sparse=None,
    batch_shape=None,
    name=None,
    tensor=None
)
reward_holder = 
action_holder = tf.placeholder(tf.int32, [None])

responsible_outputs = tf.gather(tf.reshape(out, [-1]), tf.range(0, tf.shape(out)[0] * tf.shape(out)[1], 2) + action_holder)

loss = -tf.reduce_mean(tf.log(responsible_outputs) * reward_holder)

optimizer = tf.train.AdamOptimizer()
update = optimizer.minimize(loss)
