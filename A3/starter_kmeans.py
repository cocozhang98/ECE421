import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
is_valid = False

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    return tf.transpose(tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(X,0),tf.expand_dims(MU,1))),2))

K = 3
numEpochs = 1000

X = tf.placeholder("float",shape=[None, dim])
MU = tf.Variable(tf.truncated_normal([k, dim],stddev=0.05))
dist = distanceFunc(X,MU)
loss = tf.reduce_sum(tf.reduce_min(dist,axis = 1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(numEpochs):
  cenVal, lossVal, _ = sess.run([MU, loss, optimizer], feed_dict={X: data})
  loss_history = np.append(loss_history, lossVal)
  if i % 10 == 0:
    print("iteration:", i, "loss", lossVal)