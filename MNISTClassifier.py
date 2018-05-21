from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
import tensorflow as tf
from absl import flags
from absl import app


FLAGS = flags.FLAGS

# General
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Training
flags.DEFINE_integer("epochs", 50, "Number of epochs")
flags.DEFINE_integer("batchsize", 100, "Training batchsize")


class MNISTClassifier(object):

	def __init__(self, sess):
		super(MNISTClassifier, self).__init__()
		self.sess = sess
		self.build_model()
		self.saver = tf.train.Saver(max_to_keep=1)

	def build_model(self):
		self.input = tf.placeholder(
			shape=[None, 28, 28, 1],
			dtype=tf.float32,
			name="input",
		)
		self.labels = tf.placeholder(
			shape=[None, 10],
			dtype=tf.float32,
			name="labels",
		)
		self.is_train = tf.placeholder(
			shape=None,
			dtype=tf.bool,
			name="train_mode"
		)
		conv1 = tf.layers.conv2d(
			inputs=self.input,
			filters=32,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)
		pool1 = tf.layers.max_pooling2d(
			inputs=conv1,
			pool_size=[2, 2],
			strides=2
		)
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=64,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu
		)
		pool2 = tf.layers.max_pooling2d(
			inputs=conv2,
			pool_size=[2, 2],
			strides=2
		)
		pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
		dense = tf.layers.dense(
			inputs=pool2_flat,
			units=1024,
			activation=tf.nn.relu
		)
		dropout = tf.layers.dropout(
			inputs=dense,
			rate=0.4,
			training=self.is_train,
		)
		logits = tf.layers.dense(
			inputs=dropout,
			units=10
		)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
		self.optimize = tf.train.AdamOptimizer().minimize(self.loss)
		self.predictions = tf.argmax(input=logits, axis=1)
		
	def train(self):
		mnist = tf.contrib.learn.datasets.load_dataset("mnist")
		x = np.array(mnist.train.images).reshape(-1, 28, 28, 1)
		y = np.eye(10)[np.asarray(mnist.train.labels, dtype=np.int32)]
		
		n_iter = len(x) / FLAGS.batchsize
		for i in np.arange(FLAGS.epochs):
			losses = []
			for j in np.arange(n_iter):
				start = int(j * FLAGS.batchsize)
				end = int(start + FLAGS.batchsize)
				minibatch_x = x[start:end]
				minibatch_y = y[start:end]
				if end > len(x):
					minibatch_x = np.concatenate((minibatch_x, x[:end - len(x)]))
					minibatch_y = np.concatenate((minibatch_y, y[:end - len(x)]))
				loss, _ = self.sess.run([self.loss, self.optimize], feed_dict={self.input: minibatch_x, self.labels: minibatch_y, self.is_train: True})
				losses.append(loss)
			if i % 5 == 0:
				print("Epoch #{}/{} - Mean Loss: {}".format(i + 1, FLAGS.epochs, np.mean(losses)))
				self.saver.save(self.sess, "./model/mnist_classifier/", global_step=i)
		self.saver.save(self.sess, "./model/mnist_classifier/", global_step=i)

	def test(self):
		self.saver.restore(self.sess, "./model/mnist_classifier/")
		mnist = tf.contrib.learn.datasets.load_dataset("mnist")
		x = np.array(mnist.test.images).reshape(-1, 28, 28, 1)
		y = np.eye(10)[np.asarray(mnist.test.labels, dtype=np.int32)]
		predictions = self.sess.run(self.predictions, feed_dict={self.input: x, self.is_train: False})

	def load(self):
		self.saver.restore(self.sess, tf.train.latest_checkpoint("./model/mnist_classifier/"))

def main(unused_argv):
	sess = tf.Session()
	classifier = MNISTClassifier(sess)
	sess.run(tf.global_variables_initializer())
	if FLAGS.train:
		classifier.train()
	elif FLAGS.test:
		classifier.test()


if __name__ == "__main__":
	app.run(main)
