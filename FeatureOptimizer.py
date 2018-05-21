from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import tensorflow as tf
from absl import flags
from absl import app
import numpy as np
import pickle
from PIL import Image


from MNISTClassifier import MNISTClassifier


FLAGS = flags.FLAGS

# General
# flags.DEFINE_bool("train", False, "Train")
# flags.DEFINE_bool("test", False, "Test")


class FeatureOptimizer(object):

	def __init__(self, sess):
		super(FeatureOptimizer, self).__init__()
		self.sess = sess
		self.build_model()
		self.saver = tf.train.Saver(max_to_keep=1)
		self._model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
		self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self._model_variables]
		assigns = [tf.assign(v, p) for v, p in zip(self._model_variables, self._placeholders)]
		self._assign_op = tf.group(*assigns)

	def build_model(self):
		self.input_placeholder = tf.placeholder(
			shape=[1, 28, 28, 1],
			dtype=tf.float32,
			name="input_placeholder",
		)
		self.input = tf.Variable(
			initial_value=np.random.rand(1, 28, 28, 1),
			# initial_value=np.zeros((1, 28, 28, 1)),
			dtype=tf.float32,
			name="input",
		)
		self._assign_input = tf.assign(self.input, self.input_placeholder)
		with tf.variable_scope("model"):
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
			logits = tf.layers.dense(
				inputs=dense,
				units=10
			)
		# self.loss = -tf.reduce_sum(conv2[:, :, :, 0])
		# self.loss = -tf.reduce_sum(pool2_flat[0])
		self.loss = -logits[:, 0]
		self.grad = tf.gradients(self.loss, self.input)
		self.optimize = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(self.loss, var_list=[self.input])
		self.predictions = tf.argmax(input=logits, axis=1)
		self.confidence = tf.nn.softmax(logits)

	def load(self, parameters):
		self.sess.run(self._assign_op, feed_dict=dict(zip(self._placeholders, parameters)))

	def clip_input(self, values=None):
		if values is None:
			values = self.sess.run(self.input)
		values = np.clip(values, 0., 1.)
		# val_min = np.min(values)
		# val_max = np.max(values)
		# values -= val_min
		# values /= val_max - val_min
		# values = np.around(values)
		self.sess.run(self._assign_input, feed_dict={self.input_placeholder: values})

def main(unused_argv):
	sess = tf.Session()
	# temp = MNISTClassifier(sess)
	# temp.load()
	# parameters = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	with open("parameters.pkl", 'rb') as file:
		# pickle.dump(parameters, file)
		parameters = pickle.load(file)
	optimizer = FeatureOptimizer(sess)
	sess.run(tf.global_variables_initializer())
	optimizer.load(parameters)

	# print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model"))
	# quit()
	# print(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")[0]))
	# quit()
	# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	# x = np.array(mnist.test.images).reshape(-1, 28, 28, 1)
	with open("sample.pkl", 'rb') as file:
		sample = pickle.load(file)
	# optimizer.clip_input(sample)
	# print(sess.run(optimizer.grad)[0])
	# quit()
	image, _ = sess.run([optimizer.input, optimizer.optimize])
	image = (image.reshape(28, 28) * 256).astype(np.uint8)
	image = Image.fromarray(image)
	image.resize((100, 100)).show()
	prev_loss = None
	for i in np.arange(2000):
		image, _, loss = sess.run([optimizer.input, optimizer.optimize, optimizer.loss])
		image = (image.reshape(28, 28) * 256).astype(np.uint8)
		image = Image.fromarray(image)
		scale = int((np.random.rand() - 0.5) * 28)
		image.crop((scale, scale, 28-scale, 28-scale)).resize((28,28)).rotate(np.random.rand() * 180)
		image = np.array(image) / 256.
		image = image.reshape(1, 28, 28, 1)
		optimizer.clip_input(image)
		if i % 100 == 0:
			print(i)
			print("\t" + str(loss))
		# if prev_loss is not None:
		# 	if np.abs(prev_loss - loss) < 1e-3:
		# 		break
		prev_loss = loss
	# print(image)
	# print(sess.run(optimizer.grad)[0])
	i, confidence = sess.run([optimizer.predictions, optimizer.confidence])
	print("{}: {}".format(i, confidence[0, i]))
	print(confidence)
	image = (image.reshape(28, 28) * 256).astype(np.uint8)
	image = Image.fromarray(image)
	image.resize((100, 100)).show()



if __name__ == "__main__":
	app.run(main)
