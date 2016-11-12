#Written by Sarath, modified by Carolyn
import tensorflow as tf
import os
import numpy as np
#import cPickle
from testchunk_small import testLoader
#from reader_CA_Oct27 import TextReader
#from base import Model

class nvdm_minibatch(object):

	def __init__(self, sess, input_dim, hidden_dim, encoder_hidden_dim, generator_hidden_dim,  initializer = tf.random_normal, transfer_fct = tf.nn.relu, output_activation = tf.nn.softmax, batch_size=10, learning_rate=0.001, mode='gather'):

		self.transfer_fct = transfer_fct
		self.output_activation = output_activation
		self.mode = mode
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.encoder_hidden_dim = encoder_hidden_dim
		self.generator_hidden_dim = generator_hidden_dim
		self.initializer = initializer
		self.dynamic_batch_size = tf.placeholder(tf.int32, shape=None)
		self.batch_size = batch_size
		self.learning_rate = learning_rate/batch_size #added by Carolyn - need to scale learning rate by minibatch size.
		self.x = tf.placeholder(tf.float32, [None, input_dim])

		self.mask = tf.placeholder(tf.float32, [None, input_dim])
		self.gather_mask = tf.placeholder(tf.int64, [None])
		
		self._create_network()
		self._create_loss_optimizer()

		self.saver = tf.train.Saver()
		self.init = tf.initialize_all_variables()

		self.sess = sess

	def _train(self):
		self.sess.run(self.init)

	def _initialize_weights(self):
		Weights_encoder = {}
		Biases_encoder = {}
		Weights_generator = {}
		Biases_generator = {}

		with tf.variable_scope("encoder"):
			for i in xrange(len(self.encoder_hidden_dim)):
				if i ==0:
					Weights_encoder['W_{}'.format(i)] = tf.Variable(self.initializer(self.input_dim, self.encoder_hidden_dim[i]))
					Biases_encoder['b_{}'.format(i)] = tf.Variable(tf.zeros([self.encoder_hidden_dim[i]], dtype=tf.float32))

				else:
					Weights_encoder['W_{}'.format(i)] = tf.Variable(self.initializer(self.encoder_hidden_dim[i-1], self.encoder_hidden_dim[i]))
					Biases_encoder['b_{}'.format(i)] = tf.Variable(tf.zeros([self.encoder_hidden_dim[i]], dtype=tf.float32))
			Weights_encoder['out_mean'] = tf.Variable(self.initializer(self.encoder_hidden_dim[i], self.hidden_dim))
			Weights_encoder['out_log_sigma'] = tf.Variable(self.initializer(self.encoder_hidden_dim[i], self.hidden_dim))

			Biases_encoder['out_mean'] = tf.Variable(tf.zeros([self.hidden_dim], dtype=tf.float32))
			Biases_encoder['out_log_sigma'] = tf.Variable(tf.zeros([self.hidden_dim], dtype=tf.float32))

		with tf.variable_scope("generator"):

			Weights_generator['out_mean'] = tf.Variable(self.initializer(self.hidden_dim, self.input_dim))
			Biases_generator['out_mean'] = tf.Variable(tf.zeros([self.input_dim], dtype=tf.float32)) #b_xi
	
			return Weights_encoder, Weights_generator, Biases_encoder, Biases_generator


	def _create_network(self):
		self.Weights_encoder, self.Weights_generator, self.Biases_encoder, self.Biases_generator = self._initialize_weights()

		self.z_mean, self.z_log_sigma_sq = self._encoder_network(self.Weights_encoder, self.Biases_encoder)

		eps = tf.random_normal((self.dynamic_batch_size, self.hidden_dim), 0, 1, dtype=tf.float32)

		self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
		self.X_reconstruction_mean = self._generator_network(self.Weights_generator, self.Biases_generator)

	def _encoder_network(self, weights, biases):

		encoder_results = {}
		with tf.variable_scope("encoder_function"):
			for i in xrange(len(weights) - 2):
				if i == 0:
					encoder_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(self.x, weights['W_{}'.format(i)]), biases['b_{}'.format(i)]))
				else:
					encoder_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(encoder_results['res_{}'.format(i-1)], weights['W_{}'.format(i)]), biases['b_{}'.format(i)]))

			z_mean = tf.add(tf.matmul(encoder_results['res_{}'.format(i)], weights['out_mean']), biases['out_mean'])

			z_log_sigma_sq = tf.add(tf.matmul(encoder_results['res_{}'.format(i)], weights['out_log_sigma']), biases['out_log_sigma'])

			return (z_mean, z_log_sigma_sq)


	def _generator_network(self, weights, biases):
		generator_results = {}

		with tf.variable_scope("generator_function"):
			for i in xrange(len(weights) - 2):
				if i == 0:
					decoder_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(self.z, weights['W_{}'.format(i)]), biases['b_{}'.format(i)]))
				else:
					decoder_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(decoder_results['res_{}'.format(i-1)], weights['W_{}'.format(i)]), biases['b_{}'.format(i)])) 		
			x_reconstr_mean = self.output_activation(tf.add(tf.matmul(self.z, weights['out_mean']), biases['out_mean']))
			print 'x_reconstr_mean shape', tf.shape(x_reconstr_mean)
			return x_reconstr_mean



	def _create_loss_optimizer(self):

		if self.mode != 'gather':
			self.log_recons = tf.log(self.X_reconstruction_mean + 1e-10)

		if self.mode == 'gather':
			#CA: UNCOMMENT THIS FOR ORIGINAL
			#self.interm_res = tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask) + 1e-10)
			#self.reconstr_loss = reconstr_loss = -tf.reduce_sum(self.interm_res)

			self.reconstr_loss = reconstr_loss = -tf.reduce_sum(self.x*tf.log(self.X_reconstruction_mean + 1e-10) + (1 - self.x)*tf.log(1e-10 + 1 - self.X_reconstruction_mean), 1)
			#print 'self.interm_res', self.interm_res
			#print 'self.reconstr_loss', self.reconstr_loss
			#print 'shape of reconstr_loss', tf.shape(self.reconstr_loss)
			###self.reconstr_loss = reconstr_loss = -tf.reduce_sum(self.x*tf.log(1e-10 + self.X_reconstruction_mean, [-1]) + (1 - self.x)*tf.log(1e-10 + 1 - self.X_reconstruction_mean), 1) #from https://jmetzen.github.io/2015-11-27/vae.html
			#print 'shape of self.gather_mask', tf.shape(self.gather_mask) #(1,)
			#print 'shape of tf.reshape(self.X_reconstruction_mean, [-1])', tf.shape(tf.reshape(self.X_reconstruction_mean, [-1])) #(1,)
			#print 'shape of tf.log(..)', tf.shape(tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask) + 1e-10)) #(1,)
			#print 'shape of self.x', tf.shape(self.x) #(2,) #Should be of shape (10, 1207).
			#print 'shape of gathering', tf.shape(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask)) #(1,)
			#print 'shape of mult', tf.shape(self.x*tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask) + 1e-10)) #(2,)
			#print 'shape of reduce_sum', tf.shape(-tf.reduce_sum(self.x*tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask) + 1e-10)))
			#print 'shape of reduce_sum 2', tf.shape((tf.ones_like(self.x) - self.x)*tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask) + 1e-10))
			#self.reconstr_loss = reconstr_loss = -tf.reduce_sum(self.x*tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask) + 1e-10) + (1 - self.x)*tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean, [-1]), self.gather_mask) + 1e-10), 1) #Added by CA 
			self.latent_loss = latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

			self.cost = tf.reduce_mean(reconstr_loss + latent_loss)

			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

	def partial_fit(self, X, dynamic_batch_size, MASK): #modified CA: added reconstr_loss and latent_loss

		if self.mode == 'gather':
			#print 'X is', X
			#print 'shape of X is', X.shape #[10, 1207]
			opt, cost, recons_loss, lat_loss  = self.sess.run((self.optimizer, self.cost, self.reconstr_loss, self.latent_loss), feed_dict = {self.x: X, self.dynamic_batch_size:dynamic_batch_size, self.gather_mask:MASK})

		else:
			opt, cost, recons_loss, lat_loss = self.sess.run((self.optimizer, self.cost, self.reconstr_loss, self.latent_loss), feed_dict = {self.x:X, self.dynamic_batch_size: dynamic_batch_size, self.mask:MASK})

		return cost, recons_loss, lat_loss

	def transform(self, X):
		return self.sess.run(self.z_mean, feed_dict={self.X: X})

	def generator(self, z_mu = None):

		if z_mu is None:
			z_mu = np.random.normal(size=self.network_architecture["n_z"])

		return self.sess.run(self.X_reconstruction_mean, feed_dict={self.z: z_mu})

	def reconstruct(self, X):

		return self.sess.run(self.X_reconstruction_mean, feed_dict={self.x:X})

	def save(self, checkpoint_dir, global_step = None):

		print(" [*] Saving checkpoints...")
		model_name = type(self).__name__
		model_dir = self.__class__.__name__

		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step)
		print "saved in {}".format(checkpoint_dir)

	#from base.py
	def load(self, checkpoint_dir):

		print(" [*} Loading checkpoints...")
		model_dir = self.get_model_dir()
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dr)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			print("load success")
			return True
		else:
			print("load failed")
			return False 










