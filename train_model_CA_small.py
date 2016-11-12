
from testchunk_small import testLoader
import cPickle
import numpy as np
import tensorflow as tf
from minibatch_small import nvdm_minibatch
import os
from vector_utils import find_norm

np.random.seed(0)
tf.set_random_seed(0)

def xavier_init(fan_in , fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
#     fan_in = in_and_out[0]
#     fan_out = in_and_out[1]
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


if __name__ == "__main__":


    data_ = "train_reviews_small.json"
    vocab_ = "vocab_reviews_small.pkl"
    #ifrom sklearn.datasets import fetch_20newsgroups
    #twenty_train = fetch_20newsgroups(subset='train')   
    #data_ = twenty_train.data
    #print "Download 20 news group data completed"
    A = testLoader(data_, vocab_)
    batch_size = 10 #there are 100 reviews total

    with tf.Session() as sess_1:
	#vae = nvdm_minibatch(sess_1, len(A.vocab), hidden_dim=50, encoder_hidden_dim=[500, 500], initializer=xavier_init, output_activation=tf.nn.softmax)
        vae = nvdm_minibatch(sess=sess_1 , input_dim=len(A.vocab), hidden_dim=5, encoder_hidden_dim =[20, 20], generator_hidden_dim = [20, 20], initializer = xavier_init, transfer_fct=tf.nn.relu , output_activation=tf.nn.softmax, batch_size=5, learning_rate = 0.001, mode = 'gather')
            
        vae._train()
        # Training cycle
        training_epochs = 10 #there are 100 training reviews
        batch_size = 10
        n_samples = len(data_)
        display_step = 1
        save_step = 10 #save every epoch
	count_epoch = 0
	count_minibatch = 0
	filename = "total_loss_Nov11_small.txt"
	with open(filename, 'w') as f:	
	        for epoch in range(training_epochs):
          	    batch_data = A.get_batch(batch_size)
            	    avg_cost = 0.
	            total_batch = int(n_samples / batch_size)
	            # Loop over all batches
	            batch_id = 0
		    count_epoch += 1
	            for batch_ in batch_data:
				count_minibatch +=1
		#	print 'minibatch', count_minibatch
	               		collected_data = [chunks for chunks in batch_]
	                ##### Here batch_xs ( Bag of words with count of words)
	                ##### Here mask_xs  ( Bag of words with 1 at the index of words in doc , no counts)
	                ##### Here mask_negative is not using ( Tried with negative sampling )
	      	 		batch_xs , mask_xs , mask_negative  = A._bag_of_words(collected_data)
	                ###### Here batch_flattened gives position of words in all documents into one array
	                ###### because gather_nd does not support gradients . So , we have to use tf.gather
	               		batch_flattened = np.ravel(batch_xs)
	               		index_positions = np.where( batch_flattened > 0 )[0] ####### We want locs where , data ( word ) present in document 
	
	                # Fit training using batch data
	               		if vae.mode == 'gather':
	                   
	                  		cost , R_loss_, l_loss  = vae.partial_fit(find_norm(batch_xs) , batch_xs.shape[0] , index_positions)
	                	else:
	                    		cost , R_loss_, l_loss  = vae.partial_fit(mask_xs , batch_xs.shape[0] , mask_xs.astype(np.float32))
	                	avg_cost += cost/(n_samples*batch_size) #CA: modified to divide by n_samples*batch_size
	                	print "Cost {} is".format(cost)
				f.write(str(count_epoch) + ' ' + str(count_minibatch) + ' '+ str(cost) + ' ' + str(avg_cost) + ' ' + str(R_loss_) + '\n')
	                	#if count_minibatch % 1 == 0: #every minibatches of size 5, so every 5 examples
	
	            # Display logs per epoch step
	            if epoch % display_step == 0:
	      		print "Epoch:", '%04d' % (epoch+1), \
	       		"cost=", "{:.9f}".format(avg_cost/total_batch)
	
	            if epoch % save_step == 0:
	               	vae.save(checkpoint_dir = "checkpoint", global_step = epoch)
