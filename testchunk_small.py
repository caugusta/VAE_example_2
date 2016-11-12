import numpy as np
import json
import os
from itertools import islice, chain
import cPickle
import re
from nltk.corpus import stopwords

cached_stopwords = set(stopwords.words("english"))

class testLoader():

	def __init__(self, data_path, vocab_path):
		with open(data_path, 'r') as f:
			loaded_json = json.load(f)
		self.data = loaded_json
		with open(vocab_path, 'r') as g:
			loaded_pkl = cPickle.load(g)
		self.vocab = loaded_pkl		
		self.voc_to_idx = []
		#print 'self.data', self.data
		#print 'type of self.data', type(self.data)
		#self._vocab(self.data)
		#self._vocab_inverse()
		#print 'self.vocab', self.vocab

	def get_batch(self, batch_size):
		sourceiter = iter(self.data)
		if len(self.data) % batch_size == 0:
			iterations = len(self.data) / batch_size
		else:
			iterations = int(len(self.data) / batch_size) + 1
		for i in xrange(iterations):
			batch_data = islice(sourceiter, batch_size)
			yield chain([batch_data.next()], batch_data)

	def preprocess(self, text_chunk): #treats each review separately already, as a string. Returns the text of the review, without stopwords, etc.

		words_out = re.split(' ', text_chunk)
		words_out = re.sub(r'(\w+)\.([A-Z])', r'\1\ \2', ' '.join(words_out))
		words_out = re.sub(r"[^[a-zA-Z -']+", '', ''.join(words_out))
		#words_out = re.split(r'[\s]\s*', words_out.lower())
		words_out = words_out.lower()
		words_out = re.split(' ', words_out)		
		words_out = [word for word in words_out if not word in cached_stopwords and len(word)>0 and word in self.vocab]
		return words_out

	def _bag_of_words(self, review, vocab_size=None):
		if vocab_size == None:
			vocab_size = len(self.vocab)
		#print 'review', review #CA This is actually 10 reviews - I see all 10
		#print 'len(review)', len(review) #CA This is 10.
		#lengths_vec = [len(rev) for rev in review]
		# #ends up being characters print 'lengths_vec', lengths_vec #should be 10 elements, each is the number of words in a review
		voc_to_idx = self._vocab_to_idx(review)
		self.bow = [np.bincount(idx, minlength=vocab_size) for idx in voc_to_idx]
		#print 'self.bow', self.bow[0][0:100]
		#print 'self.bow', self.bow[0][101:200]
		self.bow = np.array(self.bow)
		div = (self.bow != 0).sum(1)
		#print 'div is', div
		div = div.astype(np.float32)
		#print 'div float is', div
		self.dow = self.bow.copy()
		self.bow = self.bow / div[:, None]
		#print 'self.bow', self.bow[0][0:100]
		#print 'self.bow', self.bow[0][101:200]	
		#self.bow = np.sum(self.bow, axis=0) #Added by CA 11 November 2016, but commented out because wrong: want to preserve shape
		#div = np.full(vocab_size, len(review))
		#self.bow = np.array(self.bow)/div #Added by CA 11 November 2016: division by number of words in review
		self.dow[self.dow > 0] = 1 #Selecting words that appear in the review. This is the mask fed in to partial in train
		self.negative_mask = self.dow.copy()
		self.negative_mask[self.negative_mask == 0] = -1 #self.negative_mask = 1 or -1
		return self.bow, self.dow, self.negative_mask


	def _vocab_to_idx(self, chunk_data):
		voc_to_idx = []
		preprocessed_data = map(self.preprocess, chunk_data)
		preprocessed_data_ = []
		self.data_index = []
		for index_pos, data_ in enumerate(preprocessed_data):
			if data_:
				preprocessed_data_.append(data_)
				self.data_index.append(index_pos)
		for chunks in preprocessed_data_:
			voc_to_temp = [self.vocab[w] for w in chunks if w in self.vocab]
			voc_to_idx.append(voc_to_temp)
		voc_to_idx = np.array(voc_to_idx)
		return voc_to_idx

if __name__ == "__main__":

	data_path = "train_reviews_small.json"
	vocab_path = "vocab_reviews_small.pkl"
	batch_size = 10	
#	data_path = "/mnt/data/datasets/Amazon/AmazonTrain/train_Oct24.json"
#	vocab_path = "/mnt/data/datasets/Amazon/AmazonTrain/vocab.pkl"
#	batch_size = 100
	Instance1 = testLoader(data_path, vocab_path)
	
	batch_data = Instance1.get_batch(batch_size)
	#CAROLYN - UNCOMMENT THIS
	for batch_ in batch_data:
		collected_data = [chunks for chunks in batch_]
		bow, dow, negative_mask = Instance1._bag_of_words(collected_data)
		#bow, dow, negative_mask = Instance1._bag_of_words(batch_)
		print bow.shape, dow.shape, negative_mask.shape
		print bow.max(), dow.max(), negative_mask.max()
		print bow[0][0:100]
		print bow[0][101:200]
		print bow[0][201:300]
	
		break

#	print "first batch:"
#	print next(next(batch_data))
#	#print "NEXT REVIEW"
#	#print next(next(batch_data))
#	
#	preprocessed = Instance1.preprocess(next(next(batch_data)))
#	print 'PREPROCESSED next review'
#	print preprocessed
#
#	print 'bag of words'
#	bow1 = Instance1._bag_of_words(preprocessed)
#	print 'bow1[0][0:100]', bow1[0][0:100]

	#preprocessed2 = Instance1.preprocess(next(next(batch_data)))
	#print 'bag of words'
	#bow2 = Instance1._bag_of_words(preprocessed2)
	#print 'bow2[0][0:100]', bow2[0][0:100]
	#print 'len(bow1[0])', len(bow1[0][0])
	#print 'bag of words divided', bow1[0][0]/len(bow1[0][0])
#	print 'dow'
#	print len(bow1[1])
#	print 'negative mask'
#	print len(bow1[2])
