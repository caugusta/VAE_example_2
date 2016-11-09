#Written by Carolyn Augusta
#Purpose: create a smaller test case for the minibatches
#This creates a file with the first 100 reviews
#and the accompanying vocabulary file

import json
import re
from nltk.corpus import stopwords
import string
import cPickle

cached_stopwords = set(stopwords.words("english"))

data_path = "/mnt/data/datasets/Amazon/AmazonTrain/train_Oct24.json"

with open(data_path, 'r') as f:
	loaded_json = json.load(f)

#This is the first review
#print loaded_json[0]

first_100_reviews = loaded_json[0:100]

small_data = "train_reviews_small.json"

with open(small_data, 'w') as g:
	written_json = json.dump(first_100_reviews, g)

#Build the vocabulary
vocab_small_data = "vocab_reviews_small.pkl"

words_out = re.sub(r'(\w+)\.([A-Z])', r'\1\ \2', ' '.join(first_100_reviews))
exclude = set(string.punctuation)
words_out = ''.join(ch for ch in words_out if ch not in exclude)
words_out = ''.join([i for i in words_out if not i.isdigit()])
words_out = words_out.lower()
words_out = re.split(' ', words_out)
words_out = [word for word in words_out if not word in cached_stopwords and len(word)>0]
words_out = set(words_out)

#There are 1207 words in the vocabulary
#print len(words_out)

#Now add indexes to words_out, create a dictionary

values_ = [i for i in xrange(len(words_out))]
pkldict = dict(zip(words_out, values_))
#print pkldict.keys()[0:10]
#print pkldict.values()[0:10]

with open(vocab_small_data, 'w') as h:
	written_pkl = cPickle.dump(pkldict, h)



