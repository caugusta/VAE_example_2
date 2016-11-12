# VAE_example_2

This repo illustrates usage of a Variational Autoencoder for review text from actual user reviews on Amazon. This is a small set of 100 user reviews.

To train the model:

python train_model_CA_small.py

This will output a file total_loss_Nov11_small.py, which contains four major items:

epoch # + ' ' + minibatch # + ' ' + reconstuction loss + ' ' + KL divergence (for each review in the minibatch).

What each file does:

Preprocessing is handled in testchunk_small.py
The model is coded up in minibatch_small.py
train_reviews_small.json is the training data
vocab_reviews_small.pkl is the vocabulary
vector_utils.py contains some useful functions (we're only using one of them)

No other files are necessary to run this. The rest are just for my record keeping. Thank you to s4sarath for valuable input and especially for breaking down the reviews into minibatches.



