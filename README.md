# VAE_example_2

This repo illustrates usage of a Variational Autoencoder for review text from actual user reviews on Amazon. This is a small set of 100 user reviews.

To train the model:

python train_model_CA_small.py

You should see output like this:

Cost 34.8374481201 is
Cost 35.203125 is
Cost 38.5284080505 is
Cost 39.0439414978 is
Cost 45.2037277222 is
Cost 37.8635635376 is
Cost 28.7984447479 is
Cost 31.7935199738 is
Cost 29.0405216217 is
Cost 34.4803314209 is
Epoch: 0001 cost= 0.739152149
 [*] Saving checkpoints...
saved in checkpoint/nvdm_minibatch
Cost 34.8526992798 is
...

Also, there will be an output file total_loss_Nov11_small.py, which contains four major items:

epoch # + ' ' + minibatch # + ' ' + reconstuction loss + ' ' + KL divergence (for each review in the minibatch).

What each file does:

Preprocessing is handled in testchunk_small.py
The model is coded up in minibatch_small.py
train_reviews_small.json is the training data
vocab_reviews_small.pkl is the vocabulary
vector_utils.py contains some useful functions (we're only using one of them)

No other files are necessary to run this. The rest are just for my record keeping. Thank you to s4sarath for valuable input and especially for breaking down the reviews into minibatches.



