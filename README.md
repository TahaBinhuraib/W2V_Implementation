## Why Do we Need Word2vec?

* Preserves contextual relationship between words
* You can add and subtract word vectors and get meaningful results
* Better results in training deep neural nets

### The working of word2vec

The objective function forces the words that occur in similar contexts to have similar n-dimensional word embedding; vector representations

### The two main algorithms for word2vec:

1. CBOW:

   * Predicts the target word from the context

2. Skip Gram:

   * Predict the context words from the target

   

### Architecture

![](/home/tbh/Documents/w2v/Images/Word2Vec-CBOW-and-Skip-gram-There-are-two-different-methods-in-the-Word2Vec-algorithm.png)

* **The projection contains the hidden layer which we obtained by multiplying an *mn* matrix with the input vector.** 


