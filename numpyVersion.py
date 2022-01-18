import numpy as np
from collections import OrderedDict
from itertools import chain
import matplotlib.pyplot as plt
plt.style.use("seaborn")

with open('text.txt', 'r') as file:
    data = file.read().replace('\n', '')

def mapping(tokens):
    wordToID = {}
    idToWord = {}

    for i, token in enumerate(tokens):
        wordToID[token] = i
        idToWord[i] = token

    return wordToID, idToWord

# We have to tokenize the text before sending it to the mapping function
# Since our data is relatively clean we can use the .split method to achieve this
tokens = data.split(" ")
# We have to make sure that the mapping saves the order of the tokens
tokens = sorted(set(tokens), key=tokens.index)
wordToID, idToWord = mapping(tokens)

def oneHotEncoding(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    nTokens = len(tokens)
    
    for i in range(nTokens):
        idx = chain(
            range(max(0, i - window), i), 
            range(i, min(nTokens, i + window + 1)))
        for j in idx:
            if i == j:
                continue
            X.append(oneHotEncoding(word_to_id[tokens[i]], len(word_to_id)))
            y.append(oneHotEncoding(word_to_id[tokens[j]], len(word_to_id)))
    
    return np.asarray(X), np.asarray(y) 

X, y = generate_training_data(tokens, wordToID, 2)

# Our hidden layer will be a 5 dimensional vector, and that will be the size of our embeddings
hidden = 5
model = {
        "w1": np.random.randn(len(wordToID), hidden),
        "w2": np.random.randn(hidden, len(wordToID))
    }
    
def w2vForward(X, weights, returnCache = True):
    
    cache = {}
    cache["a1"] = X @ weights["w1"]
    cache["a2"] = cache["a1"] @ weights["w2"]
    cache["z"] = softmax(cache["a2"])
    
    if not returnCache:
        return cache["z"]
    return cache    

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

def crossEnt(z, y):
    return - np.sum(np.log(z) * y)

def back(model, X, y, alpha):
    cache  = w2vForward(X, model)
    # scratch derivation for the backpropagation
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    # Updating the weights 
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    # Final Loss
    return crossEnt(cache["z"], y)

def plot(model, back, X, y, *, iterations = 100, learningRate = 0.01):
    history = [back(model, X, y, learningRate) for _ in range(iterations)]

    plt.plot(range(len(history)), history, color="skyblue")
    plt.show()


def main():
    
    plot(model, back, X, y)

if __name__ == '__main__':
    main()