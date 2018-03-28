#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    try:
        x_l = np.linalg.norm(x, axis=1, keepdims=True)
        x /= x_l
    except:
        raise NotImplementedError
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0],
                                [1, 2]]))
    print (x)
    ans = np.array([[0.6, 0.8],
                    [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    try:
        u0 = outputVectors[target]
        z = outputVectors.dot(predicted)
        #y_hat = softmax(z)
        cost = np.log(np.sum(np.exp(z)[None].T, axis=0)) - u0.dot(predicted)
        #cost = np.sum(np.exp(z)[None].T, axis=0) - u0.dot(predicted)
        
        #gradPred = -u0 + np.sum(y_hat.reshape(-1,1).dot(outputVectors))
        gradPred = -u0 + np.sum(np.exp(z)[None].T*outputVectors, axis=0)
        
        #grad = y_hat.reshape(-1,1) * np.repeat(predicted.reshape((1,outputVectors.shape[0])), outputVectors.shape[0], axis=0)
        grad = np.exp(z)[None].T * np.tile(predicted, (outputVectors.shape[0],1))
        grad[target] -= predicted
    except:
        raise NotImplementedError
    ### END YOUR CODE
    
    assert gradPred.shape == predicted.shape
    assert grad.shape == outputVectors.shape
    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    try:
        u0 = outputVectors[target]
        s0 = sigmoid(u0.dot(predicted))
        neg_samples = outputVectors[indices[1:]]
        #s = sigmoid(-neg_samples.dot(predicted))
        s = sigmoid(-neg_samples.dot(predicted))[None].T
        cost = -np.log(s0) - np.sum(np.log(s))
        
        gradPred = (s0 - 1.)*u0 - np.sum((s - 1.)*neg_samples)
        
        grad = np.zeros(outputVectors.shape)
        grad[target] = (s0 - 1.) * predicted
        #grad[indices[1:]] = -(s - 1.) * np.repeat(predicted.reshape((1,K)), K, axis=0)
        grad[indices[1:]] = -(s - 1.) * np.tile(predicted, (K, 1))
        
    except:
        raise NotImplementedError
    ### END YOUR CODE

    assert gradPred.shape == predicted.shape
    assert grad.shape == outputVectors.shape
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    try:
        predictedword = inputVectors[tokens[currentWord]]
        for worditem in contextWords:
            target = tokens[worditem]
            temp_cost, temp_gradPred, temp_grad = word2vecCostAndGradient(predictedword, target, outputVectors, dataset)
            
            cost += temp_cost
            gradIn[tokens[currentWord]] += temp_gradPred
            gradOut += temp_grad
    except:
        raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    try:
        target = tokens[currentWord]
        predictedWord = np.zeros(inputVectors.shape[1])
        for wordItem in contextWords:
            predictedWord += inputVectors[tokens[wordItem]]
            
        cost, gradPred, gradOut = word2vecCostAndGradient(predictedWord, target, outputVectors, dataset)
        for wordItem in contextWords:
            gradIn[tokens[wordItem]] += gradPred
    except:
        raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = int(wordVectors.shape[0]/2)
    inputVectors = wordVectors[:N,:]
    outputVectors = wordVectors[N:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N, :] += gin / batchsize / denom
        grad[N:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    
    print ("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper( skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    
    print ("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),dummy_vectors)

    print ("\n=== Results ===")
    print (skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print ("")
    print (skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print ("")
    print (cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print ("")
    print (cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
