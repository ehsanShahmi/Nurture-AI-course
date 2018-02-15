import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores_f = np.zeros((num_train, num_classes))

  for i in range(num_train):
      scores_f = X[i].dot(W)
      scores_f -= np.max(scores_f)
  
      p = np.exp(scores_f) / np.sum(np.exp(scores_f))
      #print (p.shape)
      loss -= np.log(p[y[i]])
      #loss += loss_temp
        
      for j in range(num_classes):
          dW[:,j] += X[i] * (p[j] - (j == y[i]))
    
  loss /= num_train
  loss += reg * np.sum(W * W)
    
  dW /= num_train
  dW += reg * np.sum(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores_f = np.zeros((num_train, num_classes))

  scores_f = X.dot(W)
  scores_f -= np.max(scores_f)
  #print (scores_f.shape)
  
  p = np.exp(scores_f) / np.sum(np.exp(scores_f))
  #print (p.shape)
  
  index_for_y = [np.arange(num_train), y]
  #print (index)
  loss_temp = -np.log(p[index_for_y])
  #print (loss_temp.shape)
  loss = np.sum(loss_temp)
  loss = (loss / num_train) + reg * np.sum(np.square(W))
  

  ones_for_y = np.zeros((num_train,num_classes))
  ones_for_y[index_for_y] = 1
  dW = X.T.dot(p - ones_for_y)
  dW = (dW / num_train) + reg * np.sum(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

