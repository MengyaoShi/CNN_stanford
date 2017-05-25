import numpy as np
from random import shuffle
from past.builtins import xrange
from numpy import linalg as LA
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

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  X_transpose=X.transpose()

  scores=X.dot(W)
  Exp_scores=np.exp(scores)
  Norm_scores=np.sum(Exp_scores, axis=1)
  Prob_scores=Exp_scores/np.vstack(Norm_scores)
  loss+=np.sum(-np.log(Prob_scores[np.arange(X.shape[0]),y]))
  loss /=X.shape[0]
  loss+=reg*np.sum(W*W)
  dscores=Prob_scores#N by C
  dscores[range(X.shape[0]),y]-=1
  dscores/=X.shape[0]
  dW=np.dot(X.T, dscores)
  dW=dW+reg *2*W


 # for i in xrange(num_train):
 #   scores = X[i].dot(W) #1 by C
 #   Exp_scores=np.exp(scores)
 #   Norm_scores=np.sum(Exp_scores)
 #   Prob_scores=Exp_scores/Norm_scores #1 by C
 #   Li=Prob_scores[y[i]]
 #   loss+=-np.log(Li)
 #   dW[:,y[i]]+=-X_transpose[:,i]/Exp_scores[y[i]]
 #   for j in xrange(num_classes):
 #       dW[:,j]+=X_transpose[:,i]/Norm_scores
  
  

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
#  loss /= num_train
 # dW =dW/num_train
  # Add regularization to the loss.
 # loss += reg * np.sum(W * W)
  #dW= dW+ reg * 2 * W
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
  scores=X.dot(W)
  Exp_scores=np.exp(scores)
  Norm_scores=np.sum(Exp_scores, axis=1)
  Prob_scores=Exp_scores/np.vstack(Norm_scores)
  loss+=np.sum(-np.log(Prob_scores[np.arange(X.shape[0]),y]))
  loss /=X.shape[0]
  loss+=reg*np.sum(W*W)
  dscores=Prob_scores#N by C
  dscores[range(X.shape[0]),y]-=1
  dscores/=X.shape[0]
  dW=np.dot(X.T, dscores)
  dW=dW+reg *2*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

