import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:,j]=dW[:,j]+X[i,:]
        dW[:,y[i]]=dW[:,y[i]]-X[i,:]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW =dW/num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW= dW+ reg * 2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  condition_ = np.zeros(W.shape)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################



  scores=X.dot(W)# N by C
  correct_class_score= scores[np.arange(X.shape[0]),y]
  margin=scores -np.vstack(correct_class_score)+np.vstack(np.ones(correct_class_score.shape))#N by C
  locations_zero=[np.arange(scores.shape[0]),y]
  margin[locations_zero]=0
  New_margin=np.maximum(margin,np.zeros((margin.shape[0],margin.shape[1])))
  loss=np.sum(New_margin)
  loss/=num_train
  loss+=reg *np.sum(W*W)

  condition_ = np.greater(margin, np.zeros((margin.shape[0], margin.shape[1])))#N by C
  condition_[np.arange(X.shape[0]),y]=0#N by C
  dW+=(X.transpose()).dot(condition_)
  RightHandSide=(X.transpose()).dot(condition_)
  ddW=np.multiply(X, np.vstack(np.sum(condition_, axis=1)))
  print(ddW.shape)
  for i in xrange(num_train):
    dW[:,y[i]]-=ddW[i,:]
  



#  for i in xrange(num_train):
#    scores = X[i].dot(W) #1 by C
#    correct_class_score = scores[y[i]] # 1 by 1


#    margin = scores - np.repeat(correct_class_score - 1,num_classes) # 1 by C

#    condition = np.greater(margin, np.zeros(margin.shape))
#    condition[y[i]]=False #1 by C

#    dW+= np.vstack(X[i,:])*condition #D*C
#    dW[:,y[i]] -= np.multiply(X[i,:],  sum(condition))

  dW =dW/num_train
  # Add regularization to the loss.
  dW= dW+ reg * 2 * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
