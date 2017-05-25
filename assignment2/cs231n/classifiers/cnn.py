from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        F=num_filters
        C,H,W=input_dim
   
        stride=1
        pad=(filter_size-1)//2
        H2=(H+2*pad-filter_size)/stride+1
        W2=(W+2*pad-filter_size)/stride+1
        HH=filter_size
        WW=filter_size
        #after first conv, dimension is N F H2 W2
        
        pool_height=2
        pool_width=2
        pool_stride=2
        H3=(H2-pool_height)/pool_stride+1
        W3=(H2-pool_width)/pool_stride+1
        # after pooling layer, dimension is N F H3 W3

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params['W1']=np.random.normal(0, weight_scale, F*C*HH*WW).reshape(F,C,HH,WW) # W has F C H W
        print(self.params['W1'].shape) 
        self.params['W2']=np.random.normal(0, weight_scale, F*H3*W3*hidden_dim).reshape(F*H3*W3,hidden_dim)
        self.params['W3']=np.random.normal(0,weight_scale, hidden_dim*num_classes).reshape(hidden_dim,num_classes)
        self.params['b1']=np.zeros(F)
        self.params['b2']=np.zeros(hidden_dim)
        self.params['b3']=np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        print(len(self.params))
        print(W1.shape)
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        loss=0
        grads={}
        conv_out, conv_cache=conv_forward_naive(X, W1, b1, conv_param)
        out_relu,cache_relu=relu_forward(conv_out) 
        out_pool, cache_pool=max_pool_forward_naive(out_relu, pool_param)
        out_aff,cache_aff=affine_forward(out_pool, W2, b2)
        out_relu1,cache_relu1=relu_forward(out_aff)
        scores,cache_aff1=affine_forward(out_relu1, W3, b3)
        loss,dx=softmax_loss(scores, y)
        loss+=0.5*self.reg*np.sum(W1 ** 2)+0.5*self.reg*np.sum(W2 ** 2)+self.reg*np.sum(W3 ** 2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        dx1, dW3, db3=affine_backward(dx, cache_aff1)
        dW3+=self.reg*W3
        dx2=relu_backward(dx1, cache_relu1)
        dx2, dW2, db2=affine_backward(dx2, cache_aff)
        dW2+=self.reg*W2
        dx3=max_pool_backward_naive(dx2, cache_pool) 
        dx4=relu_backward(dx3, cache_relu) 
        dx, dW1, db1=conv_backward_naive(dx4, conv_cache)
        dW1+=self.reg*W1
        grads['W1']=dW1
        grads['W2']=dW2
        grads['W3']=dW3
        grads['b1']=db1
        grads['b2']=db2
        grads['b3']=db3
 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
