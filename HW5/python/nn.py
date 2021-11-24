import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    b = np.zeros(out_size)
    
    # Normalized Initialization by Xavier
    boundary = np.sqrt(6 / (in_size + out_size))
    W = np.random.uniform(-boundary, boundary, (in_size, out_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = X @ W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = []
    for i in range(len(x)):
        si = np.exp(x[i] - np.max(x[i]))
        res.append(si / np.sum(si))

    return np.array(res)

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = - np.sum(y * np.log(probs))
    preds = np.argmax(probs, axis=1)
    acc = np.sum([yi[pred] == 1 for yi, pred in zip(y, preds)]) / len(y)

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    X, _, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    delta *= activation_deriv(post_act)
    grad_W = X.T @ delta   # this gives in_size x out_size
    grad_b = np.ones(len(delta)) @ delta  # this gives 1 x out_size
    grad_X = delta @ W.T

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    XY = np.hstack((x, y))
    np.random.shuffle(XY)

    batches = []
    for i in range(0, len(y), batch_size):
        XYi = XY[i:i+batch_size]
        batches.append((XYi[:, :len(x[0])], XYi[:, len(x[0]):]))
    
    return batches
