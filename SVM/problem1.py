'''
    Problem 1: Implement linear and Gaussian kernels and hinge loss
'''

import numpy as np
from sklearn.metrics.pairwise  import euclidean_distances


def linear_kernel(X1, X2):
    
    """
    Compute linear kernel between two set of feature vectors.
    The constant 1 is not appended to the x's.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    
    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return linear kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=linear kernel evaluated on column i from X1 and column j from X2.
    """
    #########################################
    # if m1 and m2 are 1 - linear kernel of the two vectors
    # The linear kernel is the dot product? 
    #if (X1.shape[1] == 1 and X2.shape[1] == 1):
    #The linear kernel is the dot product of the two input matrices
    return (X1.T).dot(X2)
     
    #
    #########################################



def Gaussian_kernel(X1, X2, sigma=1):
    """
    Compute Gaussian kernel between two set of feature vectors.
    
    The constant 1 is not appended to the x's.
    
    For your convenience, please use euclidean_distances.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    sigma: Gaussian variance (called bandwidth)

    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return Gaussian kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=Gaussian kernel evaluated on column i from X1 and column j from X2

    """
    #########################################
    #euclidean_distances(X1, X2)
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    return np.exp(-1 * ((euclidean_distances(X1.T, X2.T) )** 2)/(2 * sigma * sigma))
    #########################################


def hinge_loss(z, y):
    """
    Compute the hinge loss on a set of training examples
    z: 1 x m vector, each entry is <w, x> + b (may be calculated using a kernel function)
    y: 1 x m label vector. Each entry is -1 or 1
    :return: 1 x m hinge losses over the m examples
    """
    #########################################
    loss = 1 - np.multiply(y,z)
    zeros = np.zeros(loss.shape)
    loss = np.maximum(loss, zeros)
    return loss
    #########################################
