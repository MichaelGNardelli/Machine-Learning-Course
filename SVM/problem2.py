# -------------------------------------------------------------------------
'''
    Problem 2: Compute the objective function and decision function of dual SVM.

'''
from problem1 import *

import numpy as np

# -------------------------------------------------------------------------
def dual_objective_function(alpha, train_y, train_X, kernel_function, sigma):
    """
    Compute the dual objective function value.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix. n: number of features; m: number training examples.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the dual objective function value at alpha
    Hint: refer to the objective function of Eq. (47).
          You can try to call kernel_function.__name__ to figure out which kernel are used.
    """
    #########################################
    dual_obj_val = np.sum(alpha)
    gauss = False
    if kernel_function.__name__ == "Gaussian_kernel":
        gauss = True
    
    in_loop = 0
    for i in range(len(alpha.T)): #Used to get value m
        for j in range(len(alpha.T)):
            if gauss:
                in_loop = in_loop + (alpha.T[i][0] * alpha.T[j][0] * train_y.T[i][0] * train_y.T[j][0] * kernel_function(np.array([train_X.T[i]]).T, np.array([train_X.T[j]]).T, sigma))
                in_loop = in_loop[0][0]
            else:
                in_loop = in_loop + alpha.T[i][0] * alpha.T[j][0] * train_y.T[i][0] * train_y.T[j][0] * kernel_function(train_X.T[i], train_X.T[j].T)

    dual_obj_val = dual_obj_val - 0.5 * in_loop

    
    return dual_obj_val
    #########################################


# -------------------------------------------------------------------------
def primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma):
    """
    Compute the primal objective function value.
    When with linear kernel:
        The primal parameter w is recovered from the dual variable alpha.
    When with Gaussian kernel:
        Can't recover the primal parameter and kernel trick needs to be used to compute the primal objective function.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    b: bias term
    C: regularization parameter of soft-SVM
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: a scalar representing the primal objective function value at alpha
    Hint: you need to use kernel trick when come to Gaussian kernel. Refer to the derivation of the dual objective function Eq. (47) to check how to find
            1/2 ||w||^2 and the decision_function with kernel trick.
    """
    #########################################
    prim_obj_val = 0
    gauss = False
    if kernel_function.__name__ == "Gaussian_kernel":
        gauss = True
    
    if gauss:
        w_fin = 0
        for k in range(train_X.shape[1]):
            for l in range(train_X.shape[1]):
                w_fin = w_fin + alpha.T[l] * alpha.T[k] * train_y.T[k] * train_y.T[l] * kernel_function(np.array([train_X.T[k]]).T, np.array([train_X.T[l]]).T, sigma)
        w_fin = w_fin * 0.5
        ep_1 = []
        #ep_zeros = np.zeros(ep.shape)
        for j in range(train_X.shape[1]):
            temp_sum = 0
            for k in range(train_X.shape[1]):
                temp_sum = temp_sum + alpha.T[k] * train_y.T[j] * train_y.T[k] * kernel_function(np.array([train_X.T[j]]).T, np.array([train_X.T[k]]).T, sigma)
            ep_1.append(1 - temp_sum[0][0] - train_y.T[j]*b)

        ep = np.array(ep_1)
        ep_zeros = np.zeros(ep.shape)
        ep = np.maximum(ep, ep_zeros)
        
        return w_fin + C * np.sum(ep)
                    
    else:
        #Change if necessary
        w = np.zeros((alpha.shape[0], train_X.shape[0]))

        for i in range(train_X.shape[1]):
            w = w + alpha.T[i] * (train_X.T[i].T) * train_y.T[i]
        w = w.T
        ep_1 = []
        for j in range(train_X.shape[0]):
            ep_1.append(1 - train_y.T[j] * (w.T.dot(train_X.T[j]) + b))
        ep = np.array(ep_1)
        ep_zeros = np.zeros(ep.shape)
        ep = np.maximum(ep, ep_zeros)
        return 0.5 * kernel_function(w, w) + C * np.sum(ep)
        # + np.sum(ep) * C #Simplest version of the primal
    #########################################


def decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X):
    """
    Compute the linear function <w, x> + b on examples in test_X, using the current SVM.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    test_X: n x m2 test feature matrix.
    b: scalar, the bias term in SVM <w, x> + b.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: 1 x m2 vector <w, x> + b
    """
    #########################################
    gauss = False
    if kernel_function.__name__ == "Gaussian_kernel":
        gauss = True
    
    if gauss:
        vec = []
        temp_sum = 0
        for k in range(len(alpha[0])):
           temp_sum = temp_sum + alpha[0][k] * train_y[0][k] * kernel_function(train_X.T[k].reshape(-1, 1), test_X, sigma)
        return temp_sum + b
    else:
        w = np.zeros((alpha.shape[0], train_X.shape[0]))
        for i in range(train_X.shape[1]):
            w = w + alpha.T[i] * (train_X.T[i].T) * train_y.T[i]
        w = w.T
        return kernel_function(w, test_X) + b
    #########################################
