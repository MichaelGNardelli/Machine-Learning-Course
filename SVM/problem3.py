# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from problem1 import *
from problem2 import *

import numpy as np
from numpy import random

import copy

class SVMModel():
    """
    The class containing information about the SVM model, including parameters, data, and hyperparameters.

    DONT CHANGE THIS DEFINITION!
    """
    def __init__(self, train_X, train_y, C, kernel_function, sigma=1):
        """
            train_X: n x m training feature matrix. n: number of features; m: number training examples.
            train_y: 1 x m labels (-1 or 1) of training data.
            C: a positive scalar
            kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
            sigma: need to be provided when Gaussian kernel is used.
        """
        # data
        self.train_X = train_X
        self.train_y = train_y
        self.n, self.m = train_X.shape

        # hyper-parameters
        self.C = C
        self.kernel_func = kernel_function
        self.sigma = sigma

        # parameters
        self.alpha = np.zeros((1, self.m))
        self.b = 0

def train(model, max_iters = 10, record_every = 1, max_passes = 1, tol=1e-6):
    """
    SMO training of SVM
    model: an SVMModel
    max_iters: how many iterations of optimization
    record_every: record intermediate dual and primal objective values and models every record_every iterations
    max_passes: each iteration can have maximally max_passes without change any alpha, used in the SMO alpha selection.
    tol: numerical tolerance (exact equality of two floating numbers may be impossible).
    :return: 4 lists (of iteration numbers, dual objectives, primal objectives, and models)
    Hint: refer to subsection 3.5 "SMO" in notes.
    """
    #########################################
    iterations = []
    dual = []
    primal = []
    models = []
    should_record = 0
    #Performs the desired number of iterations
    for t in range(max_iters):
        #Calcs the num passes
        num_passes = 0
        #Counts the number of passes to continue for a given iteration
        while num_passes < max_passes:
            #Tracks if change occurs
            num_changes = 0
            for i in range(model.m):
                #Calculates the KKT conditions
                if check_KKT(model, i):
                    j = i
                    #Ensures different from j and i
                    while j == i:
                        j = random.randint(model.m)
                    #Calcs the max and min alpha values
                    H = 0
                    L = 0
                    #Calculates if they are different values
                    if model.train_y[0][i] * model.train_y[0][j] == -1:
                        H = min((model.C, (model.C - (model.alpha[0][i] -model.alpha[0][j]))))
                        L = max((0, -(model.alpha[0][i] - model.alpha[0][j])))
                    else:
                    #Calculates the values if they are the same
                        H = min((model.C, (model.alpha[0][i] + model.alpha[0][j])))
                        L = max((0, (model.alpha[0][i] + model.alpha[0][j] - model.C)))
                    
                    #Calculates the value of g
                    g_i = 0
                    g_j = 0
                    gauss = (model.kernel_func.__name__ == "Gaussian_kernel")
                    #Cacluates g for all values
                    for count in range(model.m):
                        if gauss:
                            g_i = g_i + model.alpha[0][count] * model.train_y[0][count] * model.kernel_func(model.train_X.T[i].T, model.train_X.T[count].T, model.sigma)
                            g_j = g_j + model.alpha[0][count] * model.train_y[0][count] * model.kernel_func(model.train_X.T[j].T, model.train_X.T[count].T, model.sigma)
                            
                        else:
                            g_i = g_i + model.alpha[0][count] * model.train_y[0][count] * model.kernel_func(model.train_X.T[i], model.train_X.T[count])
                            g_j = g_j + model.alpha[0][count] * model.train_y[0][count] * model.kernel_func(model.train_X.T[j], model.train_X.T[count])
                    #Finalizes g by adding b
                    g_i = g_i + model.b 
                    g_j = g_j + model.b

                    # if gauss:
                    #     v_i = g_i - (model.alpha[0][i] * model.train_y[0][i] * model.kernel_func(model.train_X.T[i], model.train_X.T[i], model.sigma) + model.alpha[0][j] * model.train_y[0][j] * model.kernel_func(model.train_X.T[i], model.train_X.T[j], model.sigma)) - model.b
                    #     v_j = g_j - (model.alpha[0][j] * model.train_y[0][j] * model.kernel_func(model.train_X.T[j], model.train_X.T[j], model.sigma) + model.alpha[0][i] * model.train_y[0][i] * model.kernel_func(model.train_X.T[j], model.train_X.T[i], model.sigma)) - model.b
                    # else:
                    #     v_i = g_i - (model.alpha[0][i] * model.train_y[0][i] * model.kernel_func(model.train_X.T[i], model.train_X.T[i]) + model.alpha[0][j] * model.train_y[0][j] * model.kernel_func(model.train_X.T[i], model.train_X.T[j])) - model.b
                    #     v_j = g_j - (model.alpha[0][j] * model.train_y[0][j] * model.kernel_func(model.train_X.T[j], model.train_X.T[j]) + model.alpha[0][i] * model.train_y[0][i] * model.kernel_func(model.train_X.T[j], model.train_X.T[i])) - model.b

                    #Calculates E w
                    E_i = g_i - model.train_y[0][i]
                    E_j = g_j - model.train_y[0][j]

                    alpha_j_new = 0
                    if gauss:
                        alpha_j_new = model.alpha[0][j] + (model.train_y[0][j] * (g_i - model.train_y[0][i] - (g_j - model.train_y[0][j]))) / (model.kernel_func(model.train_X.T[i].T, model.train_X.T[i].T, model.sigma) + model.kernel_func(model.train_X.T[j].T, model.train_X.T[j].T, model.sigma) - 2 * model.kernel_func(model.train_X.T[i].T, model.train_X.T[j].T, model.sigma))
                    else:
                        alpha_j_new = model.alpha[0][j] + (model.train_y[0][j] * (g_i - model.train_y[0][i] - (g_j - model.train_y[0][j]))) / (model.kernel_func(model.train_X.T[i], model.train_X.T[i]) + model.kernel_func(model.train_X.T[j], model.train_X.T[j]) - 2 * model.kernel_func(model.train_X.T[i], model.train_X.T[j]))
                    
                    if alpha_j_new > H:
                        alpha_j_new = H 

                    if alpha_j_new < L:
                        alpha_j_new = L 
                    
                    if abs(alpha_j_new - model.alpha[0][j]) < tol:
                        continue
                    
                    alpha_i_new = model.alpha[0][i] + model.train_y[0][i] * model.train_y[0][j] * (model.alpha[0][j] - alpha_j_new)
                        
                    b_i = -1
                    b_j = -1
                    if gauss:
                        b_i = model.b - E_i - model.train_y[0][i] * (alpha_i_new - model.alpha[0][i]) * model.kernel_func(model.train_X.T[i].T, model.train_X.T[i].T, model.sigma) - model.train_y[0][j] * (alpha_j_new - model.alpha[0][j]) * model.kernel_func(model.train_X.T[i].T, model.train_X.T[j].T, model.sigma)   
                        b_j = model.b - E_j - model.train_y[0][i] * (alpha_i_new - model.alpha[0][i]) * model.kernel_func(model.train_X.T[i].T, model.train_X.T[j].T, model.sigma) - model.train_y[0][j] * (alpha_j_new - model.alpha[0][j]) * model.kernel_func(model.train_X.T[j].T, model.train_X.T[j].T, model.sigma)   
                    else:
                        b_i = model.b - E_i - model.train_y[0][i] * (alpha_i_new - model.alpha[0][i]) * model.kernel_func(model.train_X.T[i], model.train_X.T[i]) - model.train_y[0][j] * (alpha_j_new - model.alpha[0][j]) * model.kernel_func(model.train_X.T[i], model.train_X.T[j])   
                        b_j = model.b - E_j - model.train_y[0][i] * (alpha_i_new - model.alpha[0][i]) * model.kernel_func(model.train_X.T[i], model.train_X.T[j]) - model.train_y[0][j] * (alpha_j_new - model.alpha[0][j]) * model.kernel_func(model.train_X.T[j], model.train_X.T[j]) 

                    if b_i > 0 and b_i < model.C:
                        model.b = b_i 
                    elif b_j > 0 and b_j < model.C:
                        model.b = b_j 
                    else:
                        model.b = (b_i + b_j)/2    
                    
                    model.alpha[0][i] = alpha_i_new
                    model.alpha[0][j] = alpha_j_new

                    num_changes = num_changes + 1
                #End if for KKT loop
            #End for loop
            if num_changes == 0:
                num_passes = num_passes + 1
            else:
                num_passes = 0
            #Tells of skip
        #End of while loop
        should_record = should_record + 1
        
        if should_record == record_every:
            iterations.append(t)
            dual.append(dual_objective_function(model.alpha, model.train_y, model.train_X, model.kernel_func, model.sigma))
            primal.append(primal_objective_function(model.alpha, model.train_y, model.train_X, model.b, model.C, model.kernel_func, model.sigma)[0][0])
            models.append(model)
            should_record = 0
    #All iterations finished
    return iterations, dual, primal, models

    #########################################

def predict(model, test_X):
    """
    Predict the labels of test_X
    model: an SVMModel
    test_X: n x m matrix, test feature vectors
    :return: 1 x m matrix, predicted labels
    """
    #########################################
    res_list = []
    gauss = (model.kernel_func.__name__ == "Gaussian_kernel")
    for i in range(test_X.shape[1]):
        pred_sum = 0
        if gauss:
            for j in range(model.m):
                pred_sum = pred_sum + model.alpha[0][j] * model.train_y[0][j] * model.kernel_func(model.train_X.T[j].T, test_X.T[i].T, model.sigma)
        else:
            for j in range(model.m):
                pred_sum = pred_sum + model.alpha[0][j] * model.train_y[0][j] * model.kernel_func(model.train_X.T[j], test_X.T[i])
        if pred_sum + model.b < 0:
            res_list.append(-1)
        else:
            res_list.append(1)
        pred_sum = 0
    
    predicted_labels = np.array(res_list)
    return predicted_labels.T
    #########################################


def check_KKT(model, i):
    z = decision_function(model.alpha, model.train_y, model.train_X, model.b, model.kernel_func, model.sigma, (model.train_X.T[i]))
    
    if model.alpha[0][i] == 0 and (model.train_y[0][i] * z) <= 1:
        return True 
    elif model.alpha[0][i] > 0 and (model.train_y[0][i] * z) != 1:
        return True 
    elif model.alpha[0][i] == model.C and model.train_y[0][i] * z > 1:
        return True 
    
    if model.alpha[0][i] > model.C or model.alpha[0][i] < 0:
        return True
    
    return False
