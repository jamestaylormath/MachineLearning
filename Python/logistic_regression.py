import numpy as np # linear algebra
import pandas as pd
import math

#Sticks a column of ones to the front of X
def insert_ones(X):
    return np.c_[np.ones(X.shape[0]), X]

#Returns f_beta(X) (discussed in "Logistic Regression Explained" in Math folder)
def f(beta, X):
    return 1/(1+np.exp(-np.sum(X*beta, axis=1)))

#Sets first component of vector to 0. Useful in calculating the gradient of the regularized loglikelihood
def first_component_to_zero(vector):
    c = np.ones(vector.shape[0])
    c[0] = 0
    return vector*c

#loglikelihood function with the L2 penalty included
def loglikelihood(beta, X, Y, reg_lambda):
    return (Y*np.log(f(beta,X)) + (1-Y)*np.log(1-f(beta,X))).sum() - reg_lambda*(beta**2)[1:].sum()

#Used for diagnostic purposes. Helps user view what's going on during the fitting process
def display_gradascent_info(step, learn, grad, beta, beta_change, likelihood, beta_new, likelihood_new):
    print("step:", step)
    print("learning rate:", learn)
    print("gradient:", grad)
    print("learn*grad:", learn*grad)
    print("beta:", beta)
    print("beta_change:", beta_change)
    print("likelihood:", likelihood)
    print("new beta:", beta_new)
    print("new likelihood:", likelihood_new)
    print('\n')
    
class LogisticRegression():
    def __init__(self, reg_lambda = 0, learn = 0.5, precision = 0.001, max_iters = 1000):
        self.reg_lambda = reg_lambda
        self.learn = learn
        self.precision = precision
        self.max_iters = max_iters
        self.beta = None
        
    def fit(self, X, Y):
        learn = self.learn #set learn = initial learning rate
        beta_change = 1 #Initial value; just needs to be > precision
    
        X = insert_ones(X) #stick a column of 1's to the front of X
        self.beta = np.ones(X.shape[1]) #initial value of coefficient vector
        likelihood = loglikelihood(self.beta, X, Y, self.reg_lambda) #Initial value of loglikelihood
        can_stop = False #Can't stop gradient ascent yet, we've barely started!
    
        step = 0
        while step < self.max_iters and can_stop == False:
            grad = (Y - f(self.beta,X)).T @ X - 2*self.reg_lambda*first_component_to_zero(self.beta)
        
            beta_new = self.beta + learn*grad
            beta_change = np.abs(beta_new - self.beta).max()
            likelihood_new = loglikelihood(beta_new,X,Y,self.reg_lambda)
        
            #display_gradascent_info(step, learn, grad, self.beta, beta_change, 
            #                     likelihood, beta_new, likelihood_new)
        
            if (beta_change > self.precision) or (likelihood_new <= likelihood):
            #beta hasn't stabilized or latest change to beta was a bad one
                can_stop = False
            else: 
                can_stop = True
        
            if (likelihood_new > likelihood) or ((not math.isfinite(likelihood)) and math.isfinite(likelihood_new)):
            #latest step was good...update beta, increase learning rate, and advance
                learn *= 1.5
                self.beta = beta_new
                likelihood = likelihood_new
                step += 1
            else:
            #latest step no bueno. Reduce learning rate and don't update beta
                learn /= 2
                
    def probabilities(self,X):
        return f(self.beta, insert_ones(X))
        
