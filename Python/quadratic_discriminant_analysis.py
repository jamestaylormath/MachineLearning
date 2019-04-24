import numpy as np
import pandas as pd
import math

def get_sample_covariance_matrix(X,means):
    return (1/(X.shape[0]-1))*((X-means).T @ (X-means))

def quad_discrim_fn(x,means,cov,cov_det,cov_inv,prior):
    return (-1/2)*math.log(cov_det) + (-1/2)*((x-means).T @ cov_inv @ (x-means)) + math.log(prior)
        
class QDA():
    """Quadratic discriminant analysis model. Assumes our data comes from two populations, wherein 
    for each the pdf of X = (X_1, ..., X_p) is multivariate normal. Assigns data x to the population
    whose quadratic discriminant function is largest when evaluated at x.

    Parameters:
        priors: [float, float]
            Prior probabilities for the two populations. Should sum to 1.
    """

    def __init__(self, priors=[0.5,0.5]):
        self.priors = priors
    
    def fit(self,X,Y):
        #Separate data belonging to different populations
        X0 = X[Y==0]
        X1 = X[Y==1]
    
        #Calculate vector of sample means of each population
        means0 = X0.mean(axis = 0)
        means1 = X1.mean(axis = 0)
    
        #Calculate sample covariance matrices for each population
        cov0 = get_sample_covariance_matrix(X0,means0)
        cov1 = get_sample_covariance_matrix(X1,means1)
        
        #Calculate the determinants of the covariance matrices
        cov0_det = np.linalg.det(cov0)
        cov1_det = np.linalg.det(cov1)
        
        #Compute the Moore-Penrose pseudoinverses of the covariance matrices
        cov0_inv = np.linalg.pinv(cov0)
        cov1_inv = np.linalg.pinv(cov1)
        
        #The two discriminant (score) functions
        self.discrim_fn0 = lambda x: quad_discrim_fn(x,means0,cov0,cov0_det,cov0_inv,self.priors[0])
        self.discrim_fn1 = lambda x: quad_discrim_fn(x,means1,cov1,cov1_det,cov1_inv,self.priors[1])
    
    def predict(self,X):
        scores = [[self.discrim_fn0(x), self.discrim_fn1(x)] for x in X]
        return np.fromiter(map(lambda x: np.argmax(x), scores), dtype = np.int)
