import numpy as np
import pandas as pd

def get_sample_covariance_matrix(X,means):
    return (1/(X.shape[0]-1))*((X-means).T @ (X-means))

class LDA():
    """Linear discriminant analysis model. Assumes our data comes from two populations, wherein 
    for each the pdf of X = (X_1, ..., X_p) is multivariate normal with a common covariance matrix. 
    Assigns data x to the population whose linear score function is largest when evaluated at x.

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
    
        #Compute the pooled covariance matrix and its Moore-Penrose pseudoinverse
        pooled_covariance = ((X0.shape[0]-1)*cov0 + (X1.shape[0]-1)*cov1)/(X0.shape[0]+X1.shape[0]-2)
        inv_pcm = np.linalg.pinv(pooled_covariance)
    
        #Compute coefficients for the two score functions
        intercept0 = (-1/2)*(means0 @ inv_pcm @ means0)
        coeffs0 = means0 @ inv_pcm
        coeffs0 = np.insert(coeffs0,0,intercept0)
    
        intercept1 = (-1/2)*(means1 @ inv_pcm @ means1)
        coeffs1 = means1 @ inv_pcm
        coeffs1 = np.insert(coeffs1,0,intercept1)
        
        self.coeffs = np.array([coeffs0, coeffs1])
    
    
    def predict(self,X):
        scores = (np.insert(np.array(X),0,1,axis=1) @ self.coeffs.T) + np.log(self.priors)
        return np.fromiter(map(lambda x: np.argmax(x), scores), dtype = np.int)
