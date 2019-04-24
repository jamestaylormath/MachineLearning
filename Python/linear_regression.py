import numpy as np
import pandas as pd

#insert_ones(X) returns the matrix (1,X), obtained  by appending a column of 1's to the front of X
def insert_ones(X):
    return np.c_[np.ones(X.shape[0]), X]

class MLR():
    """Multivariate linear regression model. Models the (linear) relationship between 
    response vector Y and predictor matrix X.
    """

    def __init__(self):
        pass
    
    def fit(self,X,Y):
        #Replace X with the augmented matrix (1,X) and then compute the QR decomposition of X^T X
        X = insert_ones(X)
        Q, R = np.linalg.qr(X.T @ X)
    
        self.coeffs = np.linalg.inv(R) @ (X@Q).T @ Y
        
        train_preds = X @ self.coeffs
        
        self.rss = ((Y-train_preds)**2).sum()
        self.tss = ((Y - Y.mean())**2).sum()
        self.rsq = 1 - self.rss/self.tss
        self.mse = self.rss/X.shape[0]
        
    def predict(self,X):
        X = insert_ones(X)
        return(X @ self.coeffs)
    
    def summary(self):
        print("Coefficients: ", self.coeffs)
        print("RSS: %f \nTSS: %f \nRsq: %f \nMSE: %f \n" % (self.rss, self.tss, self.rsq, self.mse))
