import numpy as np
import pandas as pd
import math

#Returns the Euclidean distance between vectors A and B
def euclidean_distance(A,B):
    return math.sqrt(((A-B)**2).sum())

class KNN():
    def __init__(self, k = 5, norm = True, method = "reg"):
        self.k = k
        self.norm = norm
        self.method = method
        
    def predict(self, X_train, Y_train, X_test):
        #normalize data unless user requests otherwise
        if self.norm == True:
            train_mean = X_train.mean(axis=0) #vector of X_train's column means
            train_std = X_train.std(axis=0) #vector of X_train's column std devs
            X_train = (X_train - train_mean)/train_std #normalize X_train
            X_test = (X_test - train_mean)/train_std #normalize X_test
    
        preds = []
        for i,data in enumerate(X_test):
            knn_indices = np.argsort([euclidean_distance(data, x) for x in X_train])[:self.k]
            neighbor_vals = np.array([Y_train[j] for j in knn_indices])
            if self.method == "reg":
                prediction = neighbor_vals.mean()
            elif self.method == "class":
                prediction = np.argmax(np.bincount(neighbor_vals))
            preds.append(prediction)
            
        return preds
            
