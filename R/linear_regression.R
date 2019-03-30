#add_ones(X) returns the matrix (1:X), obtained by appending a column of 1's to the front of X
add_ones = function(x){
    x0 = rep(1,dim(x)[1])
    return(cbind(x0,x))
}

mlr = function(x,y){
    #Replace X with the augmented matrix (1:X) and then solve the equation
    #X^T X b = X^T y for b (the model coefficiens) with qr.solve
    x = add_ones(x)
    coeffs = qr.solve(t(x) %*% x, t(x) %*% y)
    
    #Compute details about the model fit
    train_preds = x %*% coeffs
    RSS = sum((y - train_preds)^2)
    TSS = sum((y-mean(y))^2)
    RSQ = 1 - RSS/TSS
    MSE = RSS/dim(x)[1]
    
    return(list(coeffs = coeffs, 
                rss = RSS, 
                tss = TSS, 
                rsq = RSQ, 
                mse = MSE, 
                train_preds = train_preds)
          )
}

#use model (output from mlr()) to make predictions for x_test
predict.mlr = function(model, x_test){
    x_test = add_ones(x_test)
    return(x_test %*% model$coeffs)
}
