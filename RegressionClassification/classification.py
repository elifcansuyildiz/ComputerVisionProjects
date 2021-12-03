"""
Elif Cansu YILDIZ 04/2021
"""
import numpy as np
import utils

#applies to sigmoid to X in element-wise operation
def sigmoid(X):
    return 1/(1+np.exp(-X))

#updates w values according to the gradient descent method
def grad_desc(w, grad, lr):
    return w - lr * grad

#inputs (num_samples, num_features) X and (num_features, num_dims) w, returns (num_features * num_features) hessian matrix
def hessian_matrix(X, w):
    Y_pred = sigmoid(X @ w).reshape(-1,1)
    return - Y_pred.T @ (1 - Y_pred) * X.T @ X

#updates w using newtons method
def newtons_method(w, grad, hessian, lr):
    return w + lr * np.linalg.pinv(hessian) @ grad

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns gradient with respect to w (num_features)
def log_llkhd_grad(X, Y, w):
    Y = Y.copy()
    Y[Y==-1.0] = 0.0
    return X.T @ (sigmoid(X @ w) - Y) 

####################################################################################################################################
#Classification
####################################################################################################################################
def run_classification(X_tr, Y_tr, X_te, Y_te, step_size, optim_method="grad_desc", max_iter=10000):
    print('classification with step size '+str(step_size))
    Y_tr = Y_tr.reshape(-1,1)
    Y_te = Y_te.reshape(-1,1)

    # Training
    w = np.zeros((X_tr.shape[1], 1))
    for step in range(max_iter):
        grad = log_llkhd_grad(X_tr, Y_tr, w)

        if optim_method=="grad_desc": #(Note: gradient descent can be used if a task needs 10000 iterations of training. In this case, the newton method is slow)
            w = grad_desc(w, grad, step_size)
            if step%1000 == 0:
                loss = utils.get_loss(X_tr, Y_tr, w)
                accuracy = utils.get_accuracy(X_tr, Y_tr, w)
                print('step='+str(step)+' loss='+str(loss)+' accuracy='+str(accuracy))
        else:
            # Newton Method: (Note: With Newton Method, this problem can be solved in 2 iterations)
            hessian = hessian_matrix(X_tr, w)
            w = newtons_method(w, grad, hessian, step_size)

            loss = utils.get_loss(X_tr, Y_tr, w)
            accuracy = utils.get_accuracy(X_tr, Y_tr, w)
            print('step='+str(step)+' loss='+str(loss)+' accuracy='+str(accuracy))

        

    # Testing
    loss = utils.get_loss(X_te, Y_te, w)
    accuracy = utils.get_accuracy(X_te, Y_te, w)
    print('test set loss='+str(loss)+' accuracy='+str(accuracy))
