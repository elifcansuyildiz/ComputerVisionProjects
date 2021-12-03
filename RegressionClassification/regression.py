"""
Elif Cansu YILDIZ 04/2021
"""
import numpy as np
from numpy.linalg import inv
from sklearn.cluster import KMeans
import utils

#takes features with bias X (num_samples*(1+num_features)), centers of clusters C (num_clusters*(1+num_features)) and std of RBF sigma
#returns matrix with scalar product values of features and cluster centers in higher embedding space (num_samples*num_clusters)
def RBF_embed(X, C, sigma):
    return np.exp(-np.sum((X[:, None, :] - C[None, :, :])**2, axis=2)/sigma)

#generates (num_samples_X1 * num_samples_X2) kernel matrix using RBF Kernel
def RBF_kernel_trick(X1, X2, sigma):
    return np.exp(-0.5 * np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)/sigma**2) # np.sum(...) expression is equivalent to scipy.spatial.distance.cdist

############################################################################################################
#Linear Regression
############################################################################################################

def lin_reg(X, Y):
    """
    args:
        X: training data with bias (num_samples,(1+num_features))
        Y: target values (num_samples,target_dims)
    returns:
        w: regression coefficients ((1+num_features),target_dims)
        std: variance of target prediction separately for each target dimension
    """
    w = np.linalg.pinv(X.T @ X) @ X.T @ Y   # w.shape=((1+num_features)*target_dims) w=regression coefficients
    #w = np.linalg.lstsq(X,Y,rcond=0)[0]
    pred = X @ w
    diff = Y - pred
    # We assume that the difference between predicted and ground truth Y values is independent for each dimension.
    var = np.mean(diff**2, axis=0)
    std = np.sqrt(var)
    return w

def test_lin_reg(X, Y, w):
    """
    args:
        X: training data with bias (num_samples,(1+num_features))
        Y: target values (num_samples,target_dims)
        w: regression coefficients ((1+num_features),target_dims)
    returns:
        error: fraction of mean square error
    """
    var_y = np.var(Y, axis=0)
    err = np.mean((Y - (X @ w))**2, axis=0) / var_y  #MSE/var
    return err

def run_lin_reg(X_tr, Y_tr, X_te, Y_te):
    """
        X_tr, Y_tr: training data and target values
        X_te, Y_te: test data and target values
        returns error
    """
    w= lin_reg(X_tr, Y_tr)
    err = test_lin_reg(X_te, Y_te, w)

    print('MSE/Var Linear Regression')
    print("Error: ", err)

############################################################################################################
#Dual Regression
############################################################################################################
def run_dual_reg(X_tr, Y_tr, X_val, Y_val, X_te, Y_te):
    """
    args:
        X_tr, Y_tr: training data and target values
        X_val, Y_val: validation data and target values
        X_te, Y_te: test data and target values
    returns:
        error
    """
    err_list = []
    param_list = []

    print('MSE/Var Dual Regression on Validation Data for different sigma values')
    for sigma_pow in range(-5, 3):
        sigma = np.power(3.0, sigma_pow)

        # training
        Z = RBF_kernel_trick(X_tr, X_tr, sigma)
        psi = inv(Z) @ Y_tr

        # validation
        Z_k_val = RBF_kernel_trick(X_val, X_tr, sigma)
        err_dual = test_lin_reg(Z_k_val, Y_val, psi)

        err_list.append(err_dual)
        param_list.append({"sigma":sigma})

        print('sigma= {:.4f} Error: {}'.format(sigma, err_dual))


    # Find optimum sigma values independently for each Y dimension
    min_err_index = np.argmin(err_list, axis=0)
    opt_sigma = np.stack((param_list[min_err_index[0]]["sigma"], param_list[min_err_index[1]]["sigma"]))

    # Train model using opt_sigma for Y_0
    Z = RBF_kernel_trick(X_tr, X_tr, opt_sigma[0])
    psi = inv(Z) @ Y_tr  #training
    Z_k_te_0 = RBF_kernel_trick(X_te, X_tr, opt_sigma[0])
    err_dual_te_0 = test_lin_reg(Z_k_te_0, Y_te, psi)
    
    # Train model using opt_sigma for Y_1
    Z = RBF_kernel_trick(X_tr, X_tr, opt_sigma[1])
    psi = inv(Z) @ Y_tr  #training
    Z_k_te_1 = RBF_kernel_trick(X_te, X_tr, opt_sigma[1])
    err_dual_te_1 = test_lin_reg(Z_k_te_1, Y_te, psi)

    err_dual_te = err_dual_te_0
    err_dual_te[1] = err_dual_te_1[1]

    print("\nNote: Training two seperate models for different dimensions of Y")
    print('MSE/Var dual regression for test sigma='+str(opt_sigma))
    print("Error: ", err_dual_te)

############################################################################################################
#Non Linear Regression
############################################################################################################
def run_non_lin_reg(X_tr, Y_tr, X_val, Y_val, X_te, Y_te):
    """
    args:
        X_tr, Y_tr: training data and target values
        X_val, Y_val: validation data and target values
        X_te, Y_te: test data and target values
    returns:
        error
    """
    err_list = []
    param_list = []
    
    print('MSE/Var Non Linear Regression on Validation Data for different sigma values and number of clusters')
    for num_clusters in [10, 30, 100]:
        for sigma_pow in range(-5, 3):
            sigma = np.power(3.0, sigma_pow)

            C = KMeans(n_clusters=num_clusters).fit(X_tr).cluster_centers_

            # Training
            Z = RBF_embed(X_tr, C, sigma)
            Z_b = utils.add_bias(Z)
            w = lin_reg(Z_b, Y_tr)

            # Validation
            Z_val = RBF_embed(X_val, C, sigma)
            Z_val_b = utils.add_bias(Z_val)
            err_nonlin = test_lin_reg(Z_val_b, Y_val, w)
            
            err_list.append(np.array(err_nonlin))
            param_list.append({"sigma":sigma, "num_clusters": num_clusters})

            print('For sigma={:.4f}, num_clusters={}'.format(sigma, num_clusters), end=" ")
            print("Error: ", err_nonlin)

    #print("err_list:\n",np.array(err_list))

    min_err_index = np.argmin(err_list, axis=0)
    print("min err index: ", min_err_index)
    #print("param list:\n", param_list, "\n")
    
    opt_sigma = np.stack((param_list[min_err_index[0]]["sigma"], param_list[min_err_index[1]]["sigma"]))
    opt_num_clusters = np.stack((param_list[min_err_index[0]]["num_clusters"],param_list[min_err_index[1]]["num_clusters"]))

    # Best Hyperparameters for dimension=0 for dimension=0
    C = KMeans(n_clusters=opt_num_clusters[0]).fit(X_tr).cluster_centers_
    Z = RBF_embed(X_tr, C, opt_sigma[0])
    Z_b = utils.add_bias(Z)     
    w = lin_reg(Z_b, Y_tr)
    Z_te = RBF_embed(X_te, C, opt_sigma[0])
    Z_te_b = utils.add_bias(Z_te)
    err_nonlin_0 = test_lin_reg(Z_te_b, Y_te, w)
    
    # Best Hyperparameters for dimension=0 for dimension=1
    C = KMeans(n_clusters=opt_num_clusters[1]).fit(X_tr).cluster_centers_
    Z = RBF_embed(X_tr, C, opt_sigma[1])
    Z_b = utils.add_bias(Z)     
    w = lin_reg(Z_b, Y_tr)
    Z_te = RBF_embed(X_te, C, opt_sigma[1])
    Z_te_b = utils.add_bias(Z_te)
    err_nonlin_1 = test_lin_reg(Z_te_b, Y_te, w)
    
    err_nonlin[0] = err_nonlin_0[0]
    err_nonlin[1] = err_nonlin_1[1]
    
    print("\nNote: Training two seperate models for different dimensions of Y")
    print('MSE/Var Non Linear Regression on test data for sigma={:.4f}, num_clusters={} error: {}'.format(opt_sigma[0], opt_num_clusters[0], err_nonlin[0]))
    print('MSE/Var Non Linear Regression on test data for sigma={:.4f}, num_clusters={} error: {}'.format(opt_sigma[1], opt_num_clusters[1], err_nonlin[1]))
