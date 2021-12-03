"""
Elif Cansu YILDIZ 04/2021
"""
import numpy as np

#prints block text for section headers
def print_section(text):
    filler = "##############################################"
    filler_w_text = ("##### " + text + " ").ljust(len(filler), "#")
    print("\n" + filler + "\n" + filler_w_text + "\n" + filler)

def add_bias(X):
    bias = np.ones((X.shape[0],1))
    return np.hstack((bias, X))

#applies to sigmoid to X in element-wise operation
def sigmoid(X):
    return 1/(1+np.exp(-X))

##############################################################################################################
#Auxiliary functions for Regression
##############################################################################################################

#returns features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
def read_data_reg(filename):
    data = np.loadtxt(filename)
    Y = data[:,:2]
    X = np.concatenate((np.ones((data.shape[0], 1)), data[:,2:]), axis=1)
    return Y, X

####################################################################################################################################
#Auxiliary functions for classification
####################################################################################################################################

#returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)
def read_data_cls(split):
    feat = {}
    gt = {}
    for category in [('bottle', 1), ('horse', -1)]: 
        feat[category[0]] = np.loadtxt('data/'+category[0]+'_'+split+'.txt')
        feat[category[0]] = np.concatenate((np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1)
        gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
    X = np.concatenate((feat['bottle'], feat['horse']), axis=0)
    Y = np.concatenate((gt['bottle'], gt['horse']), axis=0)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns log likelihood loss
def get_loss(X, Y, w):
    Y = Y.copy()
    Y[Y==-1.0] = 0.0
    return - np.mean( (Y * np.log(sigmoid(X @ w)) + (1-Y) * np.log(1 - sigmoid(X @ w))), axis=0)

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns accuracy
def get_accuracy(X, Y, w):
    Y_pred = sigmoid(X @ w)
    Y_pred[Y_pred>0.5] = 1.0
    Y_pred[Y_pred<=0.5] = -1.0
    return np.mean(Y_pred==Y)