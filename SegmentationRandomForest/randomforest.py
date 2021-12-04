"""
Elif Cansu YILDIZ 05/2021
"""

import numpy as np

from tree import DecisionTree

class Forest():
    def __init__(self, patches=[], labels=[], tree_param=[], n_trees=1):

        self.patches, self.labels = patches, labels
        self.tree_param = tree_param
        self.ntrees = n_trees
        self.trees = []
        for i in range(n_trees):
            self.trees.append(DecisionTree(self.patches, self.labels, self.tree_param)) # where is the bagging?!

    # Function to create ensemble of trees
    # provide your implementation
    # Should return a trained forest with n_trees
    def create_forest(self):
        for i, t in enumerate(self.trees):
            print("========== training tree-{} ==========".format(i))
            t.train()

    # Function to apply the trained Random Forest on a test image
    # provide your implementation
    # should return class for every pixel in the test image
    def test(self, I):
        all_probabilities = []
        for i, t in enumerate(self.trees):
            print("========== testing tree-{} ==========".format(i))
            pred_img, pred_prob = t.predict(I)
            all_probabilities.append(pred_prob)
        all_probabilities = np.array(all_probabilities)
        print("all:", all_probabilities.shape)
        avg_img = np.mean(all_probabilities, axis=0)
        avg_img = np.argmax(avg_img, axis=2)
        return avg_img