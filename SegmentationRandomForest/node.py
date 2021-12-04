"""
Elif Cansu YILDIZ 05/2021
"""
import numpy as np

class Node():
    def __init__(self, patches, labels, depth=0):
        self.labels = labels
        self.patches = patches
        self.depth = depth
        self.type = 'None'
        self.leftchild = -1
        self.rightchild = -1
        self.feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        self.probabilities = []

    # create a new split node
    def create_SplitNode(self, leftchild, rightchild, feature):
        self.type = "split"
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.feature = feature
        print("split-node:", feature, self.depth)

    # create a new leaf node
    def create_leafNode(self, labels, classes): 
        self.type = "leaf"
        hist = np.bincount(labels, minlength = 4)
        self.probabilities = hist / np.sum(hist)
        print("leaf-node:", self.probabilities)
        return self.probabilities