"""
Elif Cansu YILDIZ 05/2021
"""

import numpy as np
from node import Node

class DecisionTree():
    def __init__(self, patches, labels, tree_param):

        self.patches, self.labels = patches, labels
        self.depth = tree_param['depth']
        self.pixel_locations = tree_param['pixel_locations']
        self.random_color_values = tree_param['random_color_values']
        self.no_of_thresholds = tree_param['no_of_thresholds']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.nodes = [] # ???
        self.root_node = None

    # Function to train the tree
    # should return a trained tree with provided tree param
    def train(self):
        
        root_node = Node(self.patches.copy(), self.labels.copy(), depth=0)
        self.nodes.append(root_node)
        
        while len(self.nodes) > 0:
            # get the current node & etc
            current_node = self.nodes.pop()
            current_patches = current_node.patches
            current_labels = current_node.labels
            current_depth = current_node.depth
            
            if current_depth == self.depth or len(np.unique(current_labels)) == 1:
                # make this node leaf
                current_node.create_leafNode(current_labels, None)
            else:
                mask_left, mask_right, c, i, j, th = self.best_split(current_patches, current_labels)
                if mask_left is None:
                    # make this node leaf
                    current_node.create_leafNode(current_labels, None)
                else:
                    # make this node split
                    feature = {'color': c, 'pixel_location': [i, j], 'th': th}
                    leftchild = Node(current_patches[mask_left], current_labels[mask_left], current_depth+1)
                    rightchild = Node(current_patches[mask_right], current_labels[mask_right], current_depth+1)
                    current_node.create_SplitNode(leftchild, rightchild, feature)
                    self.nodes.append(leftchild)
                    self.nodes.append(rightchild)

            current_node.patches = None # free memory
            current_node.labels = None  # free memory

        # output tree will not contain any data samples
        self.root_node = root_node
        return root_node

    # Function to predict probabilities for single image
    # provide your implementation
    # should return predicted class for every pixel in the test image
    def predict(self, I):
        
        pred_labels = np.zeros((I.shape[0], I.shape[1]))
        pred_probs = np.zeros((I.shape[0], I.shape[1], self.classes))
            
        for x in range(I.shape[1]-16):
            for y in range(I.shape[0]-16):
                current_node = self.root_node
                
                while current_node.type != "leaf":
                    c = current_node.feature["color"]
                    i, j = current_node.feature["pixel_location"]
                    th = current_node.feature["th"]
                    
                    if I[y+i, x+j,c]>th:
                        current_node = current_node.leftchild
                    else:
                        current_node = current_node.rightchild

                pred_labels[y+i, x+j] = np.argmax(current_node.probabilities)
                pred_probs[y+i, x+j] = current_node.probabilities

        return np.array(pred_labels), np.array(pred_probs)

    # Function to get feature response for a random color and pixel location
    # provide your implementation
    # should return feature response for all input patches
    def getFeatureResponse(self, patches, feature):
        c, i, j = feature
        responses = patches[:, i, j, c]
        return responses

    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        mask = responses>threshold
        return mask, ~mask

    # Function to get a random pixel location
    # provide your implementation
    # should return a random location inside the patch
    def generate_random_pixel_location(self):
        return np.random.randint(self.patches.shape[1]), np.random.randint(self.patches.shape[1])

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, labels):
        hist = np.bincount(labels, minlength = self.classes)
        probabilities = hist / np.sum(hist)
        log_probabilities = np.zeros(self.classes)
        log_probabilities[probabilities>0] = np.log(probabilities[probabilities>0])
        return - np.sum(probabilities * log_probabilities)

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        return EntropyAll - ((Nleft/Nall) * Entropyleft + (Nright/Nall) * Entropyright)

    # Function to get the best split for given patches with labels
    # provide your implementation
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        
        #max_params= (info_gain, mask_left, mask_right, color, pixel_loc_i, pixel_loc_j, threshold)
        max_params = (None, None, None, None, None, None)
        max_info_gain = -9999
        
        for c in range(self.random_color_values):      
            for t in range(self.no_of_thresholds):               
                for l in range(self.pixel_locations):
                    
                    rand_c = np.random.randint(3)
                    rand_t = np.random.randint(256)
                    rand_i, rand_j = self.generate_random_pixel_location()
                    
                    feature = (rand_c, rand_i, rand_j)
                    responses = self.getFeatureResponse(patches, feature)
                    
                    mask_left, mask_right = self.getsplit(responses, rand_t)  
                    
                    Nall = len(labels)
                    Nleft = np.sum(mask_left)
                    Nright = np.sum(mask_right)
                    
                    if Nleft>=self.minimum_patches_at_leaf and Nright>=self.minimum_patches_at_leaf:
                        
                        entropy_left = self.compute_entropy(labels[mask_left])
                        entropy_right = self.compute_entropy(labels[mask_right])
                        entropy_all = self.compute_entropy(labels)
                        
                        info_gain = self.get_information_gain(entropy_left, entropy_right, entropy_all, 
                                                              Nall, Nleft, Nright)
                        if info_gain > max_info_gain:
                            max_params = (mask_left, mask_right, rand_c, rand_i, rand_j, rand_t) 
                        
        return max_params   