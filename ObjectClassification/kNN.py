"""
Elif Cansu YILDIZ 07/2021
"""
import cv2
import os
import numpy as np
from typing import List

def distance_calc(feature_bows):
    distances = []
    for i in range(len(feature_bows)):
        distances.append(np.linalg.norm(feature_bows - feature_bows[i], axis=1))
    return np.array(distances)


def knn_classification(response_hists, data: List, img_num:int, knn=1):
    img = data[img_num]
    voc_file_path = os.path.join("", 'vocabulary.yaml')
    
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.FlannBasedMatcher()
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, matcher)
    c = cv2.FileStorage(voc_file_path, cv2.FileStorage_READ)
    dictionary = c.getNode('dictionary').mat()
    bow_extractor.setVocabulary( dictionary )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, dsc = sift.detectAndCompute(gray, None)
    feature_bow = bow_extractor.compute(img, kp)
    
    diff = response_hists - feature_bow
    dist = np.linalg.norm(diff, axis=1)

    if knn == 1:
        idxs = [np.argmin(dist)]
        class_id = idxs[0]//4
    else:
        mask = np.argsort(dist)
        idxs = mask[:knn]
        class_id = idxs // 4
        
        unique, unique_counts = np.unique(class_id, return_counts=True)
        if len(unique) == 2:
            max_repeated_idx = np.argmax(unique_counts)
            class_id = unique[max_repeated_idx]
        else:
            class_id = class_id[0]
        
    return class_id

