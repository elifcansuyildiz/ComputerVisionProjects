"""
Elif Cansu YILDIZ 07/2021
"""
import cv2
import numpy as np
import os
from typing import List

def bag_of_words(image_path: str, vocabulary_path: str, data: List, img_num: int, dictionarySize: int) -> List:

    voc_file_path = os.path.join(vocabulary_path, 'vocabulary.yaml')

    if os.path.isfile(voc_file_path):
        print("Load vocabulary from %s" % voc_file_path)
        sift = cv2.xfeatures2d.SIFT_create()        
        matcher = cv2.FlannBasedMatcher()
        bow_extractor = cv2.BOWImgDescriptorExtractor(sift, matcher)
        c = cv2.FileStorage(voc_file_path, cv2.FileStorage_READ)
        dictionary = c.getNode('dictionary').mat()
        bow_extractor.setVocabulary( dictionary )

        response_hists = []
        for img in data:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, dsc = sift.detectAndCompute(gray, None)
            feature_bow = bow_extractor.compute(img, kp)
            response_hists.append(feature_bow.squeeze())

        response_hists = np.stack(response_hists)
        print("response_hists.shape:", response_hists.shape) 
        return response_hists
        
    else:
        print("Generating vocabulary")

        bow_trainer = cv2.BOWKMeansTrainer(dictionarySize)
        sift = cv2.xfeatures2d.SIFT_create()

        kps=[]
        for img in data:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, dsc = sift.detectAndCompute(gray, None)
            bow_trainer.add(dsc)
            kps.append(kp)

        dictionary = bow_trainer.cluster()
        print("dictionary.shape:", dictionary.shape)
        print("DONE")

        print("Writing vocabulary to %s" % voc_file_path )
        c = cv2.FileStorage(voc_file_path, cv2.FileStorage_WRITE)
        c.write('dictionary', dictionary)
        
        return dictionary