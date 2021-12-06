"""
Elif Cansu YILDIZ 07/2021
"""
import cv2
import os
from BoWModel import bag_of_words
from kNN import knn_classification

def load_dataset():
    #Load Dataset
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in range(265):
        filename = "images/" + str(i).zfill(3) + ".jpg"
        if i % 5 == 0:
            test_data.append(cv2.imread(filename))
            test_label.append(i//5)
        else:
            train_data.append(cv2.imread(filename))
            train_label.append(i//5)

    return train_data, train_label, test_data, test_label


if __name__ == "__main__":

    train_data, train_label, test_data, test_label = load_dataset()

    print("number of training data: ", len(train_data))
    print("number of test data: ", len(test_data))

    img_path = "images"
    voc_file_path = "vocabulary.yaml"
    dict_size = 54

    if not os.path.isfile(voc_file_path):
        #generate vocabulary(dictionary)
        dictionary = bag_of_words(image_path=img_path, vocabulary_path="", data=train_data, img_num=len(train_data), dictionarySize=dict_size)

    #Extract the histogram that corresponds to the BoW dictionary for each image
    response_hists = bag_of_words(image_path=img_path, vocabulary_path="", data=train_data, img_num=len(train_data), dictionarySize=dict_size)

    #Classify the test images with kNN Classifier
    #Classification is done using SIFT BoW Model and a k-Nearest-Neighbor classifier with Euclidean Distance metric
    print("\nTesting with KNN=1")
    stats = {"true": 0, "false": 0}
    for img_num in range(len(test_data)):
        prediction = knn_classification(response_hists, test_data, img_num, knn=1)
        
        if prediction == img_num:
            stats["true"] += 1
        else:
            stats["false"] += 1
            
    print(stats)
    print("Accuracy:", stats["true"] / len(test_data))

    print("\nTesting with KNN=3")
    stats = {"true": 0, "false": 0}
    for img_num in range(len(test_data)):
        prediction = knn_classification(response_hists, test_data, img_num, knn=3)
        
        if prediction == img_num:
            stats["true"] += 1
        else:
            stats["false"] += 1
        
    print(stats)
    print("Accuracy:", stats["true"] / len(test_data))