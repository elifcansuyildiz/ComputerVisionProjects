"""
Elif Cansu YILDIZ 05/2021
"""
import cv2
import numpy as np

class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, class_colors=[True, True, True, True], patch_size=16, num_samples=4000):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = np.array(class_colors)
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.max_num_class = 4

    # Function for sampling patches for each class
    # returns extracted patches with labels
    def extractpatches(self):
        
        imgs = []
        segmaps = []
        
        for img_path, segmap_path in zip(self.train_images_list, self.gt_segmentation_maps_list):
            imgs.append(self.read_rgb_image(img_path))
            segmaps.append(self.read_grayscale_image(segmap_path))
          
        count=0
        num_samples_per_class = self.class_colors * np.ones(self.max_num_class) * self.num_samples / np.sum(self.class_colors)
        while np.sum(num_samples_per_class) < self.num_samples:
            rand_num = np.random.randint(self.max_num_class)
            if class_colors[rand_num]:
                num_samples_per_class[rand_num] += 1
            
        patches = []
        labels = []
        
        while count < self.num_samples: 
            img_idx = np.random.randint(len(imgs))
            patch, label = self.sample_from_image(imgs[img_idx], segmaps[img_idx], self.patch_size)
            if num_samples_per_class[label]>0:
                patches.append(patch)
                labels.append(label)
                num_samples_per_class[label] -=1
                count+=1
                
        return np.array(patches), np.array(labels)
                   
    def read_rgb_image(self, path):
        img = cv2.imread(cv2.samples.findFile(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def read_grayscale_image(self, path):
        img = cv2.imread(cv2.samples.findFile(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    def sample_from_image(self, img, segmap, patch_size):
        
        x = np.random.randint(0, img.shape[1]-patch_size)
        y = np.random.randint(0, img.shape[0]-patch_size)
        x_center = x + patch_size//2
        y_center = y + patch_size//2 

        patch = img[y:y+patch_size, x:x+patch_size, :]
        label = segmap[y_center, x_center]
        
        return patch, label