# Carter McKinnon August 2022
# This document contains classes that will be used for processing
# the Labled Faces in the Wild Dataset
# Source: http://vis-www.cs.umass.edu/lfw/

import cv2
import numpy as np
import pandas as pd
from sklearn import utils
# this function will take a csv input formatted like this
# 
# name0    imageX   imageY   name1   imageZ   name2    imageA
# Carter   0012     00016
# 
# for invalid pairs, there will be another name column to the left of the image1 column
# 
# and will return a list of tuples of file paths to each image pair 
# acccording to the read me for the LFW dataset
# 
# According to the README, the path to the image is 
# lfw/name/name_xxxx.jpg

class SetBuilder():

    def __init__(self, img_col_names):
        self.img_col_names = img_col_names
    # function that pads the image refs to correspond to img paths
    def pad_img_refs(self, df):
        # changing all values to strings
        df_str = df.applymap(str)

        # padding the values
        for name in self.img_col_names:
            df_str[name] = df_str[name].str.pad(4, side='left', fillchar="0")

        return df_str
    
     # function that resizes the input to the specified input size for neural net
    def resize_image(self, img_path):
        img = cv2.imread(img_path)
        resize = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)

        return resize
    
    def get_img_pairs(self, img_paths, valid=True):
        set = []
        labels = []
        label = None

        if valid:
            label = 1
        else:
            label = 0

        for i in range(len(img_paths)):
            img0 = img_paths[i][0]
            img1 = img_paths[i][1]

            proc_img0 = self.resize_image(img0)
            proc_img1 = self.resize_image(img1)

            set.append((proc_img0, proc_img1))
            labels.append(label)

        return set, labels

    def get_path_arr(self, dataset):
        valid = []
        invalid = []
        df = pd.read_csv(dataset)
        df = self.pad_img_refs(df)

        # getting valid img pair file paths
        for i in range(len(df.iloc[:,0])):
            img0_path = "Data/lfw/{name}/{name}_{img_num}.jpg".format(name=df.iloc[:,0][i], img_num=df.iloc[:,1][i])       
            
            img1_path = "Data/lfw/{name}/{name}_{img_num}.jpg".format(name=df.iloc[:,0][i], img_num=df.iloc[:,2][i])
                
            pair = (img0_path, img1_path)
            valid.append(pair)

        # getting invalid pair file paths
        for i in range(len(df.iloc[:,3])):
            img0_path = "Data/lfw/{name}/{name}_{img_num}.jpg".format(name=df.iloc[:,3][i], img_num=df.iloc[:,4][i])       
            
            img1_path = "Data/lfw/{name}/{name}_{img_num}.jpg".format(name=df.iloc[:,5][i], img_num=df.iloc[:,6][i])
                
            pair = (img0_path, img1_path)
            invalid.append(pair)

        return valid, invalid

    def get_dataset(self, dataset):
        
        valid_paths, invalid_paths = self.get_path_arr(dataset)

        valid_set, valid_labels = self.get_img_pairs(valid_paths, valid=True)
        invalid_set, invalid_labels = self.get_img_pairs(invalid_paths, valid=False)
        
        pairs = valid_set + invalid_set
        labels = valid_labels + invalid_labels
        pairs = np.array(pairs)
        labels = np.array(labels)

        # shuffling dataset
        pairs,labels = utils.shuffle(pairs, labels)

        return pairs, labels


# This class will process the images in the dataset by identifying and isolating the face
# from the rest of the image
# 
# Notes--
"""
For some reason, haar cascade was not recognizing all faces
in the dataset, which broke some of code below, so until i find a way to fix that without
entirely removing some parts of the datasets, i am not going to use this class
while generating the initial test, train, and validation sets for my model
Initially, I was planning on training the network with a grayscaled image cropped around the 
face of the person, but that obv doesnt work if haar cascade cant recognize the face in the images.
"""

class ImgProcessor:

    # helper function that uses haar cascade to recognize face in image
    def get_face_dims(self, img):

        # building haar cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # detecting points around face
        face_dims = face_cascade.detectMultiScale(
            img,
            scaleFactor=1.5,
            minNeighbors=5,
        )

        return face_dims

    # helper function that processes the points returned by get_face_dims
    def get_dims(self, dim_list):
        dim_list = list(dim_list)
        x = dim_list[0]
        y = dim_list[1]
        w = dim_list[2]
        h = dim_list[3]
        
        return x,y,w,h

    # helper function that will return the cropped image
    def crop_and_resize(self, img, point_list):

        # getting dimensions
        x,y,w,h = self.get_dims(point_list[0])

        # cropping the image
        crop_img = img[y:y+h, x:x+w]

        # resizing the image to a 64x64 for input to neural net
        final_img = cv2.resize(crop_img, (105, 105), interpolation = cv2.INTER_AREA)

        return final_img

    # method that will process the image
    def process_image(self, img_path):
        img = cv2.imread(img_path, 0)
        dims = self.get_face_dims(img)
        processed_img = self.crop_and_resize(img, dims)

        return processed_img

