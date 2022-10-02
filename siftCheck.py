# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:57:26 2022

@author: General
"""

# PyTorch Library
import torch
# PyTorch Neural Network
import cv2
import os

# Use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# By right the app will ensure images are of same size and format before its pushed to the checker
IMAGE_SIZE = 32
def normalize(img):
	img = cv2.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
	return(img)


def createDatabase(Path):
	dataset =[]
	sift = cv2.SIFT_create()
	for filename in os.listdir(Path):
		image = cv2.imread(f'{Path}/{filename}', 0)
		image = normalize(image)
		kpt, des = sift.detectAndCompute(image,None)
		dataset.append([f'{Path}/{filename}', kpt, des])

	return dataset

def validImage(data, path):
	img1 = cv2.imread(path, 0)      # queryImage
	sift = cv2.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, cur_des = sift.detectAndCompute(img1,None)

	# Define params for flann index
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	# Create a flann index class
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	for filepath, kpt, des in data:
		i = 0
		image_exists = False
		# Matches descriptors based
		if(len(kp1)>=2 and len(kpt)>=2):
			matches = flann.knnMatch(cur_des, des, k=2)
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				i+= 1
			if i == 50:
				image_exists = True
				return image_exists

	return image_exists

