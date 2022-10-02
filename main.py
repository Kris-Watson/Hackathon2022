# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 02:24:26 2022

@author: General
"""

import labelCheck as label
import siftCheck as dupe

# 
IS_CHAIR = 0
IS_SWEIVEL = 1
IS_BED = 2
IS_SOFA = 3

# We use a test image called test.jpg, with the user giving the label of IS_CHAIR
myPath = "test.jpg"
myLabel = IS_CHAIR

# Previously uploaded data in dataset
dataPath = "prev_images"


# Check if label is accurate and if image quality is good
validLabel = label.checkImage(myPath, myLabel)

# Create a dummy dataset of sift descriptors using images located at "dataPath"
data = dupe.createDatabase(dataPath)

# Check if image is very similar to previously uploaded images
dupeFlag = dupe.validImage(data, myPath)

# If labelled accurately, image quality is good and is unique, accept image to dataset
if validLabel and not dupeFlag:
	print("your image is valid and unique")