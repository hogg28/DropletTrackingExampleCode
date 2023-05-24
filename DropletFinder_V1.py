# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:05:41 2023

@author: johnathan
"""

'''
This code is designed to find the positions of droplets in a stack of images
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from skimage.filters import unsharp_mask
from skimage.feature import match_template
from skimage.feature import peak_local_max
import pandas as pd
import pims

#%%

'''
Define the required functions for droplet finding
'''

#Make images grayscale for easier viewing
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')


# Function to find the positions of the droplets in a single image
# Input a full image, a template of the droplet, and the threshold for matching
# Returns the cross correlated image (match) and the location of the peaks corresponding to droplet locations (peaks)
# Can be used on a single image to test that the template and threshold will work

def MatchTemplate(img, template, thresh):
    match = match_template(img, template, pad_input = True)
    ij = np.unravel_index(np.argmax(match), match.shape)
    x,y  = ij[::-1]
    peaks = peak_local_max(match,min_distance=10,threshold_rel=thresh) # find our peaks
    return match, peaks


#This function does the same method as the MatchTemplate function but will store the position information in a dataframe

def labelimg(num, img):
    global features
    features = pd.DataFrame()
    match, peaks = MatchTemplate(img, template, 0.7)

    for i in range(len(peaks)):
        features = features.append([{'y': peaks[:,0][i],
                                     'x': peaks[:,1][i],
                                     'frame': num,
                                     },])
    return features


#%%
'''
Read in images and preform required preprocessing
'''

#Preprocessing function for images
#Input is an image, Output is a cropped image with sharpened edges
#Shapening is optional but I find it helps with the cross correlation

@pims.pipeline
def preprocess_img(frame):
    frame = frame[:,100:800]
    frame = unsharp_mask(frame, radius = 2, amount = 5)
    frame *= 255.0/frame.max()
    return frame



#Read in a time series of frames from desired directory
#Must edit this every time
directory = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/2DSpreading/2SDS_30deg_6hr_2secframe_10072020_1/'
prefix = '*.tif'
frames = preprocess_img(pims.ImageSequence(os.path.join(directory+prefix)))

#Take one image that has a clear view of a single droplet and crop the image to be a single droplet
template_img = frames[200]

### Crop the templates ###
template = template_img[504:531, 148:175]

#Plot the template before moving forward to confirm it is the correct image#

plt.imshow(template)
plt.show()


#%%
'''
Test matching on example image
'''

img_example = frames[4000]

match, peaks = MatchTemplate(img_example, template, 0.7)


fig, ([ax1, ax2]) = plt.subplots(ncols=2, figsize=(8, 3))

ax1.imshow(template)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(img_example)
ax2.set_axis_off()
ax2.set_title('image')
ax2.plot(peaks[:,1], peaks[:,0], 'x', markeredgecolor='blue', markerfacecolor='none', markersize=1)

plt.show()

#%%
'''
Perform template matching on all frames and store positions in a csv
'''

results = np.zeros((10**7, 3))

for idx, frame in enumerate(frames[20:400]):
    current_position = labelimg(idx, frame, template)
    rows_to_add_data = np.where(~results.any(axis=1))[0][0:len(current_position)]
    results[rows_to_add_data] = current_position
    
rows_to_keep = np.where(results.all(axis=1))[0]
results = results[rows_to_keep]
results = pd.DataFrame(results, columns=['y', 'x', 'frame'])

results.to_csv(directory+'position_data.csv')

