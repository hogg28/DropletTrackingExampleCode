# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:45:40 2023

@author: johna
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from skimage.filters import unsharp_mask
import pandas as pd
import pims
import trackpy as tp

#%%

'''
The below code will make a overlay of the trajectory information with the raw images
This is very slow and can get very busy if there are several droplets being tracked
'''

mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

@pims.pipeline
def preprocess_img(frame):
    frame = frame[:,800:1100]
    frame = unsharp_mask(frame, radius = 2, amount = 5)
    frame *= 255.0/frame.max()
    return frame


directory = 'D:/Johnathan/GlassSpreading/2SDS/Chamber10/30deg/2SDS_30deg_33percent_large_66percent_small_Jun-23-2022_1/'
prefix = 'img*.tiff'


frames = preprocess_img(pims.ImageSequence(os.path.join(directory+prefix)))

trajectories = pd.read_csv(directory+ 'trajectories_data.csv')

#%%

def createFolder(directory):
    if not os.path.exists(directory+'/TrackingByFrame_4'):
        os.mkdir(directory+'/TrackingByFrame_4')

def plotTraj (traj, k, directory, frames):
    plt.ion()

    trajectories_fig = tp.plot_traj(traj[traj.frame<=k], colorby='particle', cmap = mpl.cm.winter, superimpose=frames[k+20])
    plt.ylim(0,1020)
    plt.xlim(0,300)
    trajectories_fig.figure.savefig(directory+'/TrackingByFrame_4/trajectories' + str(k))
    return

createFolder(directory)

plt.figure()
#Plots an overlay for the first 1000 frames
for k in np.arange(1,400,1):
    plotTraj(trajectories, k, directory, frames)