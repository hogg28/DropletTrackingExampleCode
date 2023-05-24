# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:32:59 2023

@author: johnathan
"""

'''
This code is designed to track droplet positions over time using the position information
already collected in the DropletFinder script
'''

import numpy as np
import pandas as pd
import trackpy as tp

directory = 'SomeDirectory'
results = pd.read_csv(directory+'position_data.csv')

### The link_df function is putting in a lot of work here read through the tutorial and documentation 
### here: http://soft-matter.github.io/trackpy/v0.6.1/ for a better explanation then I could ever give

t = tp.link_df(results, search_range = 200, adaptive_stop=5, adaptive_step=0.99, memory=5)

def calcTraj (t, item):
    global data
    data = pd.DataFrame()
    sub = t[t.particle==item]
    dvx = np.diff(sub.x)
    dvy = np.diff(sub.y)
    for x, y, dx, dy, frame in zip(sub.x[:-1], sub.y[:-1], dvx, dvy, sub.frame[:-1],):
        data = data.append([{'dx': dx, 
                             'dy': dy, 
                             'x': x,
                             'y': y,
                             'frame': frame,
                             'particle': item,
                            }])
    return data

from joblib import Parallel, delayed
Data = Parallel(n_jobs = 12)(delayed(calcTraj)(t, item) for item in set(t.particle))


trajectories = pd.concat(Data)

trajectories.to_csv(directory+'trajectories_data.csv')



