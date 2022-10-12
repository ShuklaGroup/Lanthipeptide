import numpy as np
import glob
import mdtraj as md
import math
import matplotlib.pyplot as plt
import mdtraj as md
import os
from numpy import linalg as LA
import pickle
import pyemma
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import random

cluster_number = 150
all_eigen_ref = []

for j in range(200): 
     msm = pickle.load(open("/home/xmi4/RippsProject/procA33_WT/analysis/bootstrapping_ticdim/bt_80_" + str(j) +"_msm.pkl",'rb')) 
     eigen_ref = msm.eigenvectors_left()[0] 
     if len(eigen_ref) == cluster_number: 
         all_eigen_ref.append(eigen_ref) 
  
     else: 
         missing_index = [] 
         for i in range(msm.active_set[0], msm.active_set[-1]+1): 
             if i not in msm.active_set: 
                 missing_index.append(i) 
         for k in range(len(missing_index)): 
             missing_idx = missing_index[k] 
             eigen_ref = np.concatenate((eigen_ref[:missing_idx], [0], eigen_ref[missing_idx:])) 
         all_eigen_ref.append(eigen_ref)
        

all_eigen_ref = np.vstack(all_eigen_ref)
all_mean_array = np.mean(all_eigen_ref, axis=0)
all_sd_array = np.std(all_eigen_ref, axis=0)
index = list(range(150))
fig, axs = plt.subplots(1,1,figsize=(10,7))
plt.errorbar(index, all_mean_array, yerr = all_sd_array, fmt = "o", markersize = 5)
plt.xlim(0,150)
plt.ylim(0, 0.5)
plt.xlabel('State index',fontsize=30)
plt.ylabel('MSM Population', fontsize=30)
plt.xticks([0, 30, 60, 90, 120, 150],fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('procA3.3WT_MSM_population_with_bt.png', transparent = True, dpi = 500)
