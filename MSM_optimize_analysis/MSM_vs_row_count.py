import numpy as np
import glob
import mdtraj as md
import os
import pickle
import pyemma
import seaborn as sns
import matplotlib.pyplot as plt


cluster_number = 150
lag_time = 250
tic_dim = 10
msm_lag_time = 200
file = "./MSM/MSM-"+'procA3.3WT-cluster_kmeans_'+ 'C_'+str(cluster_number)+'_lt_' + str(lag_time)+ '_ticdim_' + str(tic_dim) + ".pkl"

dtrajs_ref = pickle.load(open(file,'rb'))
dtrajs_com = np.concatenate(dtrajs_ref)

msm_ref = pickle.load(open("./MSM_ticdim/MSM-"+'procA3.3WT-MSM_'+ 'C_'+str(cluster_number)+'_lt_' + str(lag_time)+'_ticdim_'+ str(tic_dim) +".pkl",'rb'))
eigen_ref = msm_ref.eigenvectors_left()[0]

total_pop = 0
for i,index in enumerate(msm_ref.active_set):
    total_pop = total_pop + np.count_nonzero(dtrajs_com==index)

unweighted_popu = np.empty(len(msm_ref.active_set))
weighted_popu = np.empty(len(msm_ref.active_set))
for i,index in enumerate(msm_ref.active_set):
    unweighted_popu[i] = np.count_nonzero(dtrajs_com==index)/total_pop
    weighted_popu[i] = eigen_ref[i]

fig,axs = plt.subplots(1,1,figsize=(10,7),constrained_layout=True)
plt.plot(np.log10(unweighted_popu),np.log10(weighted_popu),'o',color='green')
plt.plot([-4,0],[-4,0])
axs.set_xlim(-4,0)
axs.set_xticks(range(int(-4),int(0)+1,1))
axs.set_xticklabels(range(int(-4),int(0)+1,1))
axs.set_ylim(-4,0)
axs.set_yticks(range(int(-4),int(0)+1,1))
axs.set_yticklabels(range(int(-4),int(0)+1,1))

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Unweighted Population',**hfont,fontsize=30)
plt.ylabel('MSM Population', **hfont,fontsize=30)
plt.savefig('procA3.3WT_raw_msm.png',transparent=True,dpi=500)

