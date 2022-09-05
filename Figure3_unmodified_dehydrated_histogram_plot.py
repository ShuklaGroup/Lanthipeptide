#The script is used for plotting Figure3

import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import FormatStrFormatter

#Load data and MSM weights
MT_totdist = []
for file in sorted(glob.glob('./procA33_MT1_2/analysis/*all-dist-per-traj.npy')):
    distI = np.load(file)
    MT_totdist.append(distI)
MT_totdist = np.concatenate(MT_totdist)
MT_weights = pickle.load(open('./procA33_MT1_2/analysis/MSM-procA3.3MT1_cluster_150_ticdim_8-weights.pkl','rb'))
MT_weights = np.concatenate(MT_weights)

dehy_MT_totdist = []
for file in sorted(glob.glob('./Mut1-dehydrated-new/procAMT1-3Thr3Dhb/plot/*all-dist-per-traj.npy')):
    distI = np.load(file)
    dehy_MT_totdist.append(distI)
print(len(dehy_MT_totdist))
dehy_MT_totdist = np.concatenate(dehy_MT_totdist)
dehy_MT_weights = pickle.load(open('./Mut1-dehydrated-new/procAMT1-3Thr3Dhb/plot/MSM-procAMT1-3Thr3Dhb_cluster_80_ticdim_10-weights.pkl','rb'))
print(len(dehy_MT_weights))
dehy_MT_weights = np.concatenate(dehy_MT_weights)

#Define One dimensional histogram plot
#Six paramters: the index of feature to be plotted; the number of bins; plot title; xlabels; xticks; ysticks
def plot_hist(feat, bins_, title_, xlabel_, xticks_, yticks_):
    nSD, binsSD = np.histogram(MT_totdist[:,feat]*10, bins=bins_, density=True, weights = MT_weights)
    nSD1, binsSD1 = np.histogram(dehy_MT_totdist[:,feat]*10, bins=bins_, density=True, weights = dehy_MT_weights)
 
    #averageSD = [(binsSD[j]+binsSD[j+1])/2 for j in range(len(binsSD)-1)]

    fig, axs = plt.subplots(1,1,figsize=(10,7))
    axs.plot(binsSD[0:-1], nSD, linewidth=3, c='red')
    axs.plot(binsSD1[0:-1], nSD1, linewidth=3, c='blue')
    axs.legend(['Mutant1-Unmodified', 'Mutant1-Dehydrated'],loc='upper right',fontsize=20)
    
    axs.set_xticks(xticks_)
    axs.set_xticklabels(xticks_)
    axs.set_yticks(yticks_)
    axs.set_yticklabels(yticks_)
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs.tick_params(width=3,length=5, labelsize=20)
    
    plt.xlabel(xlabel_, fontsize=30)
    plt.ylabel('Probability Density', fontsize=28)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    
    plt.tight_layout()
    plt.savefig('MT_Prob_dist_plot_' + title_ + '.png',dpi=500)

#Plot distance of Thr11 and Cys14 of unmodified and dehydrated peptide
plot_hist(155, 100,'11_14', 'T11-C14 Distance ($\AA$)', np.arange(0,11,2), np.arange(0,1.21,0.2))

#Plot distance of Thr11 and Cys21 of unmodified and dehydrated peptide
plot_hist(162, 100, '11_21', 'T11-C21 Distance ($\AA$)', np.arange(0,31,5), np.arange(0,0.41,0.05))

#Plot distance of Cys14 and Thr18 of unmodified and dehydrated peptide
plot_hist(183, 100, '14_18', 'C14-T18 Distance ($\AA$)', np.arange(0,15,2), np.arange(0,1.51,0.3))

#Plot distance of Thr18 and Cys21 of unmodified and dehydrated peptide
plot_hist(204, 100, '18_21', 'T18-C21 Distance ($\AA$)', np.arange(0,11,2), np.arange(0,1.21,0.2))

