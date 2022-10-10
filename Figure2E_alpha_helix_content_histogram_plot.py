#The script is used for plotting Figure 2D
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pickle

#Load data and MSM weights
WT = pickle.load(open('procA3.3WT-four-dist-sub-helical.pkl','rb'))
WT_weights = pickle.load(open('MSM-procA3.3WT_cluster_150_ticdim_10-weights.pkl','rb'))
WT = np.concatenate(WT)
WT_weights = np.concatenate(WT_weights)

MT1 = pickle.load(open('procA3.3MT1_2-four-dist-sub-helical.pkl','rb'))
MT1_weights = pickle.load(open('MSM-procA3.3MT1_cluster_150_ticdim_8-weights.pkl','rb'))
MT1 = np.concatenate(MT1)
MT1_weights = np.concatenate(MT1_weights)

MT2 = pickle.load(open('procA3.3MT2_16-four-dist-sub-helical.pkl','rb'))
MT2_weights = pickle.load(open('MSM-procA3.3MT2_cluster_100_ticdim_6-weights.pkl','rb'))
MT2 = np.concatenate(MT2)
MT2_weights = np.concatenate(MT2_weights)

#Define One dimensional histogram plot
#Two paramter: the index of feature to be plotted; the number of bins
def plot_hist(feat, bins_):
    nSD, binsSD = np.histogram(WT[:,feat], bins=bins_, density=True,weights = WT_weights)
    nSD1, binsSD1 = np.histogram(MT1[:,feat], bins=bins_, density=True, weights = MT1_weights)
    nSD2, binsSD2 = np.histogram(MT2[:,feat], bins=bins_, density=True, weights = MT2_weights)
    #averageSD = [(binsSD[j]+binsSD[j+1])/2 for j in range(len(binsSD)-1)]

    fig, axs = plt.subplots(1,1,figsize=(10,7))
    axs.plot(binsSD[0:-1], nSD, linewidth=3, c='red')
    axs.plot(binsSD1[0:-1], nSD1, linewidth=3, c='blue')
    axs.plot(binsSD2[0:-1], nSD2, linewidth=3, c='orange')
    axs.legend(['Wild Type', 'Mutant1', 'Mutant2'],loc='upper right', fontsize = 18)

    axs.set_xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    axs.set_xticklabels([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    axs.set_yticks([0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00])
    axs.set_yticklabels([0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00])
    axs.tick_params(width=3,length=5, labelsize=18)
    
    plt.xlabel('Helical Content', fontsize=28)
    plt.ylabel('Probability Density', fontsize=28)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.tight_layout()
    plt.savefig('ProcA3.3_prob_helical_content.png',dpi=500)

#Plot helical content
plot_hist(4, 100)   #in the feature matrix, 4th column is helical content; bins= 100

