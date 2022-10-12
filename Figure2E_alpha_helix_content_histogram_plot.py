#The script is used for plotting Figure 2D
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pickle

#Load data and MSM weights
WT = pickle.load(open('Pcn3.3/WT/procA3.3WT-four-dist-sub-helical.pkl','rb'))
WT_weights = pickle.load(open('Pcn3.3/WT/optimized_MSM_weights.pkl','rb'))
WT = np.concatenate(WT)
WT_weights = np.concatenate(WT_weights)

Variant1 = pickle.load(open('Pcn3.3/Variant1/procA3.3MT1_2-four-dist-sub-helical.pkl','rb'))
Variant1_weights = pickle.load(open('Pcn3.3/Variant1/optimized_MSM_weights.pkl','rb'))
Variant1 = np.concatenate(Variant1)
Variant1_weights = np.concatenate(Variant1_weights)

Variant2 = pickle.load(open('Pcn3.3/Variant2/procA3.3MT2_16-four-dist-sub-helical.pkl','rb'))
Variant2_weights = pickle.load(open('Pcn3.3/Variant2/optimized_MSM_weights.pkl','rb'))
Variant2 = np.concatenate(Variant2)
Variant2_weights = np.concatenate(Variant2_weights)

#Define One dimensional histogram plot
#Two paramter: the index of feature to be plotted; the number of bins
def plot_hist(feat, bins_):
    nSD, binsSD = np.histogram(WT[:,feat], bins=bins_, density=True,weights = WT_weights)
    nSD1, binsSD1 = np.histogram(Variant1[:,feat], bins=bins_, density=True, weights = Variant1_weights)
    nSD2, binsSD2 = np.histogram(Variant2[:,feat], bins=bins_, density=True, weights = Variant2_weights)
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
    plt.savefig('PcnA3.3_prob_helical_content.png',dpi=500)

#Plot helical content
plot_hist(4, 100)   #in the feature matrix, 4th column is helical content; bins= 100

