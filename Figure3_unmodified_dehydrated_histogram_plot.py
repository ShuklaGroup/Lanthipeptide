import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import FormatStrFormatter


#Define One dimensional histogram plot
def plot_hist(data1_, data2_, feat, weights1, weights2, title_, xlabel_, xticks_, yticks_):
    nSD, binsSD = np.histogram(data1_[:,feat]*10, bins=100, density=True, weights = weights1)
    nSD1, binsSD1 = np.histogram(data2_[:,feat]*10, bins=100, density=True, weights = weights2)
 
    #averageSD = [(binsSD[j]+binsSD[j+1])/2 for j in range(len(binsSD)-1)]

    fig, axs = plt.subplots(1,1,figsize=(10,7))
    axs.plot(binsSD[0:-1], nSD, linewidth=3, c='red')
    axs.plot(binsSD1[0:-1], nSD1, linewidth=3, c='blue')
    axs.legend(['Unmodified', 'Dehydrated'],loc='upper right',fontsize=20)
    
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
    plt.savefig('Prob_dist_plot_' + title_ + '.png',dpi=500)


if __name__ == "__main__":
    data = []
    weights = []
    Variants = ['WT','dehydrated_WT','Variant1','dehydrated_Variant1']
    for elem in Variants:
        totdist = []
        for file in sorted(glob.glob('Pcn3.3/' + elem + '/features_per_trajectory/*all-dist-per-traj.npy')):
            distI = np.load(file)
            totdist.append(distI)
        data.append(np.concatenate(totdist))
        MSM_weights = pickle.load(open('Pcn3.3/' + elem + '/optimized_MSM_weights.pkl','rb'))
        weights.append(np.concatenate(MSM_weights))
        
    #Plot distance of Thr11 and Cys14 of unmodified and dehydrated peptide
    plot_hist(data[0], data[1], 155, weights[0], weights[1],'WT_11_14', 'T11-C14 Distance ($\AA$)', np.arange(0,11,2), np.arange(0,0.81,0.1))
    plot_hist(data[2], data[3], 155, weights[2], weights[3],'Variant1_11_14', 'T11-C14 Distance ($\AA$)', np.arange(0,11,2), np.arange(0,1.21,0.2))

    #Plot distance of Thr11 and Cys21 of unmodified and dehydrated peptide
    plot_hist(data[0], data[1], 162, weights[0], weights[1], 'WT_11_21', 'T11-C21 Distance ($\AA$)', np.arange(0,31,5), np.arange(0,0.21,0.05))
    plot_hist(data[2], data[3], 162, weights[2], weights[3], 'Variant1_11_21', 'T11-C21 Distance ($\AA$)', np.arange(0,31,5), np.arange(0,0.41,0.05))

    #Plot distance of Cys14 and Thr18 of unmodified and dehydrated peptide
    plot_hist(data[0], data[1], 183, weights[0], weights[1], 'WT_14_18', 'C14-T18 Distance ($\AA$)', np.arange(0,15,2), np.arange(0,0.41,0.1))
    plot_hist(data[2], data[3], 183, weights[2], weights[3], 'Variant1_14_18', 'C14-T18 Distance ($\AA$)', np.arange(0,15,2), np.arange(0,1.51,0.3))

    #Plot distance of Thr18 and Cys21 of unmodified and dehydrated peptide
    plot_hist(data[0], data[1], 204, weights[0], weights[1], 'WT_18_21', 'T18-C21 Distance ($\AA$)', np.arange(0,11,2), np.arange(0,0.81,0.1))
    plot_hist(data[2], data[3], 204, weights[2], weights[3], 'Variant1_18_21', 'T18-C21 Distance ($\AA$)', np.arange(0,11,2), np.arange(0,1.21,0.2))
