import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import sys

threshold = 0.45

variants = ['WT','Variant1','Variant2']
for elem in variants:
    df = pd.DataFrame(columns = ['Thr11-Cys14','Thr11-Cys21','Cys14-Thr18','Thr18-Cys21'])
    #load 200 bootstrapping samples and their corresponding MSM
    for j in range(200):
        files = pickle.load(open('Pcn3.3/' + elem +"/bootstrapping/bt_80_" + str(j) +"_files.pkl",'rb')) #the pickle file includes the randomly selected trajctories (80% of all trajectoires)
        msm = pickle.load(open('Pcn3.3/' + elem + "/bootstrapping/bt_80_" + str(j) +"_msm.pkl",'rb')) #the MSM for corresponding subset of all trajectories
        feature = []
        for file in files:
            feature.extend(np.load('Pcn3.3/' + elem + "/features_per_trajectory/" +file)[:,[155,162,183,204]]) #load features of each trajectory in the subset, [155, 162,183,204] is the index of feature 'Thr11-Cys14','Thr11-Cys21','Thr18-Cys14','Thr18-Cy21'

        feature = np.vstack(feature)
        boolean_list1 = feature[:,0] < threshold #if feature (residue pair distance) < thresold, we assign True; otherwise, False
        feat1 = list(map(int, boolean_list1)) #convert True to 1, and False to 0

        boolean_list2 = feature[:,1] < threshold
        feat2 = list(map(int, boolean_list2))

        boolean_list3 = feature[:,2] < threshold
        feat3 = list(map(int, boolean_list3))

        boolean_list4 = feature[:,3] < threshold
        feat4 = list(map(int, boolean_list4))

        weights = np.concatenate(msm.trajectory_weights())
        feat1_w =  np.dot(feat1,weights) #for distance betweehn Thr11 and Cys14, dot product of (ring form(1) or not(0)) and (weights) = probability of ring formation
        feat2_w =  np.dot(feat2,weights) #probability of Thr11-Cy21 ring formation
        feat3_w =  np.dot(feat3,weights) #probability of Thr18-Cys14 ring formation
        feat4_w =  np.dot(feat4,weights) #probability of Thr18-Cys21 ring formation
        df.loc[j] = [feat1_w, feat2_w, feat3_w, feat4_w]
    pickle.dump(df, open(elem + '_four_feature_bootstrapping.pkl', 'wb')) #save the ring formation probability as a pickle file

df_WT = pickle.load(open('WT_four_feature_bootstrapping.pkl','rb'))
df_Variant1 = pickle.load(open('Variant1_four_feature_bootstrapping.pkl','rb'))
df_Variant2 = pickle.load(open('Variant2_four_feature_bootstrapping.pkl','rb'))


df_WT['Type'] = 'Wild Type'
df_Variant1['Type'] = 'Variant1'
df_Variant2['Type'] = 'Variant2'

df_WT = df_WT[['Thr11-Cys14','Thr18-Cys21','Type']]
df_Variant1 = df_Variant1[['Thr11-Cys14','Thr18-Cys21','Type']]
df_Variant2 = df_Variant2[['Thr11-Cys14', 'Thr18-Cys21','Type']]

df_WT = df_WT.sort_values('Thr11-Cys14')
df_Variant1 = df_Variant1.sort_values('Thr11-Cys14')
df_Variant2 = df_Variant2.sort_values('Thr11-Cys14')

df_tot = df_WT[50:150].append(df_Variant1[50:150], ignore_index = True)
df_tot = df_tot.append(df_Variant2[50:150], ignore_index = True)
df_tot1 = pd.melt(df_tot, id_vars = ['Type'], value_vars=['Thr11-Cys14','Thr18-Cys21'], var_name = 'feat')

fig, axs = plt.subplots(1,1,figsize=(10,7))
flatui = ["#ff6b6b", '#54a0ff',"#00d2d3"]
sns.set_palette(sns.color_palette(flatui))
sns.boxplot(x = 'feat', y = 'value', data = df_tot1, hue = 'Type', order = ['Thr11-Cys14','Thr18-Cys21'])
handles, labels = axs.get_legend_handles_labels()
plt.legend(handles=handles[0:], labels=labels[0:], fontsize = 16)
plt.xlabel('')
plt.xticks([0, 1], ['Thr11-Cys14','Thr18-Cys21'], fontsize = 28)
plt.yticks(fontsize = 18)
plt.ylabel('Probability of ring formation',fontsize = 28)
plt.ylim(0,1.0)
plt.savefig('Pcn3.3_WT_MT1_MT2_two_feat_box_plot.png',dpi = 500)




