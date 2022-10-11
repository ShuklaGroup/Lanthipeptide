import numpy as np
import pandas as pd
import glob
import mdtraj as md
import matplotlib.pyplot as plt
import pickle

threshold = 0.45   #the ring formation was defined as residue-residue distances within this thresold

peptides = ['Pcn1.1', 'Pcn2.8']
feature_index = [[26,89,31,69],[34,129,38,107]] #[26,89,31,69] is index of feature for Pcn1.1, 'Cys3-Thr7','Thr12-Cys16','Cys3-Thr12','Thr7-Cys16';
                                                #[34,129,38,107] is index of feature for Pcn2.8, 'Cys3-Ser9','Ser13-Cys19','Cys3-Ser13','Ser9-Cys19'

for i in range(len(peptides)):
    #load 200 bootstrapping samples and their corresponding MSM
    df = pd.DataFrame(columns = ['feat1_w', 'feat2_w', 'feat3_w', 'feat4_w'])
    for j in range(200):
        files = pickle.load(open(peptides[i] + "/bootstrapping/bt_80_" + str(j) +"_files.pkl",'rb')) #the pickle file includes the randomly selected trajctories (80% of all trajectoires)
        msm = pickle.load(open(peptides[i] + "/bootstrapping/bt_80_" + str(j) +"_msm.pkl",'rb')) #the MSM for corresponding subset of all trajectories
        feature = []
        for file in files:
            feature.extend(np.load(peptides[i] + "/features_per_trajectory/"+ file)[:,feature_index[i]]) #load features of each trajectory in the subset
        
        feature = np.vstack(feature)
        boolean_list1 = feature[:,0] < threshold #if feature (residue pair distance) < thresold, we assign True; otherwise, False
        feat1 = list(map(int, boolean_list1)) #convert True to 1, and False to 0
        boolean_list2 = feature[:,1] < threshold
        feat2 = list(map(int, boolean_list2))
        boolean_list3 = feature[:,2] < threshold
        feat3 = list(map(int, boolean_list3))
        boolean_list4 = feature[:,3] < threshold
        feat4 = list(map(int, boolean_list4))
     
        weights = np.concatenate(msm.trajectory_weights()) #get weights from MSM
        feat1_weighted =  np.dot(feat1, weights) #for distance betweehn Cys3 and Thr7, dot product of (ring form(1) or not(0)) and (weights) = probability of ring formation
        feat2_weighted =  np.dot(feat2, weights) #for distance between Thr12 and Cys16, dot product of (ring form(1) or not(0)) and (weights) = probability of ring formation
        feat3_weighted =  np.dot(feat3, weights) #for distance between Cys3 and Thr12, dot product of (ring form(1) or not(0)) and (weights) = probability of ring formation
        feat4_weighted =  np.dot(feat4, weights) #for distance between Thr7 and Cys16, dot product of (ring form(1) or not(0)) and (weights) = probability of ring formation

        df.loc[j] = [feat1_weighted, feat2_weighted, feat3_weighted, feat4_weighted] #create a dataframe to store ring formation prob
    pickle.dump(df, open(peptides[i] + '_four_feat_bt80_200.pkl','wb')) #save dataframe of ring formation prob as a pickle file

df1 = pickle.load(open('Pcn1.1_four_feat_bt80_200.pkl','rb'))
df1_np = df1.to_numpy()
#box plot for probability of ring formation for Pcn1.1
fig, axs = plt.subplots(1,1,figsize=(10,7))
axs.boxplot(df1_np)
plt.xticks([1, 2, 3, 4], ['C3-T7', 'T12-C16', 'C3-T12', 'T7-C16'])
plt.ylabel('Probability of ring formation',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4,0.5,0.6],fontsize=20)
plt.ylim(0, 0.6)
plt.savefig('pcn1.1_box_plot.png', dpi=500)


###box plot for probability of ring formation for Pcn2.8
df2 = pickle.load(open('Pcn2.8_four_feat_bt80_200.pkl','rb'))
df2_np = df2.to_numpy()
fig, axs = plt.subplots(1,1,figsize=(10,7))
axs.boxplot(df2_np)
plt.xticks([1, 2, 3, 4], ['C3-S9', 'S13-C19', 'C3-S13', 'S9-C19'])
plt.ylabel('Probability of ring formation',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4],fontsize=20)
plt.ylim(0, 0.4)
plt.savefig('pcn2.8_box_plot.png', dpi=500)

