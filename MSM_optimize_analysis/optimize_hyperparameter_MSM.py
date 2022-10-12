import glob
import numpy as np
import pyemma
import pickle
import mdtraj as md

totdist = []
for file in sorted(glob.glob('Pcn3.3/WT/features_per_trajectory/*all-dist-per-traj.npy')):
    distI = np.load(file)
    print(file)
    totdist.append(distI)

tic_dims = [2,4,6,8,10]
cluster = [50,80,100,150,200,400]
tica_lag_time = 250
msm_lag_time = 200

for cluster_number in cluster:
    for tic_dim in tic_dims:
        tic = pyemma.coordinates.tica(totdist,lag=tica_lag_time,dim = tic_dim)
        data_tic = tic.get_output()
        file = "./MSM/MSM-" + 'procA3.3WT-cluster_kmeans_' + 'C_' + str(cluster_number) + '_lt_' + str(tica_lag_time) + '_ticdim_' + str(tic_dim) + ".pkl"
        cluster_kmeans = pyemma.coordinates.cluster_kmeans(data_tic, k=cluster_number, max_iter=100, stride=5)
        dtrajs = cluster_kmeans.dtrajs
        pickle.dump(cluster_kmeans.dtrajs,open(file,'wb'))
        #dtrajs = pickle.load(open(file,'rb'))
        msm = pyemma.msm.estimate_markov_model(dtrajs, lag = msm_lag_time)
        pickle.dump(msm,open("./MSM/MSM-" +'procA3.3WT-MSM_' +'C_'+str(cluster_number)+'_lt_' + str(tica_lag_time)+'_ticdim_' + str(tic_dim) + ".pkl",'wb'))
        score = msm.score_cv(dtrajs,score_method="VAMP2",score_k=6)
        print(np.mean(score))
        pickle.dump(score,open("./MSM/MSM-" + 'procA3.3WT-score_' + 'C_' + str(cluster_number) + '_lt_' + str(tica_lag_time) + '_ticdim_' + str(tic_dim) +".pkl",'wb'))
        del msm
        del dtrajs

del totdist   
