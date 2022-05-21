import subprocess as sbp
import numpy as np
import pandas as pd
import os

log = 'log_ctsr.txt'
result = 'result_ctsr.csv'

iterations = [0,1,2,3,4]
# epochs = [100] #20,40,80
lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001]
l2_coefs = [1e-4, 5e-4, 1e-3, 5e-3,1e-2]
drop_rate = [0.3, 0.4, 0.5, 0.6, 0.7]

count = 0
count_ite = 0

for lr in lrs:
    for lc in l2_coefs:
        for dr in drop_rate:
            count += 1
            print(count)
            for ite in iterations:
                # for epoch in epochs:
                os.system('echo "\n lr={},lc={},dr={}--" >> {}'.format( lr, lc, dr, log))
                sbp.run('python gcn_trainer.py --dataset citeseer --lr {} --l2_coef {} --drop_rate {} >> {}'.format( lr, lc, dr, log), shell=True)

#--dataset_path gammagl/datasets/cora
# f = open(log, 'r')
# lines = f.readlines()
# lines.reverse()

path = 'log_ctsr.txt'
train_num = 5
cora = pd.read_csv(path, chunksize = train_num*2)
auc_list = np.array([])

result = pd.DataFrame(columns=('lr','lc','drop_rate','Test_acc_mean','Test_acc_std'))
#count = 0
for piece in cora:
    lr = piece.iat[1,0]
    lc = piece.iat[1,1]
    keep_rate = piece.iat[1,2]
    acc_t1 = piece.iat[0,0] +" "+ piece.iat[2,0]+" "+piece.iat[4,0]+" "+piece.iat[6,0]+" "+piece.iat[8,0]
    acc_t2 =acc_t1.strip().split()
    auc_list =np.append(auc_list,float(acc_t2[2]))
    auc_list = np.append(auc_list, float(acc_t2[5]))
    auc_list = np.append(auc_list, float(acc_t2[8]))
    auc_list = np.append(auc_list, float(acc_t2[11]))
    auc_list = np.append(auc_list, float(acc_t2[14]))
    # acc_sum = float(acc_t2[2]) + float(acc_t2[5])+float(acc_t2[8])+float(acc_t2[11])+float(acc_t2[14])
    # acc_mean = acc_sum/5
    Test_acc_mean = round(auc_list.mean(),4)
    Test_acc_std = round(auc_list.std(),4)
    result = result.append({'lr':lr,'lc':lc,'drop_rate':drop_rate,
    'Test_acc_mean':Test_acc_mean,'Test_acc_std':Test_acc_std}, ignore_index=True)
 #   count = count + 5
result = result.sort_values(by = 'Test_acc_mean', ascending=False)
result.to_csv('result_cora_paddle.csv')