import subprocess as sbp
import numpy as np
import pandas as pd
import os
import argparse


def train(args):
    log = args.dataset+'_log.csv'

    lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    l2_coefs = [1e-4, 5e-4, 1e-3, 5e-3]
    drop_probs = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    hidden_dims = [16]
    count = 0
    train_num = 5
    os.system('echo hidden_dims,lr,lc,keep_rate,Test_acc >> {}'.format(log))
    for hidden_dim in hidden_dims:
        for lr in lrs:
            for lc in l2_coefs:
                for dp in drop_probs:
                    for i in range(train_num):
                        count += 1
                         print(count)
                         sbp.run('python gcn_trainer.py --dataset {} --hidden_dim {} --lr {} --l2_coef {} --keep_rate {} >> {}'.format(
                             args.dataset, hidden_dim, lr, lc,1-dp, log), shell=True)

    cora = pd.read_csv(log, chunksize = train_num)



    result = pd.DataFrame(columns=('hidden_dims','lr','lc','keep_rate','Test_acc_mean','Test_acc_std'))
    count = 0
    for piece in cora:
        hidden_dims = piece.iat[0,0]
        lr = piece.iat[0,1]
        lc = piece.iat[0,2]
        keep_rate = piece.iat[0,3]
        Test_acc_mean = round(piece.iloc[:,4].mean(),4)
        Test_acc_std = round(piece.iloc[:,4].std(),4)
        result = result.append({'hidden_dims':hidden_dims,'lr':lr,'lc':lc,'keep_rate':keep_rate,
                                'Test_acc_mean':Test_acc_mean,'Test_acc_std':Test_acc_std}, ignore_index=True)
        count = count + 5
    result = result.sort_values(by = 'Test_acc_mean', ascending=False)
    result.to_csv(args.dataset+'_result.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cora', help='dataset')
    args = parser.parse_args()
    train(args)

    
