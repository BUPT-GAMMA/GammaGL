from re import I
import numpy as np
import argparse

with open('./test_accuracy', 'r+') as f:
    # data = f.read().split('\n')[-10:-1]
    data = f.read().split('\n')[-11:-1] 
    acc = np.array([r.split(' ')[2] for r in data], dtype=np.float32)
    f.write("{} {} {:.2f} {:.2f}\n\n".format(data[0].split(' ')[0], data[0].split(' ')[1], acc.mean(), acc.std()))

