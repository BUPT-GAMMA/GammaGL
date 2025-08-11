from tensorlayerx.dataflow import Dataset
import os
os.environ['TL_BACKEND'] = 'torch'
import numpy as np

class RecDataSet(Dataset):
    def __init__(self, test_id, test_negative_id, is_training=True):
        super(RecDataSet, self).__init__()
        self.is_training = is_training
        if(self.is_training==False): #test
            self.data = (np.array(test_id), np.array(test_negative_id))

    def __getitem__(self,index):
        if(self.is_training==False):
            user = self.data[0][index][0]
            item = self.data[0][index][1]
            negtive_item = self.data[1][index]
            return user, item, negtive_item


    def __len__(self):
        #return self.x.size()
        if(self.is_training==False):
            return len(self.data[0])