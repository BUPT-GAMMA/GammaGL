import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TL_BACKEND'] = 'torch'
from gammagl.datasets import NGSIM_US_101


def test_ngsim():
    data_path = '../data'
    train_set = NGSIM_US_101(root=data_path, name='train')
    val_set = NGSIM_US_101(root=data_path, name='val')
    test_set = NGSIM_US_101(root=data_path, name='test')
    print(train_set.__len__())
    print(val_set.__len__())
    print(test_set.__len__())


# if __name__ == '__main__':
#     test_ngsim()
