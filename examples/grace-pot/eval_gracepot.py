from typing import Optional
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.optimizers import Adam
import torch
from model_ggl import LogReg
from sklearn.metrics import f1_score
import tensorflow
def get_idx_split(dataset, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = np.random.permutation(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')

def nll_loss_func(output, target):
    return torch.nn.NLLLoss()(output, target)

def log_regression(z,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    z=z.detach()
    num_hidden = tlx.get_tensor_shape(z)[1]
    y = dataset[0].y
    y = tlx.reshape(y, [-1])
    num_classes = tlx.convert_to_numpy(tlx.reduce_max(y) + 1)

    split = get_idx_split(dataset, split, preload_split)
    split = {k: v for k, v in split.items()}

    classifier = LogReg(num_hidden, num_classes)
    optimizer = Adam(lr=0.01, weight_decay=0.0)
    train_weights = classifier.trainable_weights
    nll_loss=nll_loss_func
    net_with_loss = tlx.model.WithLoss(classifier, nll_loss)
    train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, train_weights)

    best_test_mi = 0
    best_test_ma = 0
    best_val_mi = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.set_train()
        loss = train_one_step(z[split['train']], y[split['train']])

        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                    # val split is available
                test_res = evaluator.eval({
                   'y_true': y[split['test']].reshape(-1, 1),
                  'y_pred': tlx.reshape(tlx.argmax(classifier(z[split['test']]),axis=-1),[-1, 1])
                })
                test_mi, test_ma = test_res['F1Mi'], test_res['F1Ma']
                val_res = evaluator.eval({
                    'y_true': y[split['val']].reshape(-1, 1),
                    'y_pred': tlx.reshape(tlx.argmax(classifier(z[split['val']]),axis=-1),[-1, 1])
                })
                val_mi, val_ma = val_res['F1Mi'], val_res['F1Ma']
                if val_mi > best_val_mi:
                    best_val_mi = val_mi
                    best_test_mi = test_mi
                    best_test_ma = test_ma
                    best_epoch = epoch
            else:
                test_res = evaluator.eval({
                    'y_true': y[split['test']].reshape(-1, 1),
                    'y_pred': tlx.reshape(tlx.argmax(classifier(z[split['test']]),axis=-1),[-1, 1])
                })
                test_mi, test_ma = test_res['F1Mi'], test_res['F1Ma']
                if best_test_mi < test_mi:
                    best_test_mi = test_mi
                    best_test_ma = test_ma
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test mi {best_test_mi}, best test ma {best_test_ma}')
                
    return {'F1Mi': best_test_mi, 'F1Ma': best_test_ma}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = tlx.convert_to_numpy(y_pred).flatten()
        # y_pred = y_pred.flatten()
        micro = f1_score(y_true.cpu(), y_pred, average="micro")
        macro = f1_score(y_true.cpu(), y_pred, average="macro")

        return {
            'F1Mi': micro,
            'F1Ma': macro
        }

    def eval(self, res):
        return self._eval(**res)
