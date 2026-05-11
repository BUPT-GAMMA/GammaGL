import os.path
import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import warnings
import pickle

warnings.filterwarnings("ignore")

def get_word2vec_dim(model_name):
    if model_name == 'glove-wiki-gigaword-100':
        return 100

def act_fn(act_str: str):
    if act_str == 'relu':
        return tlx.relu
    elif act_str == 'leaky_relu':
        return lambda x: tlx.leaky_relu(x, negative_slope=0.2)
    elif act_str == 'tanh':
        return tlx.tanh
    elif act_str == 'elu':
        return tlx.elu
    elif act_str is None:
        return lambda x: x
    else:
        raise NotImplementedError

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, dir, file, save=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, dir, file)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, dir, file)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, dir, file):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        
        # Save with state_dict keys so later load maps consistently.
        save_dict = {}
        try:
            state = model.state_dict()
        except Exception:
            state = None

        if isinstance(state, dict) and len(state) > 0:
            for idx, (key, val) in enumerate(state.items()):
                if key is None:
                    key = f"param_{idx}"
                if not isinstance(key, str):
                    key = str(key)
                if hasattr(val, 'detach'):
                    v = val.detach()
                    if hasattr(v, 'cpu'):
                        v = v.cpu()
                    save_dict[key] = v.numpy()
                elif hasattr(val, 'numpy'):
                    save_dict[key] = val.numpy()
                else:
                    save_dict[key] = np.array(val)
        else:
            # Fallback for very old TLX behavior.
            for idx, weight in enumerate(model.trainable_weights):
                key = weight.name if (hasattr(weight, "name") and weight.name is not None) else f"param_{idx}"
                if not isinstance(key, str):
                    key = str(key)
                val = weight
                if hasattr(val, 'detach'):
                    v = val.detach()
                    if hasattr(v, 'cpu'):
                        v = v.cpu()
                    save_dict[key] = v.numpy()
                elif hasattr(val, 'numpy'):
                    save_dict[key] = val.numpy()
                else:
                    save_dict[key] = np.array(val)
        
        with open(os.path.join(dir, file), 'wb') as f:
            pickle.dump(save_dict, f)
        self.val_loss_min = val_loss
