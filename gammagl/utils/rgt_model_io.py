import numpy as np
import tensorlayerx as tlx
import os
import pickle


def save_model(state_dict, path):
    """Save model state dict to file in a backend-agnostic way."""
    save_dict = {}
    for idx, (key, val) in enumerate(state_dict.items()):
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
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)


def load_model(path):
    """Load model state dict from file in a backend-agnostic way."""
    try:
        import torch
        import tensorlayerx as tlx
        save_dict = torch.load(path, map_location='cpu')
        
        # Transpose weight matrices of Linear layers when using TF/MindSpore backend
        # PyTorch linear weight: [out_features, in_features]
        # TLX linear weight (TF backend): [in_features, out_features]
        if tlx.BACKEND in ['tensorflow', 'mindspore', 'paddle']:
            for key, val in save_dict.items():
                if key is None:
                    continue
                if not isinstance(key, str):
                    key = str(key)
                # Checking if it's a 2D weight matrix (likely from a linear layer)
                if 'weight' in key and len(val.shape) == 2:
                    # In PyTorch, some embeddings might be 2D too, but for RGT we mostly have linear layers
                    if not 'embed' in key: # Skip embedding matrices if any
                        save_dict[key] = val.t()
                        
    except Exception as e:
        try:
            with open(path, 'rb') as f:
                save_dict = pickle.load(f)
        except Exception as e2:
            raise RuntimeError(f"Failed to load model from {path}. Tried torch.load ({e}) and pickle.load ({e2})")
    return save_dict
