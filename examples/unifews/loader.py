"""Example-local data loading for UNIFEWS decouple mode.

This module depends on the optional Cython precompute extension.
Only import this if you need decouple mode (run_mlp.py).
"""

import os
import gc
import numpy as np
import tensorlayerx as tlx
from dotmap import DotMap

from gammagl.utils.data_processor import DataProcess, DataProcess_inductive, matstd_clip


def dmap2dct(chnname, dmap, processor):
    typedct = {'sgc': -2, 'gbp': -3,
               'sgc_agp': 0, 'gbp_agp': 1,
               'sgc_thr': 2, 'gbp_thr': 3}
    ctype = chnname.split('_')[0]
    dct = {}
    dct['type'] = typedct[chnname]
    dct['hop'] = dmap.hop
    dct['dim'] = processor.nfeat
    dct['delta'] = dmap.delta if type(dmap.delta) is float else 1e-5
    dct['alpha'] = dmap.alpha if (type(dmap.alpha) is float and not (ctype == 'sgc')) else 0
    dct['rra'] = (1 - dmap.rrz) if type(dmap.rrz) is float else 0
    dct['rrb'] = dmap.rrz if type(dmap.rrz) is float else 0
    return dct


def load_embedding(datastr, algo, algo_chn,
                   datapath="./data/",
                   inductive=False, multil=False,
                   seed=0, **kwargs):
    # Lazy import of precompute Cython extension (only needed for decouple mode)
    try:
        from examples.unifews.precompute.prop import A2Prop
    except ImportError as e:
        raise RuntimeError(
            "UNIFEWS decouple mode requires building examples/unifews/precompute. "
            "Run: cd examples/unifews/precompute && python setup.py build_ext --inplace"
        ) from e

    dp = DataProcess(datastr, path=datapath, seed=seed)
    dp.input(['labels', 'attr_matrix', 'deg'])
    if inductive:
        dpi = DataProcess_inductive(datastr, path=datapath, seed=seed)
        dpi.input(['attr_matrix', 'deg'])
    else:
        dpi = dp

    if (datastr.startswith('cora') or datastr.startswith('citeseer') or datastr.startswith('pubmed')):
        dp.calculate(['idx_train'])
    else:
        dp.input(['idx_train', 'idx_val', 'idx_test'])
    idx = {'train': tlx.convert_to_tensor(dp.idx_train, tlx.int64),
           'val':   tlx.convert_to_tensor(dp.idx_val, tlx.int64),
           'test':  tlx.convert_to_tensor(dp.idx_test, tlx.int64)}

    if multil:
        dp.calculate(['labels_oh'])
        dp.labels_oh[dp.labels_oh < 0] = 0
        labels = tlx.convert_to_tensor(dp.labels_oh, tlx.float32)
    else:
        dp.labels[dp.labels < 0] = 0
        labels = tlx.convert_to_tensor(dp.labels.flatten(), tlx.int64)
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}

    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)

    py_a2prop = A2Prop()
    py_a2prop.load(os.path.join(datapath, datastr), m, n, seed)
    chn = dmap2dct(algo, DotMap(algo_chn), dp)

    feat = dp.attr_matrix.transpose().astype(np.float32, order='C')
    macs_pre, time_pre = py_a2prop.compute(1, [chn], feat)
    feat = feat.transpose()
    feat = matstd_clip(feat, idx['train'], with_mean=True)

    feats = {'val':  tlx.convert_to_tensor(feat[idx['val']], tlx.float32),
             'test': tlx.convert_to_tensor(feat[idx['test']], tlx.float32)}
    feats['train'] = tlx.convert_to_tensor(feat[idx['train']], tlx.float32)

    del feat
    gc.collect()
    return feats, labels, idx, nfeat, nclass, macs_pre/1e9, time_pre
