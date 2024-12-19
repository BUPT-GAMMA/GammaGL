from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import tensorlayerx as tlx
from gammagl.data import BaseGraph
from gammagl.data.storage import BaseStorage, NodeStorage
from gammagl.utils.check import check_is_numpy


def collate(
    cls,
    data_list: List[BaseGraph],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> Tuple[BaseGraph, Mapping, Mapping]:
    # Collates a list of `Graph` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous graph objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = dict()
    for data in data_list:
        for store in data.stores:
            key_to_stores.setdefault(store._key, []).append(store)
    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device = None
    slice_dict, inc_dict = dict(), dict()
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == 'num_nodes':
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == 'ptr':
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores,
                                           increment)
            if tlx.BACKEND == 'torch':
                device = value.device if tlx.is_tensor(value) else device

            out_store[attr] = value
            if key is not None:
                slice_dict.setdefault(key, dict())[attr] = slices
                inc_dict.setdefault(key, dict())[attr] = incs
            else:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if (attr in follow_batch and tlx.is_tensor(slices)
                    and slices.ndim == 1):
                repeats = slices[1:] - slices[:-1]
                batch = repeat_interleave(tlx.convert_to_numpy(repeats).tolist())
                out_store[f'{attr}_batch'] = batch
                out_store[f'{attr}_ptr'] = cumsum(repeats)

        # In case the storage holds node, we add a top-level batch vector it:
        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            repeats = [store.num_nodes for store in stores]
            out_store.batch = repeat_interleave(repeats, )
            out_store.ptr = cumsum(repeats)

            # Sometimes stores can't get num nodes
            # repeats = [store.num_nodes for store in data_list]
            # out_store.batch = repeat_interleave(repeats, device=device)
            # out_store.ptr = cumsum(tlx.convert_to_tensor(repeats, dtype=tlx.int64))

    return out, slice_dict, inc_dict


def _collate(
        key: str,
        values: List[Any],
        data_list: List[BaseGraph],
        stores: List[BaseStorage],
        increment: bool,
) -> Tuple[Any, Any, Any]:

    elem = values[0]

    if tlx.is_tensor(elem):
        # Concatenate a list of `Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        if cat_dim is None or elem.ndim == 0:
            values = [tlx.expand_dims(value, axis=0) for value in values]
        slices = cumsum([tlx.get_tensor_shape(value)[cat_dim or 0] for value in values])
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.ndim > 1 or int(incs[-1]) != 0:
                values = [
                    # value + inc.to(value.device)
                    value + inc
                    for value, inc in zip(values, incs)
                ]
        else:
            incs = None

        # if tlx.utils.data.get_worker_info() is not None:
        #     # Write directly into shared memory to avoid an extra copy:
        #     numel = sum(value.numel() for value in values)
        #     storage = elem.storage()._new_shared(numel)
        #     out = elem.new(storage)
        # else:
        #     out = None

        value = tlx.concat(values, axis=cat_dim or 0)
        return value, slices, incs

    # elif isinstance(elem, SparseTensor) and increment:
    #     # Concatenate a list of `SparseTensor` along the `cat_dim`.
    #     # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
    #     cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
    #     cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
    #     repeats = [[value.size(dim) for dim in cat_dims] for value in values]
    #     slices = cumsum(repeats)
    #     value = cat(values, dim=cat_dim)
    #     return value, slices, None

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `tlx.Tensor`.
        value = tlx.convert_to_tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value.add_(incs)
        else:
            incs = None
        slices = tlx.arange(start=0, limit=(len(values) + 1))
        return value, slices, incs

    elif isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in elem.keys():
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(
                key, [v[key] for v in values], data_list, stores, increment)
        return value_dict, slice_dict, inc_dict

    elif (isinstance(elem, Sequence) and not isinstance(elem, str) and len(elem) > 0
          and tlx.is_tensor(elem[0])):
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(key, [v[i] for v in values],
                                           data_list, stores, increment)
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        # Other-wise, just return the list of values as it is.
        slices = tlx.arange(start=0, limit=len(values) + 1)
        return values, slices, None


###############################################################################


def repeat_interleave(
    repeats: List[int],
    device=None,
):
    outs = [tlx.constant(value=i, shape=(n, ), dtype=tlx.int64) for i, n in enumerate(repeats)]
    return tlx.concat(outs, axis=0)


def cumsum(value):
    # if not tlx.is_tensor(value):
    #     value = tlx.convert_to_tensor(value, dtype=tlx.int64)
    # out = tlx.concat([tlx.zeros(shape=(tlx.get_tensor_shape(value)[0], ), dtype=tlx.int64), tlx.cumsum(value, 0)], axis=0)
    # out = tlx.zeros([tlx.get_tensor_shape(value)[0] + 1, ] + tlx.get_tensor_shape(value)[1:])
    # out[0] = 0
    # out[1:] = tlx.cumsum(value, axis=0)
    # TODO  if assign value is ok
    if not check_is_numpy(value):
        if tlx.is_tensor(value):
            value = tlx.convert_to_numpy(value)
        else:
            value = np.array(value)
    out = np.empty((value.shape[0] + 1, ) + value.shape[1:])
    out[0] = 0
    out[1:] = np.cumsum(value, 0)
    return tlx.convert_to_tensor(out, dtype=tlx.int64)


def get_incs(key, values: List[Any], data_list: List[BaseGraph],
             stores: List[BaseStorage]):
    repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores)
    ]
    if tlx.is_tensor(repeats[0]):
        repeats = tlx.stack(repeats, axis=0)
    else:
        repeats = tlx.convert_to_tensor(repeats, dtype=tlx.int64)
    return cumsum(repeats[:-1])