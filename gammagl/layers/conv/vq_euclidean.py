import numpy as np
from functools import partial

import tensorlayerx as tlx
from einops import rearrange, repeat, reduce, pack, unpack, einsum

from typing import Callable

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def identity(t):
    return t

def _has_nan(x):
    try:
        arr = x.numpy() if hasattr(x, 'numpy') else tlx.convert_to_numpy(x)
        return bool(np.isnan(arr).any())
    except Exception:
        return False

def stop_gradient(x):
    # Compatibility across TLX versions/backends.
    if hasattr(tlx, "stop_gradient"):
        return tlx.stop_gradient(x)
    if hasattr(tlx, "ops") and hasattr(tlx.ops, "stop_gradient"):
        return tlx.ops.stop_gradient(x)
    backend = getattr(tlx, "BACKEND", None)
    if backend == "tensorflow":
        import tensorflow as tf
        return tf.stop_gradient(x)
    if hasattr(x, "detach"):
        return x.detach()
    return x

def l2norm(t):
    return t / (tlx.sqrt(tlx.reduce_sum(t * t, axis=-1, keepdims=True)) + 1e-8)

def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum(x, y, 'b i d, b j d -> b i j') * -2
    return tlx.sqrt(tlx.maximum(rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy, tlx.zeros(1)))

def log(t, eps = 1e-20):
    return tlx.log(tlx.maximum(t, tlx.convert_to_tensor(eps)))

def _check(tag, t, force=False):
    if _has_nan(t):
        print(f"[VQ-DBG] {tag}: NaN in tensor shape={t.shape}")
        return True
    return False

def ema_inplace(old, new, decay):
    """EMA update that works across all backends."""
    if _has_nan(new):
        return
    backend = getattr(tlx, "BACKEND", None)
    if backend == "tensorflow":
        old.assign(decay * old + (1 - decay) * new)
    elif hasattr(old, 'assign'):
        old.assign(decay * old + (1 - decay) * new)
    else:
        old.copy_(decay * old + (1 - decay) * new)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def uniform_init(*shape):
    t = tlx.convert_to_tensor(np.random.uniform(0, 1, size=shape).astype(np.float32))
    scale = 1.0 / (shape[-1] ** 0.5) if len(shape) > 0 else 1.0
    return t * 2 * scale - scale

def gumbel_noise(t):
    noise = tlx.convert_to_tensor(np.random.uniform(0, 1, size=t.shape).astype(np.float32))
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True
):
    dtype = logits.dtype
    size = int(logits.shape[dim])

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = tlx.argmax(sampling_logits, axis=dim)
    # 只在 dim=-1 且最后一个维度是1时才 squeeze
    if dim == -1 and len(ind.shape) > 1 and ind.shape[-1] == 1:
        ind = tlx.squeeze(ind, axis=-1)
    one_hot = tlx.ops.OneHot(depth=size, dtype=dtype)(ind)
    one_hot = tlx.cast(one_hot, dtype)

    assert not (reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    if reinmax:
        π0 = tlx.softmax(logits, axis=dim)
        π1 = (one_hot + tlx.softmax(logits / temperature, axis=dim)) / 2
        π1 = tlx.softmax(stop_gradient(log(π1) - logits) + logits, axis=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - stop_gradient(π2) + one_hot
    else:
        π1 = tlx.softmax(logits / temperature, axis=dim)
        one_hot = one_hot + π1 - stop_gradient(π1)

    return ind, one_hot

def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    denom = tlx.reduce_sum(x, axis=dim, keepdims=True)
    return (x + eps) / (denom + n_categories * eps)

def sample_vectors(samples, num):
    num_samples = samples.shape[0]
    if num_samples >= num:
        indices = np.random.permutation(num_samples)[:num]
        indices = tlx.cast(tlx.convert_to_tensor(indices), tlx.int64)
    else:
        indices = np.random.randint(0, num_samples, size=(num,))
        indices = tlx.cast(tlx.convert_to_tensor(indices), tlx.int64)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return tlx.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], axis = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_vectors_distributed(local_samples, num):
    return sample_vectors(local_samples, num)

def batched_bincount(x, *, minlength):
    batch, dtype = x.shape[0], x.dtype
    x_np = tlx.convert_to_numpy(x)
    result = np.zeros((batch, minlength), dtype=np.int64)
    for b in range(batch):
        np.add.at(result[b], x_np[b], 1)
    return tlx.convert_to_tensor(result, dtype=dtype)

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype = samples.shape[0], samples.shape[-1], samples.dtype

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = tlx.argmax(dists, axis = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = tlx.where(zero_mask, tlx.ones_like(bins), bins)

        new_means = tlx.zeros((num_codebooks, num_clusters, dim), dtype=dtype)

        expanded_buckets = repeat(buckets, 'h n -> h n d', d = dim)
        new_means_np = np.zeros((num_codebooks, num_clusters, dim), dtype=np.float32)
        samples_np = tlx.convert_to_numpy(samples)
        buckets_np = tlx.convert_to_numpy(buckets)
        for h in range(num_codebooks):
            for n in range(samples.shape[1]):
                cluster_idx = int(buckets_np[h, n])
                new_means_np[h, cluster_idx] = new_means_np[h, cluster_idx] + samples_np[h, n]
        new_means = tlx.convert_to_tensor(new_means_np, dtype=dtype)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = tlx.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    # indices: (h, b, n)
    # embeds: (h, c, d)
    # output should be: (h, b, n, d)
    batch, dim = indices.shape[1], embeds.shape[-1]
    h, c, d = embeds.shape
    _, b, n = indices.shape

    # Convert to numpy for indexing, then back to tensor
    if hasattr(indices, 'numpy'):
        indices_np = indices.numpy()
    else:
        indices_np = np.array(indices)

    if hasattr(embeds, 'numpy'):
        embeds_np = embeds.numpy()
    else:
        embeds_np = np.array(embeds)

    # Manual indexing
    result = np.zeros((h, b, n, d), dtype=embeds_np.dtype)
    for i in range(h):
        for j in range(b):
            for k in range(n):
                idx = int(indices_np[i, j, k])
                result[i, j, k, :] = embeds_np[i, idx, :]

    return tlx.convert_to_tensor(result, dtype=embeds.dtype)


def orthogonal_loss_fn(t):
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum(normed_codes, normed_codes, 'h i d, h j d -> h i j')
    return tlx.reduce_sum(cosine_sim ** 2) / (h * n ** 2) - (1 / n)


class EuclideanCodebook(tlx.nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        affine_param = False,
        sync_affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else tlx.zeros
        embed = init_fn((num_codebooks, codebook_size, dim))

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = noop
        self.all_reduce_fn = noop

        self.initted = tlx.convert_to_tensor([not kmeans_init])
        self.cluster_size = tlx.zeros((num_codebooks, codebook_size))
        self.embed_avg = embed

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = tlx.nn.Parameter(embed)
        else:
            self.embed = tlx.convert_to_tensor(embed)

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.batch_mean = None
        self.batch_variance = None

        self.codebook_mean_needs_init = tlx.convert_to_tensor([True])
        self.codebook_mean = tlx.empty((num_codebooks, 1, dim))
        self.codebook_variance_needs_init = tlx.convert_to_tensor([True])
        self.codebook_variance = tlx.empty((num_codebooks, 1, dim))

    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        if isinstance(self.embed, tlx.nn.Parameter):
            self.embed = tlx.nn.Parameter(embed)
        else:
            self.embed = embed
        self.embed_avg = embed_sum
        self.cluster_size = cluster_size
        self.initted = tlx.convert_to_tensor([True])

    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name, None)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            setattr(self, buffer_name + "_needs_init", tlx.convert_to_tensor([False]))

        if not exists(old_value) or needs_init:
            setattr(self, buffer_name, stop_gradient(new_value))
            return

        value = old_value * decay + stop_gradient(new_value) * (1 - decay)
        setattr(self, buffer_name, value)

    def update_affine(self, data, embed, mask = None):
        assert self.affine_param

        var_fn = lambda x, axis: tlx.reduce_mean((x - tlx.reduce_mean(x, axis=axis, keepdims=True)) ** 2, axis=axis)

        embed = rearrange(embed, 'h ... d -> h (...) d')

        if self.is_train:
            self.update_with_decay('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay('codebook_variance', reduce(embed, 'h n d -> h 1 d', lambda x, a: var_fn(x, a)), self.affine_param_codebook_decay)

        data = rearrange(data, 'h ... d -> h (...) d')

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        if not self.sync_affine_param:
            self.update_with_decay('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay('batch_variance', reduce(data, 'h n d -> h 1 d', lambda x, a: var_fn(x, a)), self.affine_param_batch_decay)
            return

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not tlx.reduce_any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), int(tlx.reduce_sum(tlx.cast(mask, tlx.float32))))
            sampled = rearrange(sampled, '1 ... -> ...')

            if self.learnable_codebook:
                self.embed = tlx.tensor_scatter_nd_update(self.embed, ...)
            else:
                self.embed = tlx.where(...)

            self.cluster_size = tlx.where(...)
            self.embed_avg = tlx.where(...)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not tlx.reduce_any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        if _has_nan(x):
            x = tlx.where(tlx.is_nan(x), tlx.zeros_like(x), x)

        x = tlx.cast(x, tlx.float32)

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype
        flatten, ps = pack_one(x, 'h * d')
        _check("csc_flatten", flatten)

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if not self.learnable_codebook:
            try:
                embed_arr = self.embed.numpy() if hasattr(self.embed, 'numpy') else np.array(self.embed)
            except Exception:
                embed_arr = self.embed
            embed_arr_np = np.asarray(embed_arr)
            is_nan = bool(np.isnan(embed_arr_np).any())
            if is_nan:
                print("[VQ-FIX] EuclideanCodebook: NaN detected in codebook embed, resetting...")
                self.initted = tlx.convert_to_tensor([False])
                self.cluster_size = tlx.zeros_like(self.cluster_size)
                self.embed_avg = uniform_init((self.num_codebooks, self.codebook_size, self.embed.shape[-1]))
                self.embed = tlx.convert_to_tensor(self.embed_avg)
                self.init_embed_(flatten, mask = mask)
                print("[VQ-FIX] EuclideanCodebook: reset complete, new embed has_nan=",
                      bool(np.isnan(tlx.convert_to_numpy(self.embed)).any()))

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask = mask)

        embed = self.embed if self.learnable_codebook else stop_gradient(self.embed)

        if self.affine_param:
            codebook_std = tlx.sqrt(tlx.maximum(self.codebook_variance, tlx.convert_to_tensor(1e-5)))
            batch_std = tlx.sqrt(tlx.maximum(self.batch_variance, tlx.convert_to_tensor(1e-5)))
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        dist = -cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.is_train)

        embed_ind = unpack_one(embed_ind, ps, 'h *')

        if self.is_train:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum(unpacked_onehot, embed, 'h b n c, h c d -> h b n d')
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.is_train and self.ema_update and not freeze_codebook:

            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if exists(mask):
                embed_onehot = tlx.where(mask, embed_onehot, tlx.zeros_like(embed_onehot))

            cluster_size = tlx.reduce_sum(embed_onehot, axis = 1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = einsum(flatten, embed_onehot, 'h n d, h n c -> h c d')
            if hasattr(embed_sum, 'contiguous'):
                embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * tlx.reduce_sum(self.cluster_size, axis = -1, keepdims=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            if not _has_nan(embed_normalized):
                self.embed = embed_normalized
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize = rearrange(quantize, '1 ... -> ...')
            embed_ind = rearrange(embed_ind, '1 ... -> ...')

        dist = unpack_one(dist, ps, 'h * d')

        return quantize, embed_ind, dist


class CosineSimCodebook(tlx.nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True
    ):
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.decay = decay

        embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = noop
        self.all_reduce_fn = noop

        self.initted = tlx.convert_to_tensor([not kmeans_init])
        self.cluster_size = tlx.zeros((num_codebooks, codebook_size))
        self.embed_avg = embed

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = tlx.nn.Parameter(embed)
        else:
            self.embed = tlx.convert_to_tensor(embed)

    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        if isinstance(self.embed, tlx.nn.Parameter):
            self.embed = tlx.nn.Parameter(embed)
        else:
            self.embed = embed
        self.embed_avg = embed_sum
        self.cluster_size = cluster_size
        self.initted = tlx.convert_to_tensor([True])

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not tlx.reduce_any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), int(tlx.reduce_sum(tlx.cast(mask, tlx.float32))))
            sampled = rearrange(sampled, '1 ... -> ...')

            if self.learnable_codebook:
                self.embed = tlx.tensor_scatter_nd_update(self.embed, ...)
            else:
                self.embed = tlx.where(...)
            self.embed_avg = tlx.where(...)
            self.cluster_size = tlx.where(...)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not tlx.reduce_any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        if _has_nan(x):
            x = tlx.where(tlx.is_nan(x), tlx.zeros_like(x), x)

        x = tlx.cast(x, tlx.float32)

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype

        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if self.learnable_codebook:
            _check("csc_embed_before_use", self.embed)

        embed = self.embed if self.learnable_codebook else stop_gradient(self.embed)

        dist = einsum(flatten, embed, 'h n d, h c d -> h n c')
        _check("csc_dist", dist)

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.is_train)
        embed_ind = unpack_one(embed_ind, ps, 'h *')
        _check("csc_embed_onehot", embed_onehot)

        if self.is_train:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum(unpacked_onehot, embed, 'h b n c, h c d -> h b n d')
            _check("csc_quantize", quantize)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.is_train and self.ema_update and not freeze_codebook:
            if exists(mask):
                embed_onehot = tlx.where(mask, embed_onehot, tlx.zeros_like(embed_onehot))

            bins = tlx.reduce_sum(embed_onehot, axis = 1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size, bins, self.decay)

            embed_sum = einsum(flatten, embed_onehot, 'h n d, h n c -> h c d')
            if hasattr(embed_sum, 'contiguous'):
                embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * tlx.reduce_sum(self.cluster_size, axis = -1, keepdims=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            if not _has_nan(embed_normalized):
                self.embed = l2norm(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize = rearrange(quantize, '1 ... -> ...')
            embed_ind = rearrange(embed_ind, '1 ... -> ...')

        dist = unpack_one(dist, ps, 'h * d')
        return quantize, embed_ind, dist


class VectorQuantize_E(tlx.nn.Module):
    def __init__(
        self,
        manifold,
        dim,
        codebook_size,
        codebook_dim = None,
        heads = 4,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        reinmax = False,
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        learnable_codebook = False,
        in_place_codebook_optimizer = None,
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0.
    ):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = tlx.layers.Linear(in_features=dim, out_features=codebook_input_dim, b_init=None) if requires_projection else None
        self.project_out = tlx.layers.Linear(in_features=codebook_input_dim, out_features=dim, b_init=None) if requires_projection else None

        self.has_projections = requires_projection

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'

        self.sync_update_v = sync_update_v

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through
        )

        if not exists(sync_codebook):
            sync_codebook = False

        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update
        )

        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        self._codebook.embed = codes

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack_one(indices, 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        # Manual gather for TensorFlow compatibility
        # PyTorch: codebook.gather(2, indices)
        b, h, n = indices.shape
        _, _, d = codebook.shape

        if hasattr(indices, 'numpy'):
            indices_np = indices.numpy()
        else:
            indices_np = np.array(indices)

        if hasattr(codebook, 'numpy'):
            codebook_np = codebook.numpy()
        else:
            codebook_np = np.array(codebook)

        codes_np = np.zeros((b, h, n, d), dtype=codebook_np.dtype)
        for i in range(b):
            for j in range(h):
                for k in range(n):
                    idx = int(indices_np[i, j, k])
                    codes_np[i, j, k, :] = codebook_np[j, idx, :]

        codes = tlx.convert_to_tensor(codes_np, dtype=codebook.dtype)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes = unpack_one(codes, ps, 'b * d')
        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        if self.project_out is not None:
            return self.project_out(codes)
        return codes

    def forward(
        self,
        x,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False
    ):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, heads, is_multiheaded, codebook_size, return_loss = x.shape, self.heads, self.heads > 1, self.codebook_size, exists(indices)

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        if self.project_in is not None:
            x = self.project_in(x)

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        x = self._codebook.transform_input(x)

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        )

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.is_train and not self.learnable_codebook:
            if _has_nan(quantize):
                print(f"[VQ-FIX] VectorQuantize_E: NaN in codebook output, falling back to input. shape={quantize.shape}")
                quantize = x
                distances = tlx.zeros_like(distances) if hasattr(distances, 'shape') else distances

        if should_inplace_optimize and self.is_train and not freeze_codebook:

            if exists(mask):
                loss = tlx.reduce_mean((quantize - stop_gradient(x)) ** 2)
            else:
                loss = tlx.reduce_mean((quantize - stop_gradient(x)) ** 2)

            self.in_place_codebook_optimizer.minimize(loss)

            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.is_train:
            commit_quantize = stop_gradient(quantize)

            quantize = x + stop_gradient(quantize - x)

            if self.sync_update_v > 0.:
                quantize = quantize + self.sync_update_v * (quantize - stop_gradient(quantize))

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'

            ce_loss = tlx.cross_entropy(
                rearrange(distances, dist_einops_eq, b = shape[0]),
                codes,
                ignore_index = -1
            )

            return ce_loss

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        if only_one:
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        loss = tlx.zeros((1,), dtype=x.dtype)

        if self.is_train:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h = heads)

                        embed_ind = tlx.where(~ce_loss_mask, -1, embed_ind)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        commit_loss = (commit_quantize - x) ** 2

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])

                        commit_loss = tlx.reduce_mean(commit_loss[loss_mask])
                    else:
                        commit_loss = tlx.reduce_mean((commit_quantize - x) ** 2)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), 'orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet'
                    unique_code_ids = tlx.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = np.random.permutation(num_codes)[:self.orthogonal_reg_max_codes]
                    rand_ids = tlx.cast(tlx.convert_to_tensor(rand_ids), tlx.int64)
                    codebook = tlx.gather(codebook, rand_ids, axis=1)

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)

        orig_quantize = quantize
        if self.project_out is not None:
            quantize = self.project_out(quantize)
        quantize = quantize / (tlx.sqrt(tlx.reduce_sum(quantize * quantize, axis=-1, keepdims=True)) + 1e-8)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)

        if only_one:
            orig_quantize = rearrange(orig_quantize, 'b 1 d -> b d')
            quantize = rearrange(quantize, 'b 1 d -> b d')

        if exists(mask):
            quantize = tlx.where(
                rearrange(mask, '... -> ... 1'),
                quantize,
                orig_input
            )

        return quantize, embed_ind, loss, orig_quantize
