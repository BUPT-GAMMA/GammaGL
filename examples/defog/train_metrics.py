import tensorlayerx as tlx
import numpy as np


class CrossEntropyMetric:
    r"""Accumulator for cross-entropy loss with running mean.

    Tracks cross-entropy over multiple batches and computes the epoch-level
    average.
    """
    def __init__(self):
        self.total_loss = 0.0
        self.total_samples = 0

    def update(self, preds, targets):
        r"""Update metric with a batch of predictions and targets.

        Parameters
        ----------
        preds : tensor
            Logits ``(N, C)``.
        targets : tensor
            One-hot or integer targets ``(N,)`` or ``(N, C)``.

        Returns
        -------
        float
            Batch cross-entropy loss.
        """
        return self.update_precomputed(self._compute_loss(preds, targets), preds.shape[0])

    def _compute_loss(self, preds, targets):
        if tlx.BACKEND == 'torch':
            import torch.nn.functional as F

            if len(targets.shape) > 1 and targets.shape[-1] > 1:
                targets = tlx.argmax(targets, axis=-1)
            targets = tlx.cast(targets, tlx.int64)
            loss = F.cross_entropy(preds, targets)
            return float(loss.item())
        else:
            if len(targets.shape) > 1 and targets.shape[-1] > 1:
                targets = tlx.argmax(targets, axis=-1)
            targets = tlx.cast(targets, tlx.int64)
            loss = tlx.losses.softmax_cross_entropy_with_logits(preds, targets)
            return float(tlx.convert_to_numpy(loss))

    def update_precomputed(self, loss_val, n):
        self.total_loss += float(loss_val) * int(n)
        self.total_samples += int(n)
        return float(loss_val)

    def update_masked(self, preds, targets, mask):
        if tlx.BACKEND == 'torch':
            import torch
            mask_bool = mask.bool() if not mask.dtype == torch.bool else mask
            n_valid = int(mask_bool.sum().item())
            if n_valid == 0:
                return 0.0
            valid_preds = preds[mask_bool]
            valid_targets = targets[mask_bool]
            loss_val = self._compute_loss(valid_preds, valid_targets)
            return self.update_precomputed(loss_val, n_valid)

        mask_np = tlx.convert_to_numpy(mask).astype(bool)
        n_valid = int(mask_np.sum())
        if n_valid == 0:
            return 0.0
        preds_np = tlx.convert_to_numpy(preds)[mask_np]
        targets_np = tlx.convert_to_numpy(targets)[mask_np]
        loss_val = self._compute_loss(
            tlx.convert_to_tensor(preds_np),
            tlx.convert_to_tensor(targets_np),
        )
        return self.update_precomputed(loss_val, n_valid)

    def compute(self):
        r"""Compute the accumulated mean loss."""
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples

    def reset(self):
        r"""Reset the accumulator."""
        self.total_loss = 0.0
        self.total_samples = 0


class KLDMetric:
    r"""Accumulator for KL-divergence loss with running mean.

    Uses ``F.kl_div`` semantics: KL(target || softmax(preds)).
    """
    def __init__(self):
        self.total_loss = 0.0
        self.total_samples = 0

    def update(self, preds, targets):
        r"""Update metric.

        Parameters
        ----------
        preds : tensor
            Logits ``(N, C)``.
        targets : tensor
            One-hot target probabilities ``(N, C)``.
        """
        return self.update_precomputed(self._compute_loss(preds, targets), preds.shape[0])

    def _compute_loss(self, preds, targets):
        if tlx.BACKEND == 'torch':
            import torch.nn.functional as F
            log_preds = F.log_softmax(preds, dim=-1)
            targets_f = targets.float()
            per_sample = F.kl_div(log_preds, targets_f, reduction='none')
            per_sample = per_sample.sum(dim=-1)
            per_sample = per_sample.masked_fill(per_sample.isnan(), 0.0)
            return float(per_sample.mean().item())

        preds_np = tlx.convert_to_numpy(preds)
        targets_np = tlx.convert_to_numpy(targets)
        preds_max = preds_np.max(axis=-1, keepdims=True)
        exp_preds = np.exp(preds_np - preds_max)
        log_probs = np.log(exp_preds / exp_preds.sum(axis=-1, keepdims=True) + 1e-10)
        targets_safe = np.clip(targets_np, 1e-10, None)
        kl = np.sum(targets_np * (np.log(targets_safe) - log_probs), axis=-1)
        kl = np.where(np.isnan(kl), 0.0, kl)
        return float(kl.mean()) if kl.size > 0 else 0.0

    def update_precomputed(self, loss_val, n):
        self.total_loss += float(loss_val) * int(n)
        self.total_samples += int(n)
        return float(loss_val)

    def update_masked(self, preds, targets, mask):
        if tlx.BACKEND == 'torch':
            import torch
            mask_bool = mask.bool() if not mask.dtype == torch.bool else mask
            n_valid = int(mask_bool.sum().item())
            if n_valid == 0:
                return 0.0
            valid_preds = preds[mask_bool]
            valid_targets = targets[mask_bool]
            loss_val = self._compute_loss(valid_preds, valid_targets)
            return self.update_precomputed(loss_val, n_valid)

        mask_np = tlx.convert_to_numpy(mask).astype(bool)
        n_valid = int(mask_np.sum())
        if n_valid == 0:
            return 0.0
        preds_np = tlx.convert_to_numpy(preds)[mask_np]
        targets_np = tlx.convert_to_numpy(targets)[mask_np]
        loss_val = self._compute_loss(
            tlx.convert_to_tensor(preds_np),
            tlx.convert_to_tensor(targets_np),
        )
        return self.update_precomputed(loss_val, n_valid)

    def compute(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0


class TrainLossDiscrete(tlx.nn.Module):
    r"""Training loss for DeFoG: weighted sum of node, edge, and global losses.

    ``loss = loss(X) + lambda_train[0] * loss(E) + lambda_train[1] * loss(y)``

    Parameters
    ----------
    lambda_train : list
        Two-element list ``[lambda_E, lambda_y]``. Default ``[5, 0]``.
    kld : bool
        If True, use KL-divergence instead of cross-entropy for node/edge losses.
    """
    def __init__(self, lambda_train=None, kld=False, name=None):
        super().__init__(name=name)
        if lambda_train is None:
            lambda_train = [5.0, 0.0]
        self.lambda_train = lambda_train

        LossCls = KLDMetric if kld else CrossEntropyMetric
        self.node_loss = LossCls()
        self.edge_loss = LossCls()
        self.y_loss = CrossEntropyMetric()

    def forward(self, pred_X, pred_E, pred_y, true_X, true_E, true_y):
        r"""Compute the training loss.

        Parameters
        ----------
        pred_X : tensor
            Predicted node logits ``(bs, n, dx)``.
        pred_E : tensor
            Predicted edge logits ``(bs, n, n, de)``.
        pred_y : tensor
            Predicted global logits ``(bs, dy)``.
        true_X : tensor
            True node features ``(bs, n, dx)`` (one-hot).
        true_E : tensor
            True edge features ``(bs, n, n, de)`` (one-hot).
        true_y : tensor
            True global features ``(bs, dy)`` (one-hot).

        Returns
        -------
        tensor
            Scalar loss value.
        """
        bs = pred_X.shape[0]
        n = pred_X.shape[1]
        dx = pred_X.shape[2]
        de = pred_E.shape[3]

        pred_X_flat = tlx.reshape(pred_X, [-1, dx])
        true_X_flat = tlx.reshape(true_X, [-1, dx])

        pred_E_flat = tlx.reshape(pred_E, [-1, de])
        true_E_flat = tlx.reshape(true_E, [-1, de])

        x_mask = tlx.reduce_sum(tlx.abs(true_X_flat), axis=-1) > 0
        e_mask = tlx.reduce_sum(tlx.abs(true_E_flat), axis=-1) > 0

        loss_X = self._masked_loss(self.node_loss, pred_X_flat, true_X_flat, x_mask)
        loss_E = self._masked_loss(self.edge_loss, pred_E_flat, true_E_flat, e_mask)

        if pred_y.shape[-1] > 0 and true_y is not None and true_y.shape[-1] > 0:
            pred_y_detached = pred_y.detach() if hasattr(pred_y, 'detach') else pred_y
            true_y_detached = true_y.detach() if hasattr(true_y, 'detach') else true_y
            self.y_loss.update(pred_y_detached, true_y_detached)
            true_y_labels = tlx.argmax(true_y, axis=-1)
            true_y_labels = tlx.cast(true_y_labels, tlx.int64)
            loss_y = tlx.losses.softmax_cross_entropy_with_logits(pred_y, true_y_labels)
        else:
            loss_y = tlx.convert_to_tensor(0.0)

        total_loss = loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y
        return total_loss

    def _masked_loss(self, metric, preds, targets, mask):
        """Compute loss only for non-masked (valid) positions."""
        if tlx.BACKEND == 'torch':
            import torch
            mask_bool = mask.bool() if not mask.dtype == torch.bool else mask
            if mask_bool.sum() == 0:
                return torch.tensor(0.0, device=preds.device, requires_grad=True)
            valid_preds = preds[mask_bool]
            valid_targets = targets[mask_bool]

            detached_preds = valid_preds.detach() if hasattr(valid_preds, 'detach') else valid_preds
            detached_targets = valid_targets.detach() if hasattr(valid_targets, 'detach') else valid_targets

            if isinstance(metric, KLDMetric):
                log_preds = torch.nn.functional.log_softmax(valid_preds, dim=-1)
                loss = torch.nn.functional.kl_div(
                    log_preds,
                    valid_targets.float(),
                    reduction='batchmean'
                )
                metric.update_precomputed(float(loss.detach().item()), int(mask_bool.sum().item()))
                return loss
            else:
                valid_targets_labels = tlx.argmax(valid_targets, axis=-1)
                valid_targets_labels = tlx.cast(valid_targets_labels, tlx.int64)
                loss = torch.nn.functional.cross_entropy(valid_preds, valid_targets_labels)
                metric.update_precomputed(float(loss.detach().item()), int(mask_bool.sum().item()))
                return loss
        else:
            mask_float = tlx.cast(mask, tlx.float32)
            n_valid = tlx.reduce_sum(mask_float) + 1e-8

            if isinstance(metric, KLDMetric):
                preds_np = tlx.convert_to_numpy(preds)
                targets_np = tlx.convert_to_numpy(targets)
                mask_np = tlx.convert_to_numpy(mask).astype(bool)
                valid_preds = tlx.convert_to_tensor(preds_np[mask_np])
                valid_targets = tlx.convert_to_tensor(targets_np[mask_np])
                loss_val = metric.update_precomputed(metric._compute_loss(valid_preds, valid_targets), int(mask_np.sum()))
                return tlx.convert_to_tensor(loss_val)
            else:
                true_labels = tlx.argmax(targets, axis=-1)
                true_labels = tlx.cast(true_labels, tlx.int64)
                per_sample_loss = tlx.losses.softmax_cross_entropy_with_logits(
                    preds, true_labels
                )
                if len(per_sample_loss.shape) == 0:
                    metric.update_precomputed(float(tlx.convert_to_numpy(per_sample_loss)), 1)
                    return per_sample_loss
                masked_loss = per_sample_loss * mask_float
                loss = tlx.reduce_sum(masked_loss) / n_valid
                metric.update_precomputed(float(tlx.convert_to_numpy(loss)), int(tlx.convert_to_numpy(tlx.reduce_sum(mask_float))))
                return loss

    def log_epoch_metrics(self):
        """Return epoch-level loss averages as a dict."""
        return {
            'x_CE': self.node_loss.compute(),
            'E_CE': self.edge_loss.compute(),
            'y_CE': self.y_loss.compute(),
        }

    def reset(self):
        r"""Reset all metric accumulators."""
        self.node_loss.reset()
        self.edge_loss.reset()
        self.y_loss.reset()
