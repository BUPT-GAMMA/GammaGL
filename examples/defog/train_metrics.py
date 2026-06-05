import tensorlayerx as tlx
import numpy as np

class LossMetric:
    """Generic accumulator for losses with running mean."""
    def __init__(self):
        self.total_loss = 0.0
        self.total_samples = 0

    def update_precomputed(self, loss_val, n):
        if n > 0:
            self.total_loss += float(loss_val) * int(n)
            self.total_samples += int(n)
        return float(loss_val)

    def compute(self):
        if self.total_samples == 0: return 0.0
        return self.total_loss / self.total_samples

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0

class TrainLossDiscrete(tlx.nn.Module):
    r"""Training loss for DeFoG: weighted sum of node, edge, and global losses."""
    def __init__(self, lambda_train=None, kld=False, name=None):
        super().__init__(name=name)
        self.lambda_train = lambda_train if lambda_train else [5.0, 0.0]
        self.kld = kld
        self.node_loss = LossMetric()
        self.edge_loss = LossMetric()
        self.y_loss = LossMetric()

    def forward(self, pred_X, pred_E, pred_y, true_X, true_E, true_y):
        bs, n, dx = pred_X.shape[0], pred_X.shape[1], pred_X.shape[2]
        de = pred_E.shape[-1]

        pred_X_flat = tlx.reshape(pred_X, [-1, dx])
        true_X_flat = tlx.reshape(true_X, [-1, dx])
        pred_E_flat = tlx.reshape(pred_E, [-1, de])
        true_E_flat = tlx.reshape(true_E, [-1, de])

        x_mask = tlx.reduce_sum(tlx.abs(true_X_flat), axis=-1) > 0
        e_mask = tlx.reduce_sum(tlx.abs(true_E_flat), axis=-1) > 0

        loss_X = self._masked_loss(self.node_loss, pred_X_flat, true_X_flat, x_mask)
        loss_E = self._masked_loss(self.edge_loss, pred_E_flat, true_E_flat, e_mask)

        if pred_y is not None and pred_y.shape[-1] > 0 and true_y is not None and true_y.shape[-1] > 0:
            true_y_labels = tlx.cast(tlx.argmax(true_y, axis=-1), tlx.int64)
            per_sample_loss_y = tlx.losses.softmax_cross_entropy_with_logits(pred_y, true_y_labels)
            
            if len(per_sample_loss_y.shape) == 0:
                loss_y = per_sample_loss_y
            else:
                loss_y = tlx.reduce_mean(per_sample_loss_y)
            self.y_loss.update_precomputed(float(tlx.convert_to_numpy(loss_y)), bs)
        else:
            loss_y = tlx.convert_to_tensor(0.0)

        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def _masked_loss(self, metric, preds, targets, mask):
        mask_float = tlx.cast(mask, tlx.float32)
        n_valid = tlx.reduce_sum(mask_float) + 1e-8

        if self.kld:
            preds_max = tlx.reduce_max(preds, axis=-1, keepdims=True)
            exp_preds = tlx.exp(preds - preds_max)
            log_preds = (preds - preds_max) - tlx.log(tlx.reduce_sum(exp_preds, axis=-1, keepdims=True) + 1e-10)
            targets_safe = targets + 1e-10
            kl_per_sample = tlx.reduce_sum(targets * (tlx.log(targets_safe) - log_preds), axis=-1)
            loss = tlx.reduce_sum(kl_per_sample * mask_float) / n_valid
        else:
            true_labels = tlx.cast(tlx.argmax(targets, axis=-1), tlx.int64)
            per_sample_loss = tlx.losses.softmax_cross_entropy_with_logits(preds, true_labels)
            
            # Handling scalar vs vector reduction based on backend implementation
            if len(per_sample_loss.shape) == 0:
                loss = per_sample_loss
            else:
                loss = tlx.reduce_sum(per_sample_loss * mask_float) / n_valid
        
        metric_val = float(tlx.convert_to_numpy(loss))
        metric_n = int(tlx.convert_to_numpy(tlx.reduce_sum(mask_float)))
        metric.update_precomputed(metric_val, metric_n)
        return loss

    def log_epoch_metrics(self):
        return {'x_CE': self.node_loss.compute(), 'E_CE': self.edge_loss.compute(), 'y_CE': self.y_loss.compute()}

    def reset(self):
        self.node_loss.reset()
        self.edge_loss.reset()
        self.y_loss.reset()
