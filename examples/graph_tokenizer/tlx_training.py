"""Eager TLX gradient accumulation, separated from immediate optimizer updates.

TLX WithGrad lacks torch support and TrainOneStep applies each microbatch
immediately. Its torch optimizers also read parameter.grad instead of the
supplied pairs; keep those backend details here, outside the experiment flow.
"""
from contextlib import nullcontext

import tensorlayerx as tlx
from tensorlayerx.model import WithLoss


def trainable_weights(model):
    if hasattr(model, "trainable_weights"):
        return list(model.trainable_weights)
    return list(model.parameters())  # Also accept native torch test modules.


def assign(weight, value):
    if tlx.BACKEND == "tensorflow":
        weight.assign(value)
    else:
        import torch
        with torch.no_grad():
            weight.copy_(value)


def gradient_context():
    if tlx.BACKEND == "tensorflow":
        import tensorflow as tf
        return tf.GradientTape()
    return nullcontext()


class GraphTokenLoss(WithLoss):
    def __init__(self, model, compute):
        super().__init__(backbone=model, loss_fn=compute)
        self.target_count = 0

    def forward(self, data, label):
        loss, self.target_count = self._loss_fn(data, label)
        return loss


class GradientAccumulator:
    def __init__(self, model, optimizer, check_finite):
        self.weights = trainable_weights(model)
        self.optimizer = optimizer
        self.check_finite = check_finite
        self.gradients = [None] * len(self.weights)

    def collect(self, loss, tape):
        if tlx.BACKEND == "tensorflow":
            gradients = tape.gradient(loss, self.weights)
            # Embedding gradients can be IndexedSlices; accumulation needs dense
            # tensors just as the native TLX torch embeddings use dense grads.
            import tensorflow as tf
            gradients = [None if g is None else tf.convert_to_tensor(g) for g in gradients]
        else:
            gradients = self.optimizer.gradient(loss, self.weights)
        for index, gradient in enumerate(gradients):
            if gradient is not None:
                gradient = tlx.detach(gradient) + tlx.zeros_like(gradient)
                previous = self.gradients[index]
                self.gradients[index] = gradient if previous is None else previous + gradient

    def step(self, count, max_norm, scheduler, context):
        if count:
            pairs = []
            for index, (gradient, weight) in enumerate(zip(self.gradients, self.weights)):
                if gradient is not None:
                    gradient = gradient / count
                    self.check_finite(tlx, gradient, f"{context} gradient weight[{index}]")
                    pairs.append((gradient, weight))
            # TLX torch ClipByGlobalNorm is unimplemented. Compute the same
            # global L2 clipping with TLX ops and check the norm before scaling.
            norm = tlx.sqrt(tlx.add_n([tlx.reduce_sum(g * g) for g, _ in pairs]))
            self.check_finite(tlx, norm, f"{context} gradient norm")
            scale = min(1.0, float(max_norm) / (float(tlx.convert_to_numpy(norm)) + 1e-6))
            pairs = [(gradient * scale, weight) for gradient, weight in pairs]
            for index, (gradient, _) in enumerate(pairs):
                self.check_finite(tlx, gradient, f"{context} clipped gradient weight[{index}]")
            for index, weight in enumerate(self.weights):
                self.check_finite(tlx, weight, f"{context} weight[{index}]")
            decay = getattr(self.optimizer, "graph_tokenizer_weight_decay", 0.0)
            if decay:
                # TLX Adam's weight_decay is coupled L2; retain the trainer's
                # existing AdamW semantics with decoupled decay on active weights.
                decayed = [(weight, weight * (1.0 - float(self.optimizer.lr) * decay))
                           for _, weight in pairs]
                for index, (_, value) in enumerate(decayed):
                    self.check_finite(tlx, value, f"{context} weight decay weight[{index}]")
                for weight, value in decayed:
                    assign(weight, value)
            if tlx.BACKEND == "torch":
                for weight in self.weights:
                    weight.grad = None
                for gradient, weight in pairs:
                    weight.grad = gradient
            self.optimizer.apply_gradients(pairs)
            for index, weight in enumerate(self.weights):
                self.check_finite(tlx, weight, f"{context} updated weight[{index}]")
            scheduler.step()
        self.gradients = [None] * len(self.weights)
        if tlx.BACKEND == "torch":
            for weight in self.weights:
                weight.grad = None


class WarmupCosineScheduler:
    def __init__(self, optimizer, multiplier):
        self.optimizer, self.multiplier = optimizer, multiplier
        self.base_lr = float(optimizer.lr)
        self.steps = 0
        self._set_lr()

    def _set_lr(self):
        self.optimizer.lr = self.base_lr * self.multiplier(self.steps)
        if tlx.BACKEND == "tensorflow":
            self.optimizer.adam.learning_rate.assign(self.optimizer.lr)

    def step(self):
        self.steps += 1
        self._set_lr()
