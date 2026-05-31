"""Multi-GPU training utilities for DeFoG.

Provides manual multi-GPU data parallelism via CUDA streams and threads,
extracted from defog_trainer.py.
"""

import copy
import threading
import torch


class MultiGPUTrainer:
    """Manual multi-GPU data parallelism using CUDA streams for parallel execution.

    Creates independent copies of the model+loss_wrapper on each GPU
    (via deepcopy), splits batches, runs forward+backward in parallel
    on separate CUDA streams, then averages gradients on the primary GPU.
    """
    def __init__(self, model, loss_wrapper, loss_fn, n_gpu, lr, weight_decay):
        self.n_gpu = n_gpu
        self.primary = 0
        self.models = [model]
        self.loss_wrappers = [loss_wrapper]

        # Ensure primary model is explicitly on cuda:0
        model.to('cuda:0')

        for i in range(1, n_gpu):
            m_copy = copy.deepcopy(model)
            m_copy.to(f'cuda:{i}')
            lw_copy = copy.deepcopy(loss_wrapper)
            lw_copy._backbone = m_copy
            self.models.append(m_copy)
            self.loss_wrappers.append(lw_copy)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay, amsgrad=True,
        )

        # Pre-allocate CUDA streams for each GPU
        self.streams = [torch.cuda.Stream(device=i) for i in range(n_gpu)]

    def train_step(self, data_dict):
        self.optimizer.zero_grad()

        # Split batch across GPUs
        X, E, y, mask = data_dict['X'], data_dict['E'], data_dict['y'], data_dict['node_mask']
        bs = X.shape[0]
        chunk_size = bs // self.n_gpu
        chunks = []
        start = 0
        for i in range(self.n_gpu):
            end = start + chunk_size + (1 if i < bs % self.n_gpu else 0)
            chunks.append((
                X[start:end].to(f'cuda:{i}', non_blocking=True),
                E[start:end].to(f'cuda:{i}', non_blocking=True),
                y[start:end].to(f'cuda:{i}', non_blocking=True),
                mask[start:end].to(f'cuda:{i}', non_blocking=True),
            ))
            start = end

        # Forward + backward on each GPU in parallel via CUDA streams
        per_gpu_losses = [None] * self.n_gpu
        per_gpu_grads = [None] * self.n_gpu
        errors = [None] * self.n_gpu

        def gpu_work(idx):
            try:
                stream = self.streams[idx]
                with torch.cuda.device(idx), torch.cuda.stream(stream):
                    Xi, Ei, yi, mi = chunks[idx]
                    d = {'X': Xi, 'E': Ei, 'y': yi, 'node_mask': mi}
                    loss = self.loss_wrappers[idx](d, None)
                    loss.backward()
                    per_gpu_losses[idx] = float(loss.item())
                    # Detach grads and move to primary GPU
                    grads = {}
                    for name, p in self.models[idx].named_parameters():
                        if p.grad is not None:
                            grads[name] = p.grad.to(f'cuda:{self.primary}', non_blocking=True).detach() / self.n_gpu
                    per_gpu_grads[idx] = grads
            except Exception as e:
                errors[idx] = e

        # Launch all GPU work in parallel via Python threads + CUDA streams
        threads = []
        for i in range(self.n_gpu):
            t = threading.Thread(target=gpu_work, args=(i,))
            t.start()
            threads.append(t)

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Check for errors
        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"GPU {i} failed: {err}") from err

        # Synchronize all streams to ensure computation is done
        for s in self.streams:
            s.synchronize()

        # Average loss
        avg_loss = sum(per_gpu_losses) / self.n_gpu

        # Accumulate all grads onto primary model
        torch.cuda.set_device(self.primary)
        for gpu_grads in per_gpu_grads:
            for name, p in self.models[self.primary].named_parameters():
                if name in gpu_grads:
                    if p.grad is None:
                        p.grad = gpu_grads[name]
                    else:
                        p.grad.add_(gpu_grads[name])

        # Optimizer step
        self.optimizer.step()

        # Sync weights to replicas
        with torch.no_grad():
            for i in range(1, self.n_gpu):
                for (_, p_src), (_, p_dst) in zip(
                    self.models[self.primary].named_parameters(),
                    self.models[i].named_parameters()
                ):
                    p_dst.data.copy_(p_src.data, non_blocking=True)
            # Synchronize weight copies
            for i in range(1, self.n_gpu):
                self.streams[i].synchronize()

        return avg_loss
