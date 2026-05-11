import os
os.environ['TL_BACKEND'] = 'torch'  # Set backend before importing TLX/GammaGL

import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch
from torch_geometric.data import Data as PyGData
import numpy as np
import time
import logging
import os
import pickle

from gammagl.models.rgt import RGT
from utils import load_data, get_eigen_tokens, save_model, load_model, create_logger
from gammagl.loader.rgt_loader import ExtractNodeLoader


class PretrainLoss(tlx.model.WithLoss):
    def __init__(self, model, proj, configs):
        super(PretrainLoss, self).__init__(backbone=model, loss_fn=None)
        self.proj = proj
        self.configs = configs

    def forward(self, batch, label):
        x_tuple = self.backbone_network(batch)
        loss, E = self.backbone_network.loss(x_tuple)

        x_hat = self.proj(E)

        if callable(batch.tokens):
            n_id = batch.n_id if hasattr(batch, 'n_id') and batch.n_id is not None else tlx.arange(0, int(batch.num_nodes), dtype=tlx.int64)
            tokens = batch.tokens(n_id)
        elif isinstance(batch.tokens, np.ndarray):
            tokens = batch.tokens
        elif hasattr(batch.tokens, 'shape') and len(batch.tokens.shape) >= 2:
            if batch.tokens.shape[0] == int(batch.num_nodes):
                tokens = batch.tokens
            elif hasattr(batch, 'n_id') and batch.n_id is not None:
                tokens = batch.tokens[batch.n_id]
            else:
                tokens = batch.tokens
        else:
            tokens = batch.tokens

        if torch.is_tensor(tokens):
            tokens_clean = tokens
        elif isinstance(tokens, np.ndarray):
            tokens_clean = torch.tensor(tokens, dtype=torch.float32)
        else:
            tokens_clean = torch.tensor(tlx.convert_to_numpy(tokens), dtype=torch.float32)

        loss_recon = tlx.reduce_mean(tlx.pow(x_hat - tokens_clean, 2))
        loss = loss + self.configs.weight_rec * loss_recon

        return loss


class Pretrain(object):
    def __init__(self, configs, model_path):
        self.configs = configs
        self.model_path = model_path

        if configs.use_gpu and configs.gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(configs.gpu)
            self.device = torch.device(f'cuda:{configs.gpu}')
        else:
            self.device = torch.device('cpu')

        self.logger = create_logger(
            self.configs.log_dir + self.configs.log_name
        )

        self.logger.info("=" * 80)
        self.logger.info("PRETRAINING SESSION STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration: {vars(configs)}")
        self.logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.proj = nn.Sequential(
            tlx.layers.Linear(in_features=3 * configs.embed_dim, out_features=configs.hidden_dim),
            tlx.nn.ReLU(),
            tlx.layers.Linear(in_features=configs.hidden_dim, out_features=configs.embed_dim)
        )

        # Pretrain on multiple datasets
        self.datasets = []
        self.dataloaders = []
        self.logger.info(f"Loading {len(configs.pretrain_dataset)} pretraining datasets: {configs.pretrain_dataset}")

        cache_dir = os.path.join(configs.root_path, '.pretrain_cache')
        os.makedirs(cache_dir, exist_ok=True)

        for data_name in configs.pretrain_dataset:
            self.logger.info(f"Loading dataset: {data_name}")
            t0 = time.time()

            cache_path = os.path.join(cache_dir, f"{data_name}_pyg_data.pkl")
            if os.path.exists(cache_path):
                self.logger.info(f"Loading cached data from {cache_path}")
                with open(cache_path, 'rb') as f:
                    pyg_data = pickle.load(f)
                num_nodes = pyg_data.num_nodes if pyg_data.num_nodes is not None else pyg_data.x.shape[0]
                num_edges = pyg_data.edge_index.shape[1]
                self.logger.info(f"Dataset {data_name} loaded from cache in {time.time()-t0:.2f}s (num_nodes={num_nodes}, num_edges={num_edges})")
            else:
                dataset = load_data(configs.root_path, data_name)
                data = dataset[0]
                data.tokens = get_eigen_tokens(data, configs.embed_dim)
                num_nodes = data.num_nodes if data.num_nodes is not None else data.x.shape[0]
                num_edges = data.edge_index.shape[1] if hasattr(data.edge_index, 'shape') and len(data.edge_index.shape) == 2 else 0
                self.logger.info(f"Dataset {data_name} loaded in {time.time()-t0:.2f}s (num_nodes={num_nodes}, num_edges={num_edges})")

                # Convert to PyTorch tensors for PyG NeighborLoader compatibility
                if hasattr(data.x, 'cpu'):
                    x_tensor = data.x.cpu().contiguous()
                elif hasattr(data.x, 'numpy'):
                    x_tensor = torch.from_numpy(np.array(data.x)).contiguous()
                else:
                    x_tensor = torch.tensor(np.array(data.x)).contiguous()
                if hasattr(data.edge_index, 'cpu'):
                    edge_index_tensor = data.edge_index.cpu().contiguous()
                elif hasattr(data.edge_index, 'numpy'):
                    edge_index_tensor = torch.from_numpy(np.array(data.edge_index)).contiguous()
                else:
                    edge_index_tensor = torch.tensor(np.array(data.edge_index)).contiguous()

                tokens_np = None
                if data.tokens is not None:
                    if callable(data.tokens):
                        tokens_np = data.tokens(np.arange(num_nodes))
                        if hasattr(tokens_np, 'cpu'):
                            tokens_np = tokens_np.cpu()
                            if hasattr(tokens_np, 'numpy'):
                                tokens_np = tokens_np.numpy()
                            elif hasattr(tokens_np, 'detach'):
                                tokens_np = tokens_np.detach().cpu().numpy()
                        elif hasattr(tokens_np, 'detach'):
                            tokens_np = tokens_np.detach().cpu().numpy()
                        elif hasattr(tokens_np, 'numpy'):
                            tokens_np = tokens_np.numpy()
                        tokens_np = np.array(tokens_np)
                    elif hasattr(data.tokens, 'cpu'):
                        tokens_np = data.tokens.cpu().numpy()
                    elif hasattr(data.tokens, 'numpy'):
                        tokens_np = data.tokens.numpy()
                    else:
                        tokens_np = np.array(data.tokens)

                y_np = None
                if hasattr(data, 'y') and data.y is not None:
                    if hasattr(data.y, 'cpu'):
                        y_np = data.y.cpu().numpy()
                    elif hasattr(data.y, 'numpy'):
                        y_np = data.y.numpy()
                    else:
                        y_np = np.array(data.y)

                pyg_data = PyGData(
                    x=x_tensor,
                    edge_index=edge_index_tensor,
                    y=torch.tensor(y_np) if y_np is not None else None,
                    num_nodes=num_nodes,
                    tokens=tokens_np
                )

                self.logger.info(f"Saving cached data to {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(pyg_data, f)

            self.logger.info(f"Creating ExtractNodeLoader for {data_name} (batch_size={configs.batch_size}, num_neighbors={configs.num_neighbors})")
            dataloader = ExtractNodeLoader(
                data=pyg_data,
                num_neighbors=configs.num_neighbors,
                batch_size=configs.batch_size,
                replace=False,
                capacity=configs.capacity
            )
            self.datasets.append((data_name, pyg_data))
            self.dataloaders.append(dataloader)
            self.logger.info(f"DataLoader for {data_name} created successfully")

        # Use first dataset for model input dimension
        first_data = self.datasets[0][1] if self.datasets else None
        in_dim = first_data.x.shape[1] if first_data is not None else configs.embed_dim

        self.logger.info(f"Initializing RGT model (in_dim={in_dim}, hidden_dim={configs.hidden_dim}, embed_dim={configs.embed_dim}, n_layers={configs.n_layers})")
        configs.num_features = in_dim
        configs.d_hidden = configs.hidden_dim
        configs.num_hyp_dim = configs.embed_dim
        configs.num_sph_dim = configs.embed_dim
        configs.num_euc_dim = configs.embed_dim
        if not hasattr(configs, 'hyp_k') or configs.hyp_k is None:
            configs.hyp_k = 1.0
        self.model = RGT(args=configs)
        self.logger.info("RGT model initialized")
        self.logger.info("=" * 80)

    def pretrain(self, first_load=False, start_data=None):
        self.logger.info("=" * 80)
        self.logger.info("PRETRAINING LOOP STARTED")
        self.logger.info(f"Total pretrain_iters: {self.configs.pretrain_iters}")
        self.logger.info(f"Pretraining datasets: {[d[0] for d in self.datasets]}")
        self.logger.info(f"Epochs per dataset: {self.configs.pretrain_epochs}")
        self.logger.info("=" * 80)

        global_start = time.time()
        for e in range(self.configs.pretrain_iters):
            for i, (data_name, dataloader) in enumerate(zip([d[0] for d in self.datasets], self.dataloaders)):
                if start_data is not None and e == 0:
                    if data_name != start_data:
                        self.logger.info(f"Skipping dataset {data_name} (start_data={start_data})")
                        continue
                self.logger.info(f"----------Pretraining on {data_name}--------------")
                load = True
                if i == 0 and e == 0:
                    load = first_load
                self._train_step(dataloader, data_name, self.configs.pretrain_epochs, load)
            iter_elapsed = time.time() - global_start
            self.logger.info(f"[INFO] Pretrain Iteration {e+1} completed in {iter_elapsed:.2f}s ({iter_elapsed/60:.2f} min)")

        global_elapsed = time.time() - global_start
        self.logger.info("=" * 80)
        self.logger.info(f"PRETRAINING COMPLETED")
        self.logger.info(f"Total time: {global_elapsed:.2f}s ({global_elapsed/3600:.2f} hours)")
        self.logger.info("=" * 80)
        
        if self.model_path is not None:
            save_model(self.model.state_dict(), self.model_path)
            self.logger.info(f"[INFO] Model saved to {self.model_path}")
            print(f"Model saved to {self.model_path}")

    def _train_step(self, dataloader, data_name, train_epochs, load=False):
        self.logger.info("-" * 40)
        self.logger.info(f"[INFO] Training on dataset: {data_name}")
        self.logger.info(f"[INFO] Epochs: {train_epochs}")
        self.logger.info("-" * 40)

        if load and self.model_path is not None:
            load_path = self.configs.checkpoints + self.configs.pretrained_model_path + ".pt" if hasattr(self.configs, 'checkpoints') else self.model_path
            self.logger.info(f"---------------Loading pretrained models from {load_path}-------------")
            state_dict = load_model(load_path)
            self.model.load_state_dict(state_dict)

        path = self.configs.checkpoints + self.configs.pretrained_model_path if hasattr(self.configs, 'checkpoints') else self.model_path

        optimizer = tlx.optimizers.Adam(self.configs.lr, weight_decay=self.configs.weight_decay)

        for epoch in range(train_epochs):
            epoch_start = time.time()
            self.logger.info(f"[INFO] Epoch {epoch+1}/{train_epochs} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            self.model.is_train = True
            self.proj.is_train = True
            epoch_loss = []
            total_steps = len(dataloader)
            self.logger.info(f"[INFO] Total training steps: {total_steps}")
            if total_steps == 0:
                self.logger.warning(f"Dataset {data_name} has empty dataloader, skip.")
                continue
            
            # Use TLX TrainOneStep
            net_with_loss = PretrainLoss(self.model, self.proj, self.configs)
            train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, self.model.trainable_weights + self.proj.trainable_weights)

            for step, data in enumerate(dataloader):
                step_start = time.time()
                
                self.model.set_train()
                self.proj.set_train()
                loss = train_one_step(data, None)
                if isinstance(loss, np.ndarray):
                    loss_val = float(loss.item() if loss.ndim == 0 else loss)
                elif isinstance(loss, float):
                    loss_val = loss
                elif hasattr(loss, 'item'):
                    loss_val = float(loss.item())
                else:
                    loss_val = float(tlx.convert_to_numpy(loss))

                if np.isnan(loss_val):
                    self.logger.warning(f"NaN loss detected at step {step+1}, skipping batch")
                    continue
                
                epoch_loss.append(loss_val)

                if (step + 1) % max(1, total_steps // 5) == 0 or step == total_steps - 1:
                    step_elapsed = time.time() - step_start
                    self.logger.info(f"[INFO] Step {step+1}/{total_steps} | loss={loss_val:.6f} | step_time={step_elapsed:.2f}s")

            avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
            epoch_elapsed = time.time() - epoch_start
            self.logger.info(f"[INFO] Dataset {data_name}, Epoch {epoch+1}/{train_epochs} | avg_loss={avg_loss:.6f} | epoch_time={epoch_elapsed:.2f}s")
            self.logger.info(f"[INFO] Epoch {epoch+1} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            # Clear expensive structural cache once per epoch (not per step).
            dataloader.clear_cache()

            if path is not None:
                save_model(self.model.state_dict(), f"{path}_{epoch}.pt")
                self.logger.info(f"[INFO] Model saved to {path}_{epoch}.pt")

        if path is not None:
            save_model(self.model.state_dict(), f"{path}.pt")
            self.logger.info(f"[INFO] Final model saved to {path}.pt")

    def train_step(self, batch, optimizer=None):
        if optimizer is None:
            optimizer = tlx.optimizers.Adam(self.configs.lr, weight_decay=self.configs.weight_decay)

        net_with_loss = PretrainLoss(self.model, self.proj, self.configs)
        train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, self.model.trainable_weights + self.proj.trainable_weights)

        self.model.set_train()
        self.proj.set_train()
        loss = train_one_step(batch, None)
        loss_val = float(tlx.convert_to_numpy(loss))

        if np.isnan(loss_val):
            return loss_val, None

        return loss_val, None
