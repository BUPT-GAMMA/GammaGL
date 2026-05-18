import os

import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch
import numpy as np
import time
import logging

from gammagl.models.rgt import RGT, NodeClsHead, LinkPredHead, GraphClsHead, ShotNCHead
from rgt_mappings import class_maps
from utils import (
    EarlyStopping, act_fn, get_word2vec_dim,
    cal_accuracy, cal_AUC_AP, cal_F1,
    save_model, load_model,
    create_logger,
    load_data, input_dim_dict, class_num_dict, get_eigen_tokens,
)
from gammagl.loader.rgt_loader import ExtractNodeLoader, ExtractLinkLoader, ExtractGraphLoader
from gammagl.transforms import RandomLinkSplit
import gensim.downloader as api
import os
import re
from tqdm import tqdm




def safe_to_float(value):
    """Safely convert a value to float, handling both tensors and numpy types."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if hasattr(value, 'detach'):
        return float(value.detach().cpu().numpy())
    if hasattr(value, 'numpy'):
        return float(value.numpy())
    try:
        return float(tlx.convert_to_numpy(value))
    except Exception:
        pass
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)

def set_module_trainable(module, trainable: bool):
    """Best-effort trainable switch for TLX(torch) weights."""
    for w in getattr(module, "trainable_weights", []):
        try:
            w.requires_grad = bool(trainable)
        except Exception:
            pass
        try:
            w.stop_gradient = not bool(trainable)
        except Exception:
            pass
    if hasattr(module, "parameters"):
        try:
            for p in module.parameters():
                try:
                    p.requires_grad = bool(trainable)
                except Exception:
                    pass
        except Exception:
            pass

def select_head_trainable_weights(full_model, backbone, head_module):
    """Prefer head.trainable_weights; fallback to full-backbone diff."""
    head_weights = list(getattr(head_module, "trainable_weights", []) or [])
    if len(head_weights) > 0:
        return head_weights
    full_weights = list(getattr(full_model, "trainable_weights", []) or [])
    backbone_weights = list(getattr(backbone, "trainable_weights", []) or [])
    if len(full_weights) == 0:
        return []
    if len(backbone_weights) == 0:
        return full_weights
    backbone_ids = {id(w) for w in backbone_weights}
    return [w for w in full_weights if id(w) not in backbone_ids]

def warmup_node_head_only(nc_model, in_dim):
    """Build node head params without running full pretrained forward."""
    try:
        x = tlx.zeros((2, int(in_dim)), dtype=tlx.float32)
        edge_index = tlx.convert_to_tensor(np.array([[0, 1], [1, 0]], dtype=np.int64), dtype=tlx.int64)
        _ = nc_model.head(x, edge_index, 2)
    except Exception:
        pass

def warmup_linear_head_only(linear_head, in_dim):
    """Build linear head params with tiny fake input."""
    try:
        x = tlx.zeros((2, int(in_dim)), dtype=tlx.float32)
        _ = linear_head(x)
    except Exception:
        pass


def random_split(dataset, ratios):
    """Simple random split utility to replace torch.utils.data.random_split."""
    indices = np.random.permutation(len(dataset))
    split_sizes = [int(r * len(dataset)) for r in ratios]
    split_sizes[-1] = len(dataset) - sum(split_sizes[:-1])
    splits = []
    start = 0
    for size in split_sizes:
        splits.append(dataset[indices[start:start + size]])
        start += size
    return splits


class SupervisedExp(object):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        self.configs = configs
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        self.device = None

        if pretrained_model is None:
            pretrained_model = RGT(configs=self.configs, n_layers=self.configs.n_layers, in_dim=self.configs.embed_dim,
                                      hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                                      bias=self.configs.bias,
                                      dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
            if load:
                path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path) + ".pt"
                self.logger.info(f"---------------Loading pretrained models from {path}-------------")

                pretrained_dict = load_model(path)
                pretrained_model.load_state_dict(pretrained_dict)
        
        self.pretrained_model = pretrained_model
        
        if finetune:
            self.logger.info("----------Freezing weights-----------")
            self.pretrained_model._is_frozen = True
            set_module_trainable(self.pretrained_model, False)
        else:
            self.pretrained_model._is_frozen = False
            set_module_trainable(self.pretrained_model, True)

    def load_model(self):
        raise NotImplementedError

    def load_data(self, split):
        raise NotImplementedError

    def train(self):
        pass

    def train_step(self, data, optimizer):
        pass

    def val(self, val_loader):
        pass

    def test(self, test_loader):
        pass

    def cal_loss(self, **kwargs):
        raise NotImplementedError


class NodeClassificationLoss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(NodeClassificationLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        out = self.backbone_network(data)
        
        if isinstance(label, torch.Tensor):
            label_np = label.detach().cpu().numpy()
        elif hasattr(label, 'numpy'):
            label_np = label.numpy()
        elif hasattr(label, 'tolist'):
            label_np = np.array(label.tolist())
        else:
            label_np = np.array(label)
        
        if hasattr(data, 'n_id') and data.n_id is not None:
            if isinstance(data.n_id, torch.Tensor):
                n_id_np = data.n_id.detach().cpu().numpy()
            elif hasattr(data.n_id, 'numpy'):
                n_id_np = data.n_id.numpy()
            else:
                n_id_np = np.array(data.n_id)
            
            if n_id_np.max() < len(label_np):
                y = label_np[n_id_np]
            else:
                y = label_np[:out.shape[0]]
        else:
            y = label_np[:out.shape[0]]
        
        y = np.array(y).reshape(-1)
        valid_size = min(int(out.shape[0]), len(y))
        out = out[:valid_size]
        y = y[:valid_size]
        
        y_tensor = tlx.convert_to_tensor(y, dtype=tlx.int64)
        loss = tlx.reduce_mean(self._loss_fn(out, y_tensor))
        return loss

class NodeClassification(SupervisedExp):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        super(NodeClassification, self).__init__(configs, pretrained_model, load, finetune)
        self.nc_model = self.load_model()

    def load_model(self):
        nc_model = NodeClsHead(self.pretrained_model, 3 * self.configs.embed_dim + input_dim_dict[self.configs.dataset],
                               self.configs.nc_hidden_dim,
                               class_num_dict[self.configs.dataset],
                               self.configs.drop_edge,
                               self.configs.drop_feats)
        return nc_model

    def load_data(self, split: str):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        data = dataset[0]
        data.tokens = get_eigen_tokens(data, self.configs.embed_dim)
        train_loader = ExtractNodeLoader(data, input_nodes=data.train_mask, batch_size=self.configs.batch_size,
                                   num_neighbors=self.configs.num_neighbors,
                                   capacity=self.configs.capacity)
        val_loader = ExtractNodeLoader(data, input_nodes=data.val_mask, batch_size=self.configs.batch_size,
                                         num_neighbors=self.configs.num_neighbors,
                                         capacity=self.configs.capacity)
        test_loader = ExtractNodeLoader(data, input_nodes=data.test_mask, batch_size=self.configs.batch_size,
                                         num_neighbors=self.configs.num_neighbors,
                                         capacity=self.configs.capacity)
        if split == 'test':
            return test_loader
        return dataset, train_loader, val_loader, test_loader

    def train(self):
        self.logger.info("=" * 80)
        self.logger.info("NODE CLASSIFICATION TRAINING STARTED")
        self.logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Dataset: {self.configs.dataset}")
        self.logger.info(f"nc_epochs: {self.configs.nc_epochs}, lr_nc: {self.configs.lr_nc}, exp_iters: {self.configs.exp_iters}")
        self.logger.info("=" * 80)
        dataset, train_loader, val_loader, test_loader = self.load_data("train")
        self.logger.info(f"Data loaded: train_loader size={len(train_loader)}, val_loader size={len(val_loader)}, test_loader size={len(test_loader)}")

        total_test_acc = []
        total_test_weighted_f1 = []
        total_test_macro_f1 = []
        global_start = time.time()
        for t in range(self.configs.exp_iters):
            self.logger.info("-" * 60)
            self.logger.info(f"[INFO] Experiment iteration {t+1}/{self.configs.exp_iters} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("-" * 60)
            iter_start = time.time()
            self.nc_model = self.load_model()
            self.nc_model.is_train = True
            if len(train_loader) == 0:
                self.logger.warning("Empty train_loader detected, skip this iteration.")
                continue
            
            if self.configs.finetune:
                head_in_dim = 3 * self.configs.embed_dim + input_dim_dict[self.configs.dataset]
                warmup_node_head_only(self.nc_model, head_in_dim)
                trainable_weights = select_head_trainable_weights(self.nc_model, self.pretrained_model, self.nc_model.head)
                self.logger.info(f"Finetuning mode: optimizing only head weights ({len(trainable_weights)} parameters)")
            else:
                trainable_weights = self.nc_model.trainable_weights
                self.logger.info(f"Full training mode: optimizing all weights ({len(trainable_weights)} parameters)")
            if len(trainable_weights) == 0:
                trainable_weights = self.nc_model.trainable_weights
                self.logger.warning("Selected trainable_weights is empty, fallback to full model trainable_weights.")
            
            optimizer = tlx.optimizers.Adam(self.configs.lr_nc, weight_decay=self.configs.weight_decay_nc)
            early_stop = EarlyStopping(self.configs.patience)
            
            loss_fn = tlx.losses.softmax_cross_entropy_with_logits
            net_with_loss = NodeClassificationLoss(self.nc_model, loss_fn)
            train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, trainable_weights)

            for epoch in range(self.configs.nc_epochs):
                epoch_start = time.time()
                self.logger.info(f"[INFO] Epoch {epoch+1}/{self.configs.nc_epochs} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

                epoch_loss = []
                total_steps = len(train_loader)

                for step, data in enumerate(tqdm(train_loader)):
                    step_start = time.time()
                    
                    self.nc_model.set_train()
                    loss = train_one_step(data, data.y)
                    
                    loss_val = safe_to_float(loss)
                    if np.isnan(loss_val):
                        self.logger.warning(f"NaN loss detected at step {step+1}, skipping batch")
                        continue

                    epoch_loss.append(loss_val)

                    if (step + 1) % max(1, total_steps // 5) == 0 or step == total_steps - 1:
                        step_elapsed = time.time() - step_start
                        self.logger.info(f"[INFO] Step {step+1}/{total_steps} | loss={loss_val:.6f} | step_time={step_elapsed:.2f}s")

                if len(epoch_loss) == 0:
                    self.logger.warning(f"Epoch {epoch+1}: All batches had NaN loss, skipping epoch")
                    continue

                train_loss = np.mean(epoch_loss)
                epoch_elapsed = time.time() - epoch_start

                self.logger.info(f"[INFO] Epoch {epoch+1}/{self.configs.nc_epochs} | train_loss={train_loss:.6f} | epoch_time={epoch_elapsed:.2f}s")

                if epoch % self.configs.val_every == 0:
                    val_loss, val_acc, val_weighted_f1, val_macro_f1 = self.val(val_loader)
                    self.logger.info(f"[INFO] Epoch {epoch+1} | val_loss={val_loss:.6f}, "
                                     f"val_acc={val_acc * 100: .2f}%,"
                                     f"val_weighted_f1={val_weighted_f1 * 100: .2f},"
                                     f"val_macro_f1={val_macro_f1 * 100: .2f}%")
                    
                    if np.isnan(val_loss):
                        self.logger.warning("NaN validation loss detected, stopping training")
                        break
                    
                    early_stop(val_loss, self.nc_model, self.configs.checkpoints, self.configs.task_model_path)
                    if early_stop.early_stop:
                        self.logger.info("---------Early stopping--------")
                        print("---------Early stopping--------")
                        break
            test_acc, weighted_f1, macro_f1 = self.test(test_loader)
            self.logger.info(f"[INFO] Iteration {t+1} | test_acc={test_acc * 100: .2f}%, "
                             f"weighted_f1={weighted_f1 * 100: .2f},"
                             f"macro_f1={macro_f1 * 100: .2f}%")
            total_test_acc.append(test_acc)
            total_test_weighted_f1.append(weighted_f1)
            total_test_macro_f1.append(macro_f1)
            iter_elapsed = time.time() - iter_start
            self.logger.info(f"[INFO] Experiment iteration {t+1} completed in {iter_elapsed:.2f}s")

        global_elapsed = time.time() - global_start
        mean, std = np.mean(total_test_acc), np.std(total_test_acc)
        self.logger.info(f"[INFO] Final Evaluation Acc is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_weighted_f1), np.std(total_test_weighted_f1)
        self.logger.info(f"[INFO] Final Evaluation weighted F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_macro_f1), np.std(total_test_macro_f1)
        self.logger.info(f"[INFO] Final Evaluation macro F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")
        self.logger.info(f"[INFO] Total training time: {global_elapsed:.2f}s ({global_elapsed/3600:.2f} hours)")
        self.logger.info("=" * 80)

    def val(self, val_loader):
        self.nc_model.is_train = False
        val_loss = []
        trues = []
        preds = []
        for data in val_loader:
            out = self.nc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, data)
            val_loss.append(loss)
            trues.append(true)
            preds.append(pred)
        trues = np.concatenate(trues, axis=-1)
        preds = np.concatenate(preds, axis=-1)
        acc = cal_accuracy(preds, trues)
        weighted_f1, macro_f1 = cal_F1(preds, trues)
        self.nc_model.is_train = True
        return np.mean(val_loss), acc, weighted_f1, macro_f1

    def test(self, test_loader=None):
        test_loader = self.load_data("test") if test_loader is None else test_loader
        self.nc_model.is_train = False
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        task_dict = load_model(path)
        self.nc_model.load_state_dict(task_dict)
        trues = []
        preds = []
        for data in test_loader:
            out = self.nc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, data)
            trues.append(true)
            preds.append(pred)
        trues = np.concatenate(trues, axis=-1)
        preds = np.concatenate(preds, axis=-1)
        test_acc = cal_accuracy(preds, trues)
        weighted_f1, macro_f1 = cal_F1(preds, trues)
        self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                         f"weighted_f1={weighted_f1 * 100: .2f},"
                         f"macro_f1={macro_f1 * 100: .2f}%")
        return test_acc, weighted_f1, macro_f1

    def cal_loss(self, output, label, data):
        if isinstance(label, torch.Tensor):
            label_np = label.detach().cpu().numpy()
        elif hasattr(label, 'numpy'):
            label_np = label.numpy()
        elif hasattr(label, 'tolist'):
            label_np = np.array(label.tolist())
        else:
            label_np = np.array(label)
        
        if hasattr(data, 'n_id') and data.n_id is not None:
            if isinstance(data.n_id, torch.Tensor):
                n_id_np = data.n_id.detach().cpu().numpy()
            elif hasattr(data.n_id, 'numpy'):
                n_id_np = data.n_id.numpy()
            else:
                n_id_np = np.array(data.n_id)
            
            if n_id_np.max() < len(label_np):
                y = label_np[n_id_np]
            else:
                y = label_np[:output.shape[0]]
        else:
            y = label_np[:output.shape[0]]
        
        y = np.array(y).reshape(-1)
        valid_size = min(int(output.shape[0]), len(y))
        out = output[:valid_size]
        y = y[:valid_size]
            
        y_tensor = tlx.convert_to_tensor(y, dtype=tlx.int64)
        loss = tlx.reduce_mean(tlx.losses.softmax_cross_entropy_with_logits(out, y_tensor))
        pred = np.array(tlx.convert_to_numpy(tlx.argmax(out, axis=-1)))
        return safe_to_float(loss), pred, y


class LinkPredictionLoss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(LinkPredictionLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        pred, target = self.backbone_network(data)
        if isinstance(pred, torch.Tensor):
            pred_tensor = pred.to(torch.float32)
        else:
            pred_tensor = tlx.convert_to_tensor(pred, dtype=tlx.float32)
        label_tensor = tlx.convert_to_tensor(target, dtype=tlx.float32)
        loss = self._loss_fn(pred_tensor, label_tensor)
        return loss

class LinkPrediction(SupervisedExp):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        super(LinkPrediction, self).__init__(configs, pretrained_model, load, finetune)
        self.lp_model = self.load_model()

    def load_data(self, split):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        train_data, val_data, test_data = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=False,
                                                          add_negative_train_samples=False)(dataset[0])
        train_data.tokens = get_eigen_tokens(train_data, self.configs.embed_dim)
        val_data.tokens = train_data.tokens
        test_data.tokens = train_data.tokens
        train_loader = ExtractLinkLoader(train_data, batch_size=self.configs.batch_size,
                                   num_neighbors=self.configs.num_neighbors,
                                         neg_sampling_ratio=1.,
                                   capacity=self.configs.capacity)
        val_loader = ExtractLinkLoader(val_data, batch_size=self.configs.batch_size,
                                     num_neighbors=self.configs.num_neighbors,
                                       neg_sampling_ratio=1.,
                                     capacity=self.configs.capacity)
        test_loader = ExtractLinkLoader(test_data, batch_size=self.configs.batch_size,
                                     num_neighbors=self.configs.num_neighbors,
                                        neg_sampling_ratio=1.,
                                     capacity=self.configs.capacity)
        if split == 'test':
            return test_loader
        return train_data, train_loader, val_loader, test_loader

    def load_model(self):
        lp_model = LinkPredHead(self.pretrained_model,
                                3 * self.configs.embed_dim + input_dim_dict[self.configs.dataset],
                                self.configs.embed_dim_lp)
        return lp_model

    def train(self):
        self.logger.info("=" * 80)
        self.logger.info("LINK PREDICTION TRAINING STARTED")
        self.logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Dataset: {self.configs.dataset}")
        self.logger.info(f"lp_epochs: {self.configs.lp_epochs}, lr_lp: {self.configs.lr_lp}, exp_iters: {self.configs.exp_iters}")
        self.logger.info("=" * 80)
        train_data, train_loader, val_loader, test_loader = self.load_data(None)
        self.logger.info(f"Data loaded: train_loader size={len(train_loader)}, val_loader size={len(val_loader)}, test_loader size={len(test_loader)}")

        total_test_auc, total_test_ap = [], []
        global_start = time.time()
        for _ in range(self.configs.exp_iters):
            self.logger.info("-" * 60)
            self.logger.info(f"[INFO] Experiment iteration started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("-" * 60)
            iter_start = time.time()
            self.lp_model = self.load_model()
            self.lp_model.is_train = True
            if len(train_loader) == 0:
                self.logger.warning("Empty train_loader detected, skip this iteration.")
                continue
            
            if self.configs.finetune:
                head_in_dim = 3 * self.configs.embed_dim + input_dim_dict[self.configs.dataset]
                warmup_linear_head_only(self.lp_model.head, head_in_dim)
                trainable_weights = select_head_trainable_weights(self.lp_model, self.pretrained_model, self.lp_model.head)
                self.logger.info(f"Finetuning mode: optimizing only head weights ({len(trainable_weights)} parameters)")
            else:
                trainable_weights = self.lp_model.trainable_weights
                self.logger.info(f"Full training mode: optimizing all weights ({len(trainable_weights)} parameters)")
            if len(trainable_weights) == 0:
                trainable_weights = self.lp_model.trainable_weights
                self.logger.warning("Selected trainable_weights is empty, fallback to full model trainable_weights.")
            
            optimizer = tlx.optimizers.Adam(self.configs.lr_lp, weight_decay=self.configs.weight_decay_lp)
            early_stop = EarlyStopping(self.configs.patience)
            
            loss_fn = tlx.losses.sigmoid_cross_entropy
            net_with_loss = LinkPredictionLoss(self.lp_model, loss_fn)
            train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, trainable_weights)

            for epoch in range(self.configs.lp_epochs):
                epoch_start = time.time()
                self.logger.info(f"[INFO] Epoch {epoch+1}/{self.configs.lp_epochs} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

                epoch_loss = []
                total_steps = len(train_loader)

                for step, data in enumerate(tqdm(train_loader)):
                    step_start = time.time()
                    
                    self.lp_model.set_train()
                    loss = train_one_step(data, None)

                    loss_val = safe_to_float(loss)
                    epoch_loss.append(loss_val)

                    if (step + 1) % max(1, total_steps // 5) == 0 or step == total_steps - 1:
                        step_elapsed = time.time() - step_start
                        self.logger.info(f"[INFO] Step {step+1}/{total_steps} | loss={loss_val:.6f} | step_time={step_elapsed:.2f}s")

                if len(epoch_loss) == 0:
                    self.logger.warning(f"Epoch {epoch+1}: All batches had NaN loss, skipping epoch")
                    continue
                
                train_loss = np.mean(epoch_loss)
                epoch_elapsed = time.time() - epoch_start

                self.logger.info(
                    f"[INFO] Epoch {epoch+1}/{self.configs.lp_epochs} | train_loss={train_loss:.6f} | epoch_time={epoch_elapsed:.2f}s"
                )

                if epoch % self.configs.val_every == 0:
                    val_loss, val_auc, val_ap = self.val(val_loader)
                    self.logger.info(f"[INFO] Epoch {epoch+1} | val_loss={val_loss:.6f}, val_auc={val_auc * 100: .2f}%, "
                                f"val_ap={val_ap * 100: .2f}%")
                    early_stop(val_loss, self.lp_model, self.configs.checkpoints, self.configs.task_model_path)
                    if early_stop.early_stop:
                        self.logger.info("---------Early stopping--------")
                        print("---------Early stopping--------")
                        break
            test_auc, test_ap = self.test(test_loader)
            self.logger.info(f"[INFO] test_auc={test_auc * 100: .2f}%, "
                             f"test_ap={test_ap * 100: .2f}%")
            total_test_auc.append(test_auc)
            total_test_ap.append(test_ap)
            iter_elapsed = time.time() - iter_start
            self.logger.info(f"[INFO] Experiment iteration completed in {iter_elapsed:.2f}s")

        global_elapsed = time.time() - global_start
        mean_auc, std_auc = np.mean(total_test_auc), np.std(total_test_auc)
        mean_ap, std_ap = np.mean(total_test_ap), np.std(total_test_ap)
        self.logger.info(f"[INFO] Final Evaluation AUC={mean_auc * 100: .2f}% +- {std_auc * 100: .2f}%, "
                         f"Evaluation AP={mean_ap * 100: .2f}% +- {std_ap * 100: .2f}%")
        self.logger.info(f"[INFO] Total training time: {global_elapsed:.2f}s ({global_elapsed/3600:.2f} hours)")
        self.logger.info("=" * 80)

    def train_step(self, data, optimizer):
        pred, label = self.lp_model(data)
        label_tensor = tlx.convert_to_tensor(label, dtype=tlx.float32)
        pred_tensor = tlx.convert_to_tensor(pred, dtype=tlx.float32)
        loss = tlx.losses.sigmoid_cross_entropy(pred_tensor, label_tensor)

        return safe_to_float(loss), np.array(tlx.convert_to_numpy(pred)), np.array(tlx.convert_to_numpy(label))

    def val(self, val_loader):
        self.lp_model.is_train = False
        val_loss = []
        val_label = []
        val_pred = []
        for data in val_loader:
            pred, label = self.lp_model(data)
            label_tensor = tlx.convert_to_tensor(label, dtype=tlx.float32)
            pred_tensor = tlx.convert_to_tensor(pred, dtype=tlx.float32)
            loss = tlx.losses.sigmoid_cross_entropy(pred_tensor, label_tensor)
            val_loss.append(safe_to_float(loss))
            val_label.append(np.array(tlx.convert_to_numpy(label)))
            val_pred.append(np.array(tlx.convert_to_numpy(pred)))
        val_loss = np.mean(val_loss)
        val_pred = np.concatenate(val_pred, axis=-1)
        val_label = np.concatenate(val_label, axis=-1)
        val_auc, val_ap = cal_AUC_AP(val_pred, val_label)
        self.lp_model.is_train = True
        return val_loss, val_auc, val_ap

    def test(self, test_loader):
        test_loader = self.load_data("test") if test_loader is None else test_loader
        self.lp_model.is_train = False
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        task_dict = load_model(path)
        self.lp_model.load_state_dict(task_dict)
        test_label = []
        test_pred = []
        for data in test_loader:
            pred, label = self.lp_model(data)
            test_label.append(np.array(tlx.convert_to_numpy(label)))
            test_pred.append(np.array(tlx.convert_to_numpy(pred)))
        test_pred = np.concatenate(test_pred, axis=-1)
        test_label = np.concatenate(test_label, axis=-1)
        test_auc, test_ap = cal_AUC_AP(test_pred, test_label)
        return test_auc, test_ap


class GraphClassificationLoss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(GraphClassificationLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        out = self.backbone_network(data)
        batch_size = data.batch_size
        
        actual_size = out.shape[0]
        out = out[:batch_size] if actual_size >= batch_size else out
        
        if label is None:
            y = np.zeros(out.shape[0], dtype=np.int64)
        else:
            if hasattr(label, 'numpy'):
                label = label.numpy()
            elif hasattr(label, 'detach'):
                label = label.detach().cpu().numpy()
            else:
                label = np.array(label)
            y = np.array(label[:out.shape[0]]).reshape(-1)
            
        y_tensor = tlx.convert_to_tensor(y, dtype=tlx.int64)
        loss = tlx.reduce_mean(self._loss_fn(out, y_tensor))
        return loss

class GraphClassification(SupervisedExp):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        super(GraphClassification, self).__init__(configs, pretrained_model, load, finetune)
        self.gc_model = self.load_model()

    def load_model(self):
        gc_model = GraphClsHead(self.pretrained_model, 2 * self.configs.embed_dim + input_dim_dict[self.configs.dataset],
                               self.configs.nc_hidden_dim,
                               class_num_dict[self.configs.dataset],
                               self.configs.drop_feats)
        return gc_model

    def load_data(self, split: str):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        train, val, test = random_split(dataset, [0.7, 0.1, 0.2])
        train_loader = ExtractGraphLoader(train, batch_size=self.configs.batch_size,
                                   capacity=self.configs.capacity)
        val_loader = ExtractGraphLoader(val, batch_size=self.configs.batch_size,
                                         capacity=self.configs.capacity)
        test_loader = ExtractGraphLoader(test, batch_size=self.configs.batch_size,
                                         capacity=self.configs.capacity)
        if split == 'test':
            return test_loader
        return dataset, train_loader, val_loader, test_loader

    def train(self):
        self.logger.info("=" * 80)
        self.logger.info("GRAPH CLASSIFICATION TRAINING STARTED")
        self.logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Dataset: {self.configs.dataset}")
        self.logger.info(f"gc_epochs: {self.configs.gc_epochs}, lr_gc: {self.configs.lr_gc}, exp_iters: {self.configs.exp_iters}")
        self.logger.info("=" * 80)
        dataset, train_loader, val_loader, test_loader = self.load_data("train")
        self.logger.info(f"Data loaded: train_loader size={len(train_loader)}, val_loader size={len(val_loader)}, test_loader size={len(test_loader)}")

        total_test_acc = []
        total_test_weighted_f1 = []
        total_test_macro_f1 = []
        global_start = time.time()
        for t in range(self.configs.exp_iters):
            self.logger.info("-" * 60)
            self.logger.info(f"[INFO] Experiment iteration {t+1}/{self.configs.exp_iters} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("-" * 60)
            iter_start = time.time()
            self.gc_model = self.load_model()
            self.gc_model.is_train = True
            if len(train_loader) == 0:
                self.logger.warning("Empty train_loader detected, skip this iteration.")
                continue
            optimizer = tlx.optimizers.Adam(self.configs.lr_gc, weight_decay=self.configs.weight_decay_gc)
            early_stop = EarlyStopping(self.configs.patience)
            
            if self.configs.finetune:
                head_in_dim = 2 * self.configs.embed_dim + input_dim_dict[self.configs.dataset]
                warmup_linear_head_only(self.gc_model.head, head_in_dim)
                trainable_weights = select_head_trainable_weights(self.gc_model, self.pretrained_model, self.gc_model.head)
                self.logger.info(f"Finetuning mode: optimizing only head weights ({len(trainable_weights)} parameters)")
            else:
                trainable_weights = self.gc_model.trainable_weights
                self.logger.info(f"Full training mode: optimizing all weights ({len(trainable_weights)} parameters)")
            if len(trainable_weights) == 0:
                trainable_weights = self.gc_model.trainable_weights
                self.logger.warning("Selected trainable_weights is empty, fallback to full model trainable_weights.")
            
            loss_fn = tlx.losses.softmax_cross_entropy_with_logits
            net_with_loss = GraphClassificationLoss(self.gc_model, loss_fn)
            train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, trainable_weights)

            for epoch in range(self.configs.gc_epochs):
                epoch_start = time.time()
                self.logger.info(f"[INFO] Epoch {epoch+1}/{self.configs.gc_epochs} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

                epoch_loss = []
                total_steps = len(train_loader)

                for step, data in enumerate(tqdm(train_loader)):
                    step_start = time.time()
                    data.tokens = get_eigen_tokens(data, self.configs.embed_dim)
                    data.x = tlx.ones((data.num_nodes, 128))
                    data.n_id = tlx.arange(start=0, limit=int(data.num_nodes), delta=1, dtype=tlx.int64)
                    
                    self.gc_model.set_train()
                    loss = train_one_step(data, data.y)

                    epoch_loss.append(safe_to_float(loss))

                    if (step + 1) % max(1, total_steps // 5) == 0 or step == total_steps - 1:
                        step_elapsed = time.time() - step_start
                        self.logger.info(f"[INFO] Step {step+1}/{total_steps} | loss={loss:.6f} | step_time={step_elapsed:.2f}s")

                if len(epoch_loss) == 0:
                    self.logger.warning(f"Epoch {epoch+1}: All batches had NaN loss, skipping epoch")
                    continue

                train_loss = np.mean(epoch_loss)
                epoch_elapsed = time.time() - epoch_start

                self.logger.info(f"[INFO] Epoch {epoch+1}/{self.configs.gc_epochs} | train_loss={train_loss:.6f} | epoch_time={epoch_elapsed:.2f}s")

                if epoch % self.configs.val_every == 0:
                    val_loss, val_acc, val_weighted_f1, val_macro_f1 = self.val(val_loader)
                    self.logger.info(f"[INFO] Epoch {epoch+1} | val_loss={val_loss:.6f}, "
                                     f"val_acc={val_acc * 100: .2f}%,"
                                     f"val_weighted_f1={val_weighted_f1 * 100: .2f},"
                                     f"val_macro_f1={val_macro_f1 * 100: .2f}%")
                    early_stop(val_loss, self.gc_model, self.configs.checkpoints, self.configs.task_model_path)
                    if early_stop.early_stop:
                        self.logger.info("---------Early stopping--------")
                        print("---------Early stopping--------")
                        break
            test_acc, weighted_f1, macro_f1 = self.test(test_loader)
            self.logger.info(f"[INFO] Iteration {t+1} | test_acc={test_acc * 100: .2f}%, "
                             f"weighted_f1={weighted_f1 * 100: .2f},"
                             f"macro_f1={macro_f1 * 100: .2f}%")
            total_test_acc.append(test_acc)
            total_test_weighted_f1.append(weighted_f1)
            total_test_macro_f1.append(macro_f1)
            iter_elapsed = time.time() - iter_start
            self.logger.info(f"[INFO] Experiment iteration {t+1} completed in {iter_elapsed:.2f}s")

        global_elapsed = time.time() - global_start
        mean, std = np.mean(total_test_acc), np.std(total_test_acc)
        self.logger.info(f"[INFO] Final Evaluation Acc is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_weighted_f1), np.std(total_test_weighted_f1)
        self.logger.info(f"[INFO] Final Evaluation weighted F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_macro_f1), np.std(total_test_macro_f1)
        self.logger.info(f"[INFO] Final Evaluation macro F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")
        self.logger.info(f"[INFO] Total training time: {global_elapsed:.2f}s ({global_elapsed/3600:.2f} hours)")
        self.logger.info("=" * 80)

    def val(self, val_loader):
        self.gc_model.is_train = False
        val_loss = []
        trues = []
        preds = []
        for data in val_loader:
            data.tokens = get_eigen_tokens(data, self.configs.embed_dim)
            data.x = tlx.ones((data.num_nodes, 128))
            data.n_id = tlx.arange(start=0, limit=int(data.num_nodes), delta=1, dtype=tlx.int64)
            out = self.gc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, data)
            val_loss.append(loss)
            trues.append(true)
            preds.append(pred)
        trues = np.concatenate(trues, axis=-1)
        preds = np.concatenate(preds, axis=-1)
        acc = cal_accuracy(preds, trues)
        weighted_f1, macro_f1 = cal_F1(preds, trues)
        self.gc_model.is_train = True
        return np.mean(val_loss), acc, weighted_f1, macro_f1

    def test(self, test_loader=None):
        test_loader = self.load_data("test") if test_loader is None else test_loader
        self.gc_model.is_train = False
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        task_dict = load_model(path)
        self.gc_model.load_state_dict(task_dict)
        trues = []
        preds = []
        for data in test_loader:
            data.tokens = get_eigen_tokens(data, self.configs.embed_dim)
            data.x = tlx.ones((data.num_nodes, 128))
            data.n_id = tlx.arange(start=0, limit=int(data.num_nodes), delta=1, dtype=tlx.int64)
            out = self.gc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, data)
            trues.append(true)
            preds.append(pred)
        trues = np.concatenate(trues, axis=-1)
        preds = np.concatenate(preds, axis=-1)
        test_acc = cal_accuracy(preds, trues)
        weighted_f1, macro_f1 = cal_F1(preds, trues)
        self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                         f"weighted_f1={weighted_f1 * 100: .2f},"
                         f"macro_f1={macro_f1 * 100: .2f}%")
        return test_acc, weighted_f1, macro_f1

    def cal_loss(self, output, label, data):
        if isinstance(label, torch.Tensor):
            label_np = label.detach().cpu().numpy()
        elif hasattr(label, 'numpy'):
            label_np = label.numpy()
        elif hasattr(label, 'tolist'):
            label_np = np.array(label.tolist())
        else:
            label_np = np.array(label)
        
        if hasattr(data, 'n_id') and data.n_id is not None:
            if isinstance(data.n_id, torch.Tensor):
                n_id_np = data.n_id.detach().cpu().numpy()
            elif hasattr(data.n_id, 'numpy'):
                n_id_np = data.n_id.numpy()
            else:
                n_id_np = np.array(data.n_id)
            
            if n_id_np.max() < len(label_np):
                y = label_np[n_id_np]
            else:
                y = label_np[:output.shape[0]]
        else:
            y = label_np[:output.shape[0]]
        
        y = np.array(y).reshape(-1)
        valid_size = min(int(output.shape[0]), len(y))
        out = output[:valid_size]
        y = y[:valid_size]
            
        y_tensor = tlx.convert_to_tensor(y, dtype=tlx.int64)
        loss = tlx.reduce_mean(tlx.losses.softmax_cross_entropy_with_logits(out, y_tensor))
        pred = np.array(tlx.convert_to_numpy(tlx.argmax(out, axis=-1)))
        return safe_to_float(loss), pred, y


# ---------------------------------------------------------------------------
# Few-Shot Node Classification (merged from rgt_transfer.py)
# ---------------------------------------------------------------------------

class TransferNodeClassificationLoss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(TransferNodeClassificationLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        out = self.backbone_network(data)
        batch_size = data.batch_size
        out = out[:batch_size]
        
        if label is None:
            y = np.zeros(batch_size, dtype=np.int64)
        else:
            if hasattr(label, 'numpy'):
                label = label.numpy()
            elif hasattr(label, 'detach'):
                label = label.detach().cpu().numpy()
            else:
                label = np.array(label)
            y = np.array(label[:batch_size]).reshape(-1)
            
        y_tensor = tlx.convert_to_tensor(y, dtype=tlx.int64)
        loss = tlx.reduce_mean(self._loss_fn(out, y_tensor))
        return loss


class FewShotNC:
    def __init__(self, configs, load: bool = False):
        self.configs = configs
        self.load = load
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        self.load_word2vec(self.configs.pretrained_word2vec)
        self.supp_sets = self.configs.pretrain_dataset
        self.query_set = self.configs.query_set
        supp_embed_dict, query_embed_dict = self.get_class_embedding(self.supp_sets, self.query_set)
        self.merge_embeddings(supp_embed_dict, query_embed_dict)
        self.load_model()
        self.k_shot = self.configs.k_shot

    def load_model(self):
        pretrained_model = RGT(configs= self.configs,n_layers=self.configs.n_layers, in_dim=self.configs.embed_dim,
                                  hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                                  bias=self.configs.bias,
                                  dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
        if self.load:
            path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path) + ".pt"
            self.logger.info(f"---------------Loading pretrained models from {path}-------------")
            pretrained_dict = load_model(path)
            model_dict = pretrained_model.state_dict()
            model_dict.update(pretrained_dict)
            pretrained_model.load_state_dict(model_dict)

            self.logger.info("----------Freezing weights-----------")
            for module in pretrained_model.modules():
                for param in module.parameters():
                    try:
                        param.requires_grad = False
                    except Exception:
                        pass
        self.nc_model = ShotNCHead(pretrained_model, self.class_embeddings,
                            3 * self.configs.embed_dim + input_dim_dict[self.query_set],
                            self.configs.task_hidden_dim,
                            get_word2vec_dim(self.configs.pretrained_word2vec),
                                   self.configs.drop_edge,
                                   self.configs.drop_feats)

    def load_data(self, data_name, finetune=False):
        if self.k_shot == 0:
            num_per_class = None
        else:
            num_per_class = self.k_shot
        if data_name == self.query_set:
            if finetune:
                dataset = load_data(root=self.configs.root_path, data_name=self.query_set, num_per_class=num_per_class)
                data = self.convert_label(dataset[0].clone(), self.query_set)
                data.tokens = get_eigen_tokens(data, self.configs.embed_dim)
                query_loader = ExtractNodeLoader(data, input_nodes=data.train_mask, batch_size=self.configs.batch_size,
                                                 num_neighbors=self.configs.num_neighbors,
                                                 capacity=self.configs.capacity)
            else:
                dataset = load_data(root=self.configs.root_path, data_name=self.query_set)
                data = self.convert_label(dataset[0].clone(), self.query_set)
                data.tokens = get_eigen_tokens(data, self.configs.embed_dim)
                query_loader = ExtractNodeLoader(data, input_nodes=data.test_mask, batch_size=self.configs.batch_size,
                                                num_neighbors=self.configs.num_neighbors,
                                                capacity=self.configs.capacity)
            return data, query_loader

        dataset = load_data(root=self.configs.root_path, data_name=data_name)
        data = self.convert_label(dataset[0].clone(), data_name)
        data.tokens = get_eigen_tokens(data, self.configs.embed_dim)
        dataloader = ExtractNodeLoader(data, batch_size=self.configs.batch_size,
                                       num_neighbors=self.configs.num_neighbors,
                                       capacity=self.configs.capacity)
        return data, dataloader

    def train(self, load_trained_model=False):
        total_test_acc = []
        total_test_weighted_f1 = []
        total_test_macro_f1 = []
        for t in range(self.configs.exp_iters):
            self.load_model()
            if self.k_shot > 0:
                self.logger.info(f"----------Fine-tuning on {self.query_set}---------")
                self._train_step(load=load_trained_model, data_name=self.query_set,
                                train_epochs=self.configs.shot_epochs, finetune=True,
                                lr=self.configs.lr_few_nc)
            test_acc, weighted_f1, macro_f1 = self.test()
            total_test_acc.append(test_acc)
            total_test_weighted_f1.append(weighted_f1)
            total_test_macro_f1.append(macro_f1)
        mean, std = np.mean(total_test_acc), np.std(total_test_acc)
        self.logger.info(f"Evaluation Acc is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_weighted_f1), np.std(total_test_weighted_f1)
        self.logger.info(f"Evaluation weighted F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_macro_f1), np.std(total_test_macro_f1)
        self.logger.info(f"Evaluation macro F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")

    def test(self):
        data, test_loader = self.load_data(self.configs.query_set)
        self.nc_model.is_train = False
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.trained_model_path_FSL) + ".pt"
        self.logger.info(f"--------------Loading from {path}--------------------")
        self.nc_model.load_state_dict(load_model(path))
        trues = []
        preds = []
        for data in test_loader:
            out = self.nc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, data.batch_size)
            trues.append(true)
            preds.append(pred)
        trues = np.concatenate(trues, axis=-1)
        preds = np.concatenate(preds, axis=-1)
        test_acc = cal_accuracy(preds, trues)
        weighted_f1, macro_f1 = cal_F1(preds, trues)
        self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                         f"weighted_f1={weighted_f1 * 100: .2f},"
                         f"macro_f1={macro_f1 * 100: .2f}%")
        return test_acc, weighted_f1, macro_f1

    def _train_step(self, load, data_name, train_epochs, lr, finetune=False):
        if load:
            load_path = os.path.join(self.configs.checkpoints, self.configs.trained_model_path_FSL) + f".pt"
            self.logger.info(f"---------------Loading trained models from {load_path}-------------")
            pretrained_dict = load_model(load_path)
            model_dict = self.nc_model.state_dict()
            model_dict.update(pretrained_dict)
            self.nc_model.load_state_dict(model_dict)

        path = os.path.join(self.configs.checkpoints, self.configs.trained_model_path_FSL)
        data, dataloader = self.load_data(data_name, finetune=finetune)
        if len(dataloader) == 0:
            self.logger.warning("Empty dataloader detected, skip training.")
            return

        optimizer = tlx.optimizers.Adam(lr, weight_decay=self.configs.weight_decay)
        
        loss_fn = tlx.losses.softmax_cross_entropy_with_logits
        net_with_loss = TransferNodeClassificationLoss(self.nc_model, loss_fn)
        train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, self.nc_model.trainable_weights)

        for epoch in range(train_epochs):
            epoch_loss = []
            trues = []
            preds = []
            for data in tqdm(dataloader):
                self.nc_model.set_train()
                loss = train_one_step(data, data.y)
                
                self.nc_model.set_eval()
                out = self.nc_model(data)
                _, pred, true = self.cal_loss(out, data.y, data.batch_size)
                
                if np.isnan(tlx.convert_to_numpy(loss)):
                    continue

                epoch_loss.append(float(tlx.convert_to_numpy(loss)))
                trues.append(true)
                preds.append(pred)
            trues = np.concatenate(trues, axis=-1)
            preds = np.concatenate(preds, axis=-1)
            train_loss = np.mean(epoch_loss)
            train_acc = cal_accuracy(preds, trues)
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")
            self.logger.info(f"---------------Saving pretrained models to {path}.pt-------------")
            save_model(self.nc_model.state_dict(), path + f".pt")

    def load_word2vec(self, word2vec_path='glove-wiki-gigaword-100'):
        self.word2vec = api.load(word2vec_path)
   
    def get_class_embedding(self, supp_sets: list, query_set: str):
        sp_pattern = r'[ -.%]+'
        supp_embed_dict = {}
        for data_name in supp_sets:
            supp_embed_dict[data_name] = {}
            cls_map = class_maps[data_name]
            for k, word in cls_map.items():
                text = re.split(sp_pattern, word)
                supp_embed_dict[data_name][k] = np.mean([self.word2vec[t.lower()] for t in text], axis=0)

        query_embed_dict = {}
        cls_map = class_maps[query_set]
        for k, word in cls_map.items():
            text = re.split(sp_pattern, word)
            query_embed_dict[k] = np.mean([self.word2vec[t.lower()] for t in text if t != ''], axis=0)

        return supp_embed_dict, query_embed_dict

    def merge_embeddings(self, supp_embed_dict, query_embed_dict):
        merge_embed_dict = {}
        offset_dict = {}
        for i, (data_name, d) in enumerate(supp_embed_dict.items()):
            offset = len(merge_embed_dict)
            offset_dict[data_name] = offset
            for key, value in d.items():
                merge_embed_dict[key + offset] = value
        offset = len(merge_embed_dict)
        offset_dict[self.query_set] = offset
        for key, value in query_embed_dict.items():
            merge_embed_dict[key + offset] = value
        merge_embed = np.stack([em for em in merge_embed_dict.values()], axis=0)
        self.class_embeddings = tlx.convert_to_tensor(merge_embed)
        self.offset_dict = offset_dict

    def convert_label(self, data, data_name):
        data.y = data.y + self.offset_dict[data_name]
        return data

    def cal_loss(self, output, label, batch_size):
        out = output[:batch_size]
        if label is None:
            y = np.zeros(batch_size, dtype=np.int64)
        else:
            if hasattr(label, 'numpy'):
                label = label.numpy()
            elif hasattr(label, 'detach'):
                label = label.detach().cpu().numpy()
            else:
                label = np.array(label)
            y = np.array(label[:batch_size]).reshape(-1)
            
        y_tensor = tlx.convert_to_tensor(y, dtype=tlx.int64)
        
        loss = tlx.reduce_mean(tlx.losses.softmax_cross_entropy_with_logits(out, y_tensor))
        pred = np.array(tlx.convert_to_numpy(tlx.argmax(out, axis=-1)))
        return float(tlx.convert_to_numpy(loss)), pred, y
