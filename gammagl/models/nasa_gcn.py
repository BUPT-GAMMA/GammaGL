import tensorlayerx as tlx
from tensorlayerx.nn import Module as Model
from gammagl.layers.conv import GCNConv
from gammagl.utils import mask_to_index
# from gammagl.mpops import unsorted_segment_mean

class NASA_GCN(Model):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout_rate, temp, alpha, name=None):
        super().__init__(name=name)
        self.conv1 = GCNConv(in_channels=feature_dim, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=num_classes)
        self.dropout = tlx.layers.Dropout(p=dropout_rate)
        self.temp = temp
        self.alpha = alpha
        self.elu = tlx.layers.ELU()

    def _compute_L_CR(self, aug_graph_edge_index, aug_pred_softmax, num_nodes):
        src, dst = aug_graph_edge_index[0], aug_graph_edge_index[1]

        # 1. Compute average of neighbor predictions (ŷ_i)
        aug_pred_softmax_src = tlx.gather(aug_pred_softmax, src)
        #avg_pred = unsorted_segment_mean(data=aug_pred_softmax_src, segment_ids=dst, num_segments=num_nodes)
        avg_pred = tlx.ops.unsorted_segment_mean(aug_pred_softmax_src, dst, num_segments=num_nodes)

        # Handle nodes with no incoming messages if unsorted_segment_mean results in NaN/Inf
        #avg_pred = tlx.where(tlx.is_finite(avg_pred), avg_pred, tlx.zeros_like(avg_pred))
        is_inf_avg_pred = tlx.is_inf(avg_pred)
        is_nan_avg_pred = tlx.is_nan(avg_pred)
        avg_pred_finite_mask = tlx.logical_and(
            tlx.logical_not(is_inf_avg_pred),
            tlx.logical_not(is_nan_avg_pred)
        )
        avg_pred = tlx.where(avg_pred_finite_mask, avg_pred, tlx.zeros_like(avg_pred))

        # 2. Sharpening (p_i)
        avg_pred_eps = avg_pred + 1e-12
        pow_avg_pred = tlx.pow(avg_pred_eps, 1.0 / self.temp)
        sharp_pseudo_labels = pow_avg_pred / (tlx.reduce_sum(pow_avg_pred, axis=1, keepdims=True) + 1e-12)
        sharp_pseudo_labels_detached = tlx.ops.stop_gradient(sharp_pseudo_labels)

        # 3. Compute KL Divergence Loss
        p_dst_detached = tlx.gather(sharp_pseudo_labels_detached, dst)
        q_src_softmax = tlx.gather(aug_pred_softmax, src)
        
        log_p_dst_detached = tlx.log(p_dst_detached + 1e-12)
        log_q_src_softmax = tlx.log(q_src_softmax + 1e-12)
        
        kl_div_elements = p_dst_detached * (log_p_dst_detached - log_q_src_softmax)
        kl_div_per_edge = tlx.reduce_sum(kl_div_elements, axis=1)

        if tlx.get_tensor_shape(kl_div_per_edge)[0] > 0:
            loss_cr = tlx.reduce_mean(kl_div_per_edge)
        else:
            loss_cr = tlx.convert_to_tensor(0.0, dtype=tlx.float32)
            
        return loss_cr

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        h = self.dropout(x)
        h = self.conv1(h, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        h = self.elu(h)
        h = self.dropout(h)
        logits = self.conv2(h, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        return logits

    def compute_nasa_loss(self, output_logits_aug, augmented_graph, original_graph_labels, original_graph_train_mask):
        train_indices = mask_to_index(original_graph_train_mask)
        gathered_logits_aug = tlx.gather(output_logits_aug, train_indices)
        gathered_labels = tlx.gather(original_graph_labels, train_indices)
        
        loss_ce = tlx.losses.softmax_cross_entropy_with_logits(gathered_logits_aug, gathered_labels)

        aug_pred_softmax = tlx.softmax(output_logits_aug, axis=-1)
        loss_cr = self._compute_L_CR(
            augmented_graph.edge_index,
            aug_pred_softmax,
            tlx.get_tensor_shape(augmented_graph.x)[0]
        )
        
        total_loss = loss_ce + self.alpha * loss_cr
        return total_loss

    def predict(self, x, edge_index, edge_weight=None, num_nodes=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        h = self.elu(h)
        logits = self.conv2(h, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        return logits