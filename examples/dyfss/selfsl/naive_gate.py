r"""
Naive gate
"""
from .base_gate import BaseGate

import tensorlayerx as tlx  

class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_experts, top_k=3):
        super().__init__(num_experts)
    
        self.num_experts = num_experts
        self.gate = tlx.nn.Linear(in_features=d_model, out_features=self.num_experts)
        # 将PyTorch的Parameter替换为TensorLayerX的Variable
        self.w_noise = tlx.Variable(tlx.zeros((d_model, self.num_experts)), name='w_noise')
        self.top_k = top_k
    
        # 其他初始化代码保持不变
        self.softplus = tlx.nn.Softplus()
        self.softmax = tlx.nn.Softmax(axis=1)
        self.mean = tlx.convert_to_tensor([0.0])
        self.std = tlx.convert_to_tensor([1.0])
        assert (self.top_k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            # 修改：获取设备名称字符串
            device_str = str(x.device)
            return tlx.to_device(tlx.convert_to_tensor([0], dtype=x.dtype), device=device_str)
        
        return tlx.reduce_variance(x.float()) / (tlx.reduce_mean(x.float())**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        # 修改：使用tlx.reduce_sum替代tlx.ops.sum
        return tlx.reduce_sum((gates > 0), axis=0)

    def clean_gating(self, inp):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        # 使用axis替代dim
        gate_top_k_val, gate_top_k_idx = tlx.ops.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )
        gate_top_k_val = tlx.reshape(gate_top_k_val, shape=(-1, self.top_k))
    
        gate_score = tlx.ops.softmax(gate_top_k_val, axis=-1)
    
        # 修改：获取设备名称字符串
        device_str = str(gate.device)
        new_gates_score = tlx.to_device(tlx.zeros_like(gate), device=device_str)
        
        # 修改：使用一维索引和一维更新值
        batch_size = gate.shape[0]
        num_experts = gate.shape[1]
        
        # 创建一维索引
        batch_indices = tlx.tile(tlx.expand_dims(tlx.arange(0, batch_size), 1), [1, self.top_k])
        batch_indices = tlx.reshape(batch_indices, [-1])
        expert_indices = tlx.reshape(gate_top_k_idx, [-1])
        
        # 计算一维索引
        flat_indices = batch_indices * num_experts + expert_indices
        
        # 更新值
        updates = tlx.reshape(gate_score, [-1])
        
        # 使用一维索引和一维更新值
        flat_tensor = tlx.reshape(new_gates_score, [-1])
        flat_tensor = tlx.tensor_scatter_nd_update(flat_tensor, 
                                                  tlx.expand_dims(flat_indices, 1), 
                                                  updates)
        gates = tlx.reshape(flat_tensor, new_gates_score.shape)

        # 修改这一行
        load = self._gates_to_load(gates)
        return gates, load

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # 使用shape属性替代size()方法
        batch = clean_values.shape[0]
        m = noisy_top_values.shape[1]
        
        top_values_flat = tlx.reshape(noisy_top_values, shape=(-1,))

        threshold_positions_if_in = tlx.arange(start=0, limit=batch) * m + self.top_k
        threshold_if_in = tlx.expand_dims(tlx.gather(top_values_flat, threshold_positions_if_in), axis=1)

        is_in = tlx.greater(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = tlx.expand_dims(tlx.gather(top_values_flat, threshold_positions_if_out), axis=1)

        # 使用自定义实现替代Normal分布
        # 计算标准正态分布的累积分布函数(CDF)
        def normal_cdf(x):
            # 使用近似计算标准正态分布的CDF
            # 参考: https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
            z = tlx.abs(x) * 0.7071067811865475  # 1/sqrt(2)
            # 使用sigmoid函数近似CDF
            y = 0.5 + 0.5 * tlx.tanh(0.5 * 1.5976 * x)  # 近似CDF
            return y
        
        # 标准化输入
        prob_if_in = normal_cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal_cdf((clean_values - threshold_if_out) / noise_stddev)
        
        prob = tlx.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, inp, is_noisy, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.gate(inp)
        if (is_noisy == 1) and train:
            raw_noise_stddev = tlx.matmul(inp, self.w_noise)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            # 使用 tlx 的随机函数替代 torch.randn_like
            noisy_logits = clean_logits + (tlx.random.normal(shape=clean_logits.shape) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 替换logits.topk为tlx.ops.topk
        top_logits, top_indices = tlx.ops.topk(
            logits, k=min(self.top_k + 1, self.num_experts), dim=1, largest=True, sorted=False
        )
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = tlx.zeros_like(logits)
        
        # 修改：使用一维索引和一维更新值
        batch_size = logits.shape[0]
        num_experts = logits.shape[1]
        
        # 创建一维索引
        batch_indices = tlx.tile(tlx.expand_dims(tlx.arange(0, batch_size), 1), [1, self.top_k])
        batch_indices = tlx.reshape(batch_indices, [-1])
        expert_indices = tlx.reshape(top_k_indices, [-1])
        
        # 计算一维索引
        flat_indices = batch_indices * num_experts + expert_indices
        
        # 更新值
        updates = tlx.reshape(top_k_gates, [-1])
        
        # 使用一维索引和一维更新值
        flat_tensor = tlx.reshape(zeros, [-1])
        flat_tensor = tlx.tensor_scatter_nd_update(flat_tensor, 
                                                  tlx.expand_dims(flat_indices, 1), 
                                                  updates)
        gates = tlx.reshape(flat_tensor, zeros.shape)
        
        if (is_noisy == 1) and self.top_k < self.num_experts and train:
            # 使用tlx.reduce_sum替代.sum
            load = tlx.reduce_sum(self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits), axis=0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, is_noisy=0, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        if is_noisy == 1:
            gates, load = self.noisy_top_k_gating(x, is_noisy, self.training)
        else:
            gates, load = self.clean_gating(x)
        # 修改：使用tlx.reduce_sum替代gates.sum
        importance = tlx.reduce_sum(gates, axis=0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        return gates, loss

