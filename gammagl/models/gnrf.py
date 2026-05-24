import tensorlayerx as tlx
from tensorlayerx import nn
import os
from gammagl.mpops import unsorted_segment_sum, unsorted_segment_mean
TL_BACKEND = os.environ.get('TL_BACKEND', 'torch')
'''
ODEIntAdapter 是一个抽象类，用于适配不同的 ODE 求解器。
子类需要实现 odeint 方法，该方法接收一个函数 func、初始值 y0、时间步 t、方法 method、相对误差 rtol、绝对误差 atol 和选项 options。
返回值是一个包含解的张量，形状为 (t.shape[0], *y0.shape)。
torch:method支持implicit_adams,dopri5,dopri8,rk4,bosh3,adaptive_heun,explicit_adams等
tensorflow:由于 TensorFlow 图模式与变量跟踪机制，部分未参与计算图的网络层可能导致训练或梯度问题,当前适配器未实现 adjoint 模式,method支持dopri5,rk4,euler,midpoint,adaptive_heun等
由于paddlepaddle和mindspore缺少成熟统一的 Neural ODE 求解生态,本模块中为手动实现odeint方法,method 参数当前仅作接口兼容保留
paddlepaddle:rk4求解,固定步长积分,由t_paddle指定,method/rtol/atol/adjoint 参数当前未实际生效
mindspore:euler求解,固定步长积分,由t_ms指定,method/rtol/atol/adjoint 参数当前未实际生效

ODEIntAdapter is an abstract class for adapting different ODE solvers.
Subclasses need to implement the odeint method, which receives a function func, initial value y0, time steps t, method method, relative tolerance rtol, absolute tolerance atol, and options options.
The return value is a tensor containing the solution, with shape (t.shape[0], *y0.shape).
torch: method supports implicit_adams, dopri5, dopri8, rk4, bosh3, adaptive_heun, explicit_adams, etc.
tensorflow: Due to TensorFlow's graph mode and variable tracking mechanism, some network layers not involved in the computation graph may cause training or gradient issues. The current adapter does not implement adjoint mode. method supports dopri5, rk4, euler, midpoint, adaptive_heun, etc.
Since paddlepaddle and mindspore lack a mature and unified Neural ODE solving ecosystem, the odeint method is manually implemented in this module. The method parameter is currently retained only for interface compatibility.
paddlepaddle: RK4 solver with fixed step size integration, specified by t_paddle. The method/rtol/atol/adjoint parameters are not currently effective.
mindspore: Euler solver with fixed step size integration, specified by t_ms. The method/rtol/atol/adjoint parameters are not currently effective.
'''

class ODEIntAdapter:
    @staticmethod
    def odeint(func, y0, t, method='dopri5', rtol=1e-3, atol=1e-6, options=None):
        raise NotImplementedError

class TorchODEIntAdapter(ODEIntAdapter):
    @staticmethod
    def odeint(func, y0, t, method='dopri5', rtol=1e-3, atol=1e-6, options=None, adjoint=False):
        try:
            if adjoint:
                from torchdiffeq import odeint_adjoint as torch_odeint
            else:
                from torchdiffeq import odeint as torch_odeint
        except ImportError:
            raise ImportError("使用 PyTorch 后端需要安装 torchdiffeq: pip install torchdiffeq")
        
        import torch

        y0_torch = y0._tensor if hasattr(y0, '_tensor') else torch.as_tensor(y0)
        t_torch = t._tensor if hasattr(t, '_tensor') else torch.as_tensor(t)
     
        class TorchFuncWrapper(torch.nn.Module):
            def __init__(self, original_func):
                super().__init__()
                self.original_func = original_func
                params = []
                if hasattr(original_func, 'trainable_weights'):
                    for w in original_func.trainable_weights:
                        wt = w._tensor if hasattr(w, '_tensor') else w
                        if not isinstance(wt, torch.nn.Parameter):
                            wt = torch.nn.Parameter(wt)
                        params.append(wt)
                elif hasattr(original_func, 'parameters'):
                    for w in original_func.parameters():
                        wt = w._tensor if hasattr(w, '_tensor') else w
                        if not isinstance(wt, torch.nn.Parameter):
                            wt = torch.nn.Parameter(wt)
                        params.append(wt)
                self.params = torch.nn.ParameterList(params)
                
            def forward(self, t_val, y_val):
                res = self.original_func(t_val, y_val)
                if hasattr(res, '_tensor'):
                    return res._tensor
                return res

        torch_func = TorchFuncWrapper(func)
        
        kwargs = {}
        if adjoint:
            kwargs['adjoint_params'] = tuple(torch_func.parameters())
            
        res_torch = torch_odeint(
            torch_func,
            y0_torch,
            t_torch,
            method=method,
            rtol=rtol,
            atol=atol,
            options=options,
            **kwargs
        )
        return tlx.convert_to_tensor(res_torch)


class TensorFlowODEIntAdapter(ODEIntAdapter):
    @staticmethod
    def odeint(func, y0, t, method='dopri5', rtol=1e-3, atol=1e-6, options=None, adjoint=False):
        try:
            from tfdiffeq import odeint as tf_odeint
        except ImportError:
            raise ImportError("使用 TensorFlow 后端需要安装 tfdiffeq: pip install tfdiffeq")

        import tensorflow as tf

        y0_tf = y0._tensor if hasattr(y0, '_tensor') else tf.convert_to_tensor(y0)
        t_tf = t._tensor if hasattr(t, '_tensor') else tf.convert_to_tensor(t)

        def tf_func(t_val, y_val):
            res = func(t_val, y_val)
            if hasattr(res, '_tensor'):
                return res._tensor
            return res

        ode_options = options or {}
        ode_options.update({'rtol': rtol, 'atol': atol})

        res_tf = tf_odeint(tf_func, y0_tf, t_tf, method=method, **ode_options)
        return tlx.convert_to_tensor(res_tf)


class PaddleODEIntAdapter(ODEIntAdapter):
    @staticmethod
    def odeint(func, y0, t, method='dopri5', rtol=1e-3, atol=1e-6, options=None, adjoint=False):
        try:
            import paddle
        except ImportError:
            raise ImportError("使用 PaddlePaddle 后端需要安装 paddlepaddle")

        y0_paddle = y0._tensor if hasattr(y0, '_tensor') else paddle.to_tensor(y0)
        t_paddle = t._tensor if hasattr(t, '_tensor') else paddle.to_tensor(t)

        def rk4_step(func, t_val, y_val, dt):
            k1 = func(t_val, y_val)
            k2 = func(t_val + dt/2, y_val + dt*k1/2)
            k3 = func(t_val + dt/2, y_val + dt*k2/2)
            k4 = func(t_val + dt, y_val + dt*k3)
            return y_val + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        def paddle_func(t_val, y_val):
            res = func(t_val, y_val)
            if hasattr(res, '_tensor'):
                return res._tensor
            return res

        ys = [y0_paddle]
        for i in range(1, len(t_paddle)):
            dt = t_paddle[i] - t_paddle[i-1]
            y_next = rk4_step(paddle_func, t_paddle[i-1], ys[-1], dt)
            ys.append(y_next)

        res_paddle = paddle.stack(ys, axis=0)
        return tlx.convert_to_tensor(res_paddle)


class MindSporeODEIntAdapter(ODEIntAdapter):
    @staticmethod
    def odeint(func, y0, t, method='dopri5', rtol=1e-3, atol=1e-6, options=None, adjoint=False):
        try:
            import mindspore as ms
            import mindspore.ops as ops
        except ImportError:
            raise ImportError("使用 MindSpore 后端需要安装 mindspore")

        y0_ms = y0._tensor if hasattr(y0, '_tensor') else ms.Tensor(y0)
        t_ms = t._tensor if hasattr(t, '_tensor') else ms.Tensor(t)

        ys = [y0_ms]
        for i in range(1, len(t_ms)):
            dt = t_ms[i] - t_ms[i-1]
            dy = func(t_ms[i-1], ys[-1])
            if hasattr(dy, '_tensor'):
                dy_ms = dy._tensor
            else:
                dy_ms = dy
            y_next = ys[-1] + dt * dy_ms
            ys.append(y_next)

        res_ms = ops.stack(ys, axis=0)
        return tlx.convert_to_tensor(res_ms)


_ADAPTERS = {
    'torch': TorchODEIntAdapter,
    'pytorch': TorchODEIntAdapter,
    'tensorflow': TensorFlowODEIntAdapter,
    'tf': TensorFlowODEIntAdapter,
    'paddle': PaddleODEIntAdapter,
    'mindspore': MindSporeODEIntAdapter,
    'ms': MindSporeODEIntAdapter,
}

def get_adapter(backend=None):
    backend = backend or TL_BACKEND
    adapter = _ADAPTERS.get(backend.lower())
    if adapter is None:
        raise ValueError(f"不支持的后端: {backend}. 请从 {list(_ADAPTERS.keys())} 中选择。")
    return adapter


def odeint(func, y0, t, method='dopri5', rtol=1e-3, atol=1e-6, options=None, backend=None, adjoint=False):
    adapter = get_adapter(backend)
    return adapter.odeint(func, y0, t, method=method, rtol=rtol, atol=atol, options=options, adjoint=adjoint)


class SimpleMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0):
        super().__init__()
        self.lins = nn.ModuleList()
        self.dropout_layer = tlx.nn.Dropout(p=dropout)
        in_dims = [in_channels] + [hidden_channels] * (num_layers - 1)
        out_dims = [hidden_channels] * (num_layers - 1) + [out_channels]
        for i, o in zip(in_dims, out_dims):
            self.lins.append(tlx.nn.Linear(in_features=i, out_features=o))
            
    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = self.dropout_layer(x)
            x = lin(x)
            if i < len(self.lins) - 1:
                x = tlx.nn.ReLU()(x)
        return x

class GNRF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.edge_index = None         
        self.damping = args.damping
        self.edgenet = args.edgenet
        if self.edgenet:
            self.mlp_1 = SimpleMLP(
                in_channels=2 * args.num_hid,
                hidden_channels=args.num_hid,
                out_channels=args.num_hid,
                num_layers=2,
                dropout=args.dropout
            )
            if args.channel_curv:
                self.mlp_2 = SimpleMLP(
                    in_channels=2 * args.num_hid,
                    hidden_channels=args.num_hid,
                    out_channels=args.num_hid,
                    num_layers=2,
                    dropout=args.dropout
                )
            else:
                self.mlp_2 = SimpleMLP(
                    in_channels=2 * args.num_hid,
                    hidden_channels=args.num_hid,
                    out_channels=1,
                    num_layers=2,
                    dropout=args.dropout
                )
        else:
            self.a = tlx.Variable(initial_value=tlx.convert_to_tensor(0.5, dtype=tlx.float32),trainable=True)

    def set_edges(self, edge_index):
        self.edge_index = edge_index

    def curvature(self, H_i, H_j):
        curv = tlx.concat([H_i, H_j], axis=1)           
        curv = tlx.nn.ReLU()(self.mlp_1(curv))          
        num_nodes = int(tlx.reduce_max(self.edge_index).item() + 1) if hasattr(tlx.reduce_max(self.edge_index), 'item') else int(tlx.reduce_max(self.edge_index)) + 1
        curv = unsorted_segment_sum(curv, self.edge_index[0], num_segments=num_nodes)
        curv = tlx.concat([tlx.gather(curv, self.edge_index[0]),tlx.gather(curv, self.edge_index[1])], axis=1)   
        curv = self.mlp_2(curv)                         
        return curv

    def forward(self, t, H):
        eps = 1e-8
        if self.damping:
            norm = tlx.sqrt(tlx.reduce_sum(tlx.square(H), axis=1, keepdims=True) + eps)
            H = H / norm
        H_i = tlx.gather(H, self.edge_index[0])
        H_j = tlx.gather(H, self.edge_index[1])
        if self.edgenet:
            curv = self.curvature(H_i, H_j)
        else:
            curv = tlx.clip_by_value(self.a, eps, 1.0)
            curv = tlx.ones((H_i.shape[0], 1)) * curv
        if self.damping:
            cos = tlx.reduce_sum(H_i * H_j, axis=1, keepdims=True)   
            H_edge = curv * (H_j - cos * H_i)                         
        else:
            H_edge = curv * (H_j - H_i)                           
        dH = unsorted_segment_mean(H_edge, self.edge_index[0], num_segments=H.shape[0])
        if self.damping:
             if hasattr(dH, '_tensor'): dH_t2 = dH._tensor
             else: dH_t2 = dH
             norm_dH = tlx.sqrt(tlx.reduce_sum(tlx.square(dH_t2), axis=1, keepdims=True) + eps)
             dH = dH_t2 / norm_dH
        return dH
    


class GNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if getattr(args, 'use_bn_in', False):
            self.bn_in = tlx.nn.BatchNorm1d(num_features=args.num_hid, momentum=0.9) 
        if getattr(args, 'use_mlp_in', False):
            self.mlp_in = SimpleMLP(
                in_channels=args.num_hid,
                hidden_channels=args.num_hid,
                out_channels=args.num_hid,
                num_layers=2,
                dropout=args.dropout
            )
        self.lin_in = tlx.nn.Linear(in_features=args.num_feat, out_features=args.num_hid)
        
        if getattr(args, 'use_bn_out', False):
            self.bn_out = tlx.nn.BatchNorm1d(num_features=args.num_hid, momentum=0.9)
        if getattr(args, 'use_mlp_out', False):
            self.mlp_out = SimpleMLP(
                in_channels=args.num_hid,
                hidden_channels=args.num_hid,
                out_channels=args.num_hid,
                num_layers=2,
                dropout=args.dropout
            )
        self.lin_out = tlx.nn.Linear(in_features=args.num_hid, out_features=args.num_class)
       
        self.ODE_block = GNRF(args)
        self.t = tlx.convert_to_tensor([args.t_start, args.t_end], dtype=tlx.float32)
        self.t = tlx.to_device(self.t, args.device)
        self.solver = args.solver
        self.adjoint = args.adjoint
        self.tol_scale = args.tol_scale
        
        self.dropout = tlx.nn.Dropout(p=args.dropout)

    def pre_transform(self, x, edge_index):
        x = self.dropout(x)
        x = self.lin_in(x)
        x = tlx.nn.ReLU()(x)
        if getattr(self.args, 'use_mlp_in', False):
            x = self.mlp_in(x)
            x = tlx.nn.ReLU()(x)
        if getattr(self.args, 'use_bn_in', False):
            x = self.bn_in(x)
        return x

    def solve_ODE(self, x_0, edge_index):
        self.ODE_block.set_edges(edge_index)
        rtol = self.tol_scale * 1e-9
        atol = self.tol_scale * 1e-7
        
        trajectory = odeint(
            func=self.ODE_block,
            y0=x_0,
            t=self.t,
            method=self.solver,          
            rtol=rtol,
            atol=atol,
            adjoint=self.adjoint         
        )
        end_state = trajectory[-1]
        return end_state

    def post_transform(self, x, edge_index):
        x = tlx.nn.ReLU()(x)
        if getattr(self.args, 'use_bn_out', False):
            x = self.bn_out(x)
        if getattr(self.args, 'use_mlp_out', False):
            x = self.mlp_out(x)
            x = tlx.nn.ReLU()(x)
        x = self.dropout(x)
        x = self.lin_out(x)
        return x

    def forward(self, x, edge_index):
        x = self.pre_transform(x, edge_index)
        x = self.solve_ODE(x, edge_index)   
        x = self.post_transform(x, edge_index)
        return x
