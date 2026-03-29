import tensorlayerx.nn as nn
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import igraph as ig
import scipy
import scipy.sparse as sp
import torch
import numpy as np
import scipy.linalg as slin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.cpu().detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input

trace_expm = TraceExpm.apply

class LBFGSBScipy(torch.optim.Optimizer):
    """Wrap L-BFGS-B algorithm, using scipy routines.

    Courtesy: Arthur Mensch's gist
    https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
    """

    def __init__(self, params):
        defaults = dict()
        super(LBFGSBScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGSBScipy doesn't support per-parameter options"
                             " (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel = sum([p.numel() for p in self._params])

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)



    def _gather_flat_bounds(self):
        bounds = []
        for p in self._params:
            if hasattr(p, 'bounds'):
                b = p.bounds
            else:
                b = [(None, None)] * p.numel()
            bounds += b
        return bounds

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)


    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            flat_params = flat_params.to(torch.get_default_dtype())
            self._distribute_flat_params(flat_params)
            loss = closure()
            loss = loss.item()
            flat_grad = self._gather_flat_grad().cpu().detach().numpy()
            return loss, flat_grad.astype('float64')

        initial_params = self._gather_flat_params()
        initial_params = initial_params.cpu().detach().numpy()

        bounds = self._gather_flat_bounds()

        # Magic
        sol = sopt.minimize(wrapped_closure,
                            initial_params,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds
                            )  # options={'maxiter': 15}

        final_params = torch.from_numpy(sol.x)
        final_params = final_params.to(torch.get_default_dtype())
        self._distribute_flat_params(final_params)

class model_p1_MLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(model_p1_MLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        nums = dims[1]
        self.dims = dims
        self.node_nums = nums
        self.w_est = torch.ones((d, d))
        self.p1_est = torch.ones((d, d))
        self.p2_est = torch.ones((d, d))
        self.w_est.requires_grad = True
        self.p1_est.requires_grad = True
        self.p2_est.requires_grad = True



    def forward(self, Xlags, adj1):  # [n, d] -> [n, d]
        # M = XW + A@Xlag1@P1 + A@Xlag2@P2
        M = torch.matmul(Xlags[1:], self.w_est) + torch.matmul(torch.matmul(adj1, Xlags[:-1]), self.p1_est)
        return M

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        h = trace_expm(self.w_est * self.w_est) - d  # (Zheng et al. 2018)
        return h

    def diag_zero(self):
        diag_loss = torch.trace(self.w_est * self.w_est)
        return diag_loss

class model_p2_MLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(model_p2_MLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        nums = dims[1]
        self.dims = dims
        self.node_nums = nums
        self.w_est = torch.ones((d, d))
        self.p1_est = torch.ones((d, d))
        self.p2_est = torch.ones((d, d))
        self.w_est.requires_grad = True
        self.p1_est.requires_grad = True
        self.p2_est.requires_grad = True

    def forward(self, X, adj1, adj2):  # [n, d] -> [n, d]
        # M = XW + A@Xlag1@P1 + A@Xlag2@P2
        M = torch.matmul(X[2:],self.w_est) + torch.matmul(torch.matmul(adj1, X[1:-1]),self.p1_est) + torch.matmul(torch.matmul(adj2, X[:-2]),self.p2_est)
        return M

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        h = trace_expm(self.w_est * self.w_est) - d  # (Zheng et al. 2018)
        return h

    def diag_zero(self):
        diag_loss = torch.trace(self.w_est * self.w_est)
        #diag_loss=tlx.ops.convert_to_tensor(np.trace(tlx.ops.convert_to_numpy((self.w_est * self.w_est).detach())),dtype="float32")
        return diag_loss

def squared_loss(output, target):
    n = target.shape[0] * target.shape[1]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

def L1Norm(matrix):
    return  torch.abs(matrix).sum()

def p1_dual_ascent_step(model, X_lags, adj1, lambda1, lambda2, lambda3,rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy([model.w_est,model.p1_est,model.p2_est])
    primal = torch.Tensor([0.0]).float()
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_lags, adj1)
            loss = squared_loss(X_hat, X_lags[1:])
            h_val = model.h_func()
            diag_loss = model.diag_zero()
            penalty1 = 0.5 * rho * h_val * h_val + alpha * h_val
            primal_obj = primal + loss + 100 * penalty1 + 1000 * diag_loss +  lambda1 * L1Norm(model.w_est) + lambda2 * L1Norm(model.p1_est)
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func()
        if h_new.item() > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new

def p2_dual_ascent_step(model, X_torch, adj1, adj2, lambda1, lambda2, lambda3, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy([model.w_est,model.p1_est,model.p2_est])
    primal = torch.Tensor([0.0]).float().to(device)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch, adj1, adj2)
            loss = squared_loss(X_hat, X_torch[2:])
            h_val = model.h_func()
            diag_loss = model.diag_zero()
            penalty1 = 0.5 * rho * h_val * h_val + alpha * h_val


            primal_obj = primal + loss + 100 * penalty1 + 1000 * diag_loss + lambda1 * L1Norm(model.w_est) + lambda2 * L1Norm(model.p1_est) + lambda3 * L1Norm(model.p2_est)
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func()
        if h_new.item() > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new

def p1_linear_model(model: nn.Module,
                      Xlags,
                      adj1,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = p1_dual_ascent_step(model, Xlags, adj1, lambda1, lambda2, lambda3,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.w_est
    W_est = W_est.detach().numpy()

    P1_est = model.p1_est
    P1_est = P1_est.detach().numpy()


    return W_est, P1_est

def p2_linear_model(model: nn.Module,
                      X,
                      adj1,
                      adj2,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      max_iter: int = 1000,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = p2_dual_ascent_step(model, X, adj1, adj2, lambda1, lambda2, lambda3,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.w_est
    W_est = W_est.detach().numpy()

    P1_est = model.p1_est
    P1_est = P1_est.detach().numpy()

    P2_est = model.p2_est
    P2_est = P2_est.detach().numpy()

    return W_est, P1_est, P2_est

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def _graph_to_adjmat(G):
    return np.array(G.get_adjacency().data)

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)

    elif graph_type == 'BA':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_parameter(B, w_ranges=((-1.0, -0.5), (0.5, 1.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X

def simulate_linear_sem_with_P1(W, P, lagX, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale, time):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + time * z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + time * z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + time * z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + time * z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())

    PG = ig.Graph.Weighted_Adjacency(P.tolist())

    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)

        P_parents = PG.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], 0) + _simulate_single_equation(
            lagX[:, P_parents], P[P_parents, j], scale_vec[j], 1)
    return X

def simulate_linear_sem_with_P2(W, P, P2, lagX, lagX2, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())

    PG = ig.Graph.Weighted_Adjacency(P.tolist())

    P2G = ig.Graph.Weighted_Adjacency(P2.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)

        P_parents = PG.neighbors(j, mode=ig.IN)

        P2_parents = P2G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j]) + _simulate_single_equation(
            lagX[:, P_parents], P[P_parents, j], scale_vec[j]) + _simulate_single_equation(lagX2[:, P2_parents],
                                                                                           P2[P2_parents, j],
                                                                                           scale_vec[j])
    return X

def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """

    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """

    d = B_true.shape[0]
    # linear index of nonzeros
    # pred_und = np.flatnonzero(B_est == -1) # 记录B_est中等于-1的坐标
    pred = np.flatnonzero(B_est == 1)  # 记录B_est中等于1的坐标
    cond = np.flatnonzero(B_true)  # 记录B_true中不等于0的坐标，那cond应该等于pred_und + pred
    cond_reversed = np.flatnonzero(B_true.T)  # 记录B_true.T中不等于0的坐标
    cond_skeleton = np.concatenate([cond, cond_reversed])  # 把B_true和B_true.T中不等于0的坐标串联起来成为skeleton，意思是真实dag的无向图

    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # B_true中不等于0的坐标（等于1或-1的）和 B_est中等于1的坐标，看命中了多少，返回两个数组中的共同元素
    # treat undirected edge favorably
    # 看 B_est中等于-1的元素 和 无向图中不为0元素的 的命中情况，返回两个数组中的共同元素
    # 真正的命中情况是将上述两个命中串联起来，意思是如果是-1，算只要无向图中相应位置能对上，那也算是命中

    # false pos
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    # np.setdiff1d 返回的是在 pred 中但不在 cond 中的元素值（也就是边的下标）
    # 这部分属于预测有向边预测错的

    # false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    # 在pred_und中，但不在cond_skeleton中的
    # 属于预测无向边也预测错的

    # false_pos = np.concatenate([false_pos, false_pos_und])
    # 总的预测错的是二者串联的结果

    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    # 这部分是 预测为1 但实际不为1的这部分
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # 这部分是 预测为1但实际不为1 和 转置之后为1 的交集，方向预测反的有哪些

    # compute ratio
    # pred_size等于所有预测结果的个数，预测为1的个数+预测为-1的个数
    pred_size = len(pred)

    cond_neg_size = d * d - len(cond)  # 实际为负

    # 0.5 * d * (d - 1)相当于一个上三角矩阵的元素个数 减去 B_true中不等于0的坐标，相当于所有应该为0的元素个数

    fdr = float(len(false_pos)) / max(pred_size, 1)
    # fdr是总的预测错的/总的预测数
    # 总的预测错的 = （（返回的是在 pred 中但不在 cond_skeleton 中的元素值）+（在pred_und中，但不在cond_skeleton中的）
    # 总的预测数（pred预测为1的 + pred_und预测为-1的）

    tpr = float(len(true_pos)) / max(len(cond), 1)

    fpr = float(len(false_pos)) / max(cond_neg_size, 1)

    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size

def generate_adj(n):  # n是样本数量
    B = scipy.sparse.rand(n, n, density=0.1, format='coo', dtype=int)
    i = torch.LongTensor([B.row.tolist(), B.col.tolist()])
    v = torch.FloatTensor(B.data.tolist())
    A = torch.sparse.FloatTensor(i, v, torch.Size([n, n])).to_dense()
    A = A.numpy()
    A[np.abs(A) > 0] = 1
    A = np.triu(A, 1)
    adj = (A + A.T)
    adj = adj + np.eye(A.shape[0])

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    tmp = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    tmp = tmp.todense()
    return tmp

def generate_tri(num_nodes, graph_type, low_value, high_value):  # num_nodes是变量数
    if graph_type == 'ER':
        w_decay = 1.0
        i = 1
        neg = 0.5
        if num_nodes < 10:
            degree = 2
        else:
            degree = float(num_nodes / 10)
        u_i = np.random.uniform(low=low_value, high=high_value, size=[num_nodes, num_nodes]) / (w_decay ** i)
        u_i[np.random.rand(num_nodes, num_nodes) < neg] *= -1

        # ER图的设置
        prob = degree / num_nodes
        b = (np.random.rand(num_nodes, num_nodes) < prob).astype(float)
        print("b***********************", b)
        # 生成a矩阵
        a = (b != 0).astype(float) * u_i
        print("a**********************", a)
        p_mat = a
        p_true = matrix_to_zerone(p_mat)

    elif graph_type == 'SBM':  ########################################################################
        p_in = 0.3
        p_out = 0.3 * p_in
        part1 = int(0.5 * num_nodes)
        part2 = num_nodes - part1
        G = ig.Graph.SBM(n=num_nodes, pref_matrix=[[p_in, p_out], [p_out, p_in]], block_sizes=[part1, part2])
        a = _graph_to_adjmat(G)
        p_true = a
        p_mat = simulate_parameter(p_true)

    print("p1_mat", p_mat)
    return p_mat, p_true

def matrix_to_zerone(m):
    m_ = m.copy()
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] != 0:
                m_[i][j] = 1
    return m_

def convert_Tensor_p1(adj1,X):
    adj1_torch=torch.Tensor(adj1)
    X_torch = torch.Tensor(X)
    return adj1_torch,X_torch

def convert_Tensor_p2(adj1,adj2,X):
    adj1_torch=torch.Tensor(adj1)
    adj2_torch = torch.Tensor(adj2)
    X_torch = torch.Tensor(X)
    return adj1_torch,adj2_torch,X_torch

def convert_by_threshold(adj_Tensor,thre):
    adj_Tensor[np.abs(adj_Tensor)<thre]=0
    return adj_Tensor

def data_pre_p1(n, d, s0, w_graph_type, p_graph_type, sem_type):
    # n节点个数，d采样个数，s0期望生成的边个数，w权重矩阵，p时滞权重矩阵
    num_step = 5
    Xbase = []
    w_true = simulate_dag(d, s0, w_graph_type)
    w_mat = simulate_parameter(w_true)

    adj1 = generate_adj(n)

    Xbase1 = simulate_linear_sem(w_mat, n, sem_type, noise_scale=0.5)
        # 加噪图结构矩阵
    p1_mat, p1_true = generate_tri(d, p_graph_type, low_value=0.0, high_value=2)
        # 生成p矩阵

    for i in range(num_step):
        Xbase1 = simulate_linear_sem_with_P1(w_mat, p1_mat, adj1 @ Xbase1, n, sem_type, noise_scale=1)
        Xbase.append(Xbase1)
        # 模拟生成基于p矩阵的加噪图结构矩阵

    return Xbase, adj1, w_true, w_mat, p1_true, p1_mat

def data_pre_p2(n, d, s0, w_graph_type, p_graph_type, sem_type):
    # n节点个数，d采样个数，s0期望生成的边个数，w权重矩阵，p时滞权重矩阵
    num_step = 5
    Xbase = []
    w_true = simulate_dag(d, s0, w_graph_type)
    w_mat = simulate_parameter(w_true)

    adj1 = generate_adj(n)
    adj2 = generate_adj(n)

    Xbase1 = simulate_linear_sem(w_mat, n, sem_type, noise_scale=0.3)
    p1_mat, p1_true = generate_tri(d, p_graph_type, low_value=0.0, high_value=2)

    p2_mat, p2_true = generate_tri(d, p_graph_type, low_value=0.5, high_value=1)
    Xbase.append(Xbase1)

    Xbase.append(Xbase1)

    for i in range(num_step + 1):
        Xbase1 = simulate_linear_sem_with_P2(w_mat, p1_mat, p2_mat, adj1 @ Xbase[-1], adj2 @ Xbase[-2], n,
                                                  sem_type,
                                                  noise_scale=1)
        Xbase.append(Xbase1)

    return Xbase, adj1, adj2, w_true, w_mat, p1_true, p1_mat, p2_true, p2_mat





