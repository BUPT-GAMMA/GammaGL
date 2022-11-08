import torch
import torch_gspmm

x = torch.ones((3, 10), requires_grad=True).float()
weight = torch.ones(3).float()
index = torch.tensor([[1,0,1], [0,1,2]])

out = torch_gspmm.spmm_sum(index, weight, x)
print(out)
out.backward(torch.ones_like(out))
print(x.grad)