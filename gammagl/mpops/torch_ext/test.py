import torch
import torch_operator

msg = torch.tensor([[2,3,4],[4,2,8],[4,9,7]],dtype=torch.float32)
dst = torch.tensor([0,2,2])
temp = torch_operator.segment_max(msg,dst,3)
print(temp)

msg = torch.tensor([[2,3,4],[4,2,8],[4,9,7]],dtype=torch.float32)
dst = torch.tensor([0,2,2])
temp = torch_operator.segment_sum(msg,dst,3)
print(temp)

msg = torch.tensor([[2,3,4],[4,2,8],[4,9,7]],dtype=torch.float32)
dst = torch.tensor([0,2,2])
temp = torch_operator.segment_mean(msg,dst,3)
print(temp)