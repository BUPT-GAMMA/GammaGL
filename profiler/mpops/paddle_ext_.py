import paddle
from paddle_ext import unsorted_segment_sum

src = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32', stop_gradient=False)
# src = paddle.to_tensor([1, 2, 3, 4, 5, 6], dtype='float32').reshape((2, 3)) 
index = paddle.to_tensor([0, 1, 0], dtype=paddle.int64)
out = unsorted_segment_sum(src, index, 3)
print(out)
gout = paddle.rand(out.shape, dtype='float32')
print(gout)
paddle.autograd.backward([out], [gout], True)
print(src.gradient())
