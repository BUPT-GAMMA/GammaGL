import cupy as cp

def test1():
    print("using paddle")
    from paddle.utils.dlpack import from_dlpack
    for i in range(100):
        a = cp.random.random((1024*128, 1024), dtype=cp.float32)
        b = from_dlpack(a.toDlpack())
        print(i)
def test0():
    for i in range(100):
        a = cp.random.random((1024*128, 1024), dtype=cp.float32)
def test2():
    from torch import from_dlpack
    print("using torch")
    for i in range(100):
        a = cp.random.random((1024*128, 1024), dtype=cp.float32)
        b = from_dlpack(a.toDlpack())
test1()