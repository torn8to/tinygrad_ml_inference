from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad import TinyJit
import os
import timeit



cuda_enable = 0
debug_level = 2
jit_level = 0
if cuda_enable== 0:
    os.environ["CLANG"] = "1"
#os.environ["CUDA"] = str(cuda_enable)
os.environ["NV"] = str(cuda_enable)

@Context(DEBUG=debug_level,JIT=1,)
@TinyJit
def add():
    t1 = Tensor.ones(1000_000_00).reshape(-1,2,1000,1000)*1.25
    t2 = Tensor.arange(1000_000_00).reshape(-1,2,1000,1000)/1000
    t3 = t1 * t2
    t3 = t3**2
    t3.numpy()


class methodContextSwitching():
    def __init__(self,backend, jit_level,debug_level):
        assert backend in ["NV", "CUDA", "CLANG", "GPU"]
        self.backend = backend
        self.jit_level = jit_level
        self.debug_level = debug_level
        os.environ[self.backend] = "1"


    @Context(DEBUG=debug_level,JIT=jit_level)
    @TinyJit
    def opo(self):
        t1 = Tensor.ones(1000_000_00).reshape(-1,2,1000,1000)*1.25
        t2 = Tensor.arange(1000_000_00).reshape(-1,2,1000,1000)/1000
        t3 = t1 + t2
        t3 = t3**2
        t3.numpy()

obj = methodContextSwitching("NV",1,4)
times = timeit.repeat(obj.opo,repeat=5,number=5)
print(times)
