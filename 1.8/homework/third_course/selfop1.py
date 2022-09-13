import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import ms_kernel
from mindspore.nn import Cell
import math

ms.set_context(mode=ms.GRAPH_MODE,device_target="CPU")

def custom_sin(x):
    return np.sin(x)

def infer_shape(x):
    return x

def infer_dtype(x):
    return x

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.cus_sin = ops.Custom(func = custom_sin,
                out_shape = infer_shape,
                out_dtype = infer_dtype,
                func_type="pyfunc")


    def construct(self,x):
        x = self.cus_sin(x)

        return x

x = np.array([np.pi/6, np.pi/3, np.pi/2]).astype(np.float32)
out = Net()(ms.Tensor(x))
print (out)
