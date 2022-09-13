import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import ms_kernel
from mindspore.nn import Cell

@ms_kernel
def tensor_add_3d(x,y):
    result = output_tensor(x.shape, x.dtype)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                result[i,j,k] = x[i,j,k] + y[i,j,k]

    return result

tensor_add_3d_op = ops.Custom(func = tensor_add_3d)

input_tensor_x = ms.Tensor(np.random.normal(0,1,[2,3,4]).astype(np.float32))
print("x\n",input_tensor_x)
input_tensor_y = ms.Tensor(np.random.normal(0,1,[2,3,4]).astype(np.float32))
print("y\n",input_tensor_y)
result_cus = tensor_add_3d_op(input_tensor_x,input_tensor_y)
print("hybrid, tensor_add_3d_op\n",result_cus)

# pyfunc
ms.set_context(mode=ms.GRAPH_MODE,device_target="CPU")

def infer_shape_py(x,y):
    return x

def infer_dtype_py(x,y):
    return x

tensor_add_3d_py_func = ops.Custom(func = tensor_add_3d,
                out_shape = infer_shape_py,
                out_dtype = infer_dtype_py,
                func_type="pyfunc")

result_pyfunc = tensor_add_3d_py_func(input_tensor_x,input_tensor_y)
print ("pyfunc\n",result_pyfunc)
