import tensorflow as tf
import torch as th
import numpy as np
import tfpyth

session = tf.Session()


def get_torch_function():
    a = tf.placeholder(tf.float32, name="a")
    b = tf.placeholder(tf.float32, name="b")
    c = 3 * a + 4 * b * b

    f = tfpyth.torch_from_tensorflow(session, [a, b], c).apply
    return f


f = get_torch_function()
a = th.tensor(1, dtype=th.float32, requires_grad=True)
b = th.tensor(3, dtype=th.float32, requires_grad=True)
x = f(a, b)

assert x == 39.0

x.backward()

assert np.allclose((a.grad, b.grad), (3.0, 24.0))
