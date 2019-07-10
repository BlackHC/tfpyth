# TfPyTh

[![Build Status](https://travis-ci.com/BlackHC/tfpyth.svg?branch=master)](https://travis-ci.com/BlackHC/tfpyth) [![codecov](https://codecov.io/gh/BlackHC/tfpyth/branch/master/graph/badge.svg)](https://codecov.io/gh/BlackHC/tfpyth)

> Putting TensorFlow back in PyTorch, back in TensorFlow (with differentiable TensorFlow PyTorch adapters).

Do you have a codebase that uses TensorFlow and one that uses PyTorch and want to train a model that uses both end-to-end?

This library makes it possible without having to rewrite either codebase! 

It allows you to wrap a TensorFlow graph to make it callable (and differentiable) through PyTorch, and vice-versa, using simple functions.

The only caveat is that tensors have to be copied and routed through the CPU until TensorFlow supports `__cuda_array_interface` (please star the [GitHub issue](https://github.com/tensorflow/tensorflow/issues/29039)).

## Install

```
pip install tfpyth
```

### Example

```python
import tensorflow as tf
import torch as th
import numpy as np
import tfpyth

session = tf.Session()

def get_torch_function():
    a = tf.placeholder(tf.float32, name='a')
    b = tf.placeholder(tf.float32, name='b')
    c = 3 * a + 4 * b * b

    f = tfpyth.torch_from_tensorflow(session, [a, b], c).apply
    return f

f = get_torch_function()
a = th.tensor(1, dtype=th.float32, requires_grad=True)
b = th.tensor(3, dtype=th.float32, requires_grad=True)
x = f(a, b)

assert x == 39.

x.backward()

assert np.allclose((a.grad, b.grad), (3., 24.))
```

## What it's got

### `torch_from_tensorflow`

Creates a PyTorch function that is differentiable by evaluating a TensorFlow output tensor given input placeholders.

### `eager_tensorflow_from_torch`

Creates an eager Tensorflow function from a PyTorch function.

### `tensorflow_from_torch`

Creates a TensorFlow op/tensor from a PyTorch function.

## Future work

- [ ] support JAX
- [ ] support higher-order derivatives
