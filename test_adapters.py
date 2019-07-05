import tensorflow as tf
import torch as th
import numpy as np

def tensorflow_function(_tf_session, _input_tensors, _output_tensor, dtype=tf.float32):
    # create gradient placeholders\
    _output_tensors = [_output_tensor]
    _gradient_placeholders = [tf.placeholder(dtype=dtype, name=f'gradient{i}') for i, _ in enumerate(_output_tensors)]
    _gradients = tf.gradients(
        ys=_output_tensors, xs=_input_tensors, grad_ys=_gradient_placeholders, unconnected_gradients='zero'
    )

    class TensorFlowFunction(th.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            assert len(args) == len(_input_tensors)

            feed_dict = {input: value.detach().numpy() for input, value in zip(_input_tensors, args)}
            outputs = _tf_session.run(_output_tensors, feed_dict)

            tensors = [th.as_tensor(output) for output in outputs]

            ctx.save_for_backward(*args, *tensors)

            return tensors[0]

        @staticmethod
        def backward(ctx, *grad_outputs):
            assert len(grad_outputs) == len(_gradient_placeholders)
            inputs = ctx.saved_tensors[:len(_input_tensors)]
            outputs = ctx.saved_tensors[-len(_output_tensors):]

            feed_dict = {}
            feed_dict.update({input: value.detach().numpy() for input, value in zip(_input_tensors, inputs)})
            feed_dict.update({gradient_output: value.detach().numpy() for gradient_output, value in
                              zip(_gradient_placeholders, grad_outputs)})

            gradients = _tf_session.run(_gradients, feed_dict)
            # Need to return a tuple or pytorch will misunderstand it and only see one tensor ("of tensors")
            tensors = tuple(th.as_tensor(gradient) for gradient in gradients)
            return tuple(tensors)

    return TensorFlowFunction()

def eager_pytorch_function(func):
    @tf.custom_gradient
    def compute(*inputs):
        input_tensors = [th.tensor(input.numpy(), requires_grad=True) for input in inputs]
        output_tensor = func(*input_tensors)

        def compute_grad(d_output):
            d_output_tensor = th.tensor(d_output.numpy(), requires_grad=False)
            grad_tensors = th.autograd.grad([output_tensor], input_tensors, grad_outputs=[d_output_tensor], allow_unused=True)
            grads = [tf.convert_to_tensor(tensor.numpy()) for tensor in grad_tensors]
            return grads

        return tf.convert_to_tensor(output_tensor.detach().numpy()), compute_grad

    return compute


def pytorch_function(func, inp, Tout, name=None):
    eager_compute = eager_pytorch_function(func)

    return tf.py_function(eager_compute, inp, Tout, name=name)


def test_pytorch_in_tensorflow_eager_mode():
    tf.enable_eager_execution()
    tfe = tf.contrib.eager

    def pytorch_expr(a, b):
        return 3 * a + 4 * b * b

    x = eager_pytorch_function(pytorch_expr)

    assert tf.math.equal(x(tf.convert_to_tensor(1.), tf.convert_to_tensor(3.)), 39.)

    dx = tfe.gradients_function(x)
    assert all(tf.math.equal(dx(tf.convert_to_tensor(1.), tf.convert_to_tensor(3.)), [3., 24.]))
    tf.disable_eager_execution()


def test_pytorch_in_tensorflow_graph_mode():
    session = tf.Session()

    def pytorch_expr(a, b):
        return 3 * a + 4 * b * b

    a = tf.placeholder(tf.float32, name='a')
    b = tf.placeholder(tf.float32, name='b')
    c = pytorch_function(pytorch_expr, [a, b], tf.float32)
    c_grad = tf.gradients([c], [a, b], unconnected_gradients='zero')

    assert np.allclose(session.run([c, c_grad[0], c_grad[1]], {a: 1., b: 3.}), [39., 3., 24.])


def test_tensorflow_in_pytorch():
    session = tf.Session()

    def get_tf_function():
        a = tf.placeholder(tf.float32, name='a')
        b = tf.placeholder(tf.float32, name='b')
        c = 3 * a + 4 * b * b

        f = tensorflow_function(session, [a, b], c).apply
        return f

    f = get_tf_function()
    a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
    b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
    x = f(a_, b_)

    assert x == 39.

    x.backward()

    assert np.allclose((a_.grad, b_.grad), (3., 24.))
