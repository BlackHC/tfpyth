import tensorflow as tf
import torch as th
import numpy as np
import tfpyth


def test_pytorch_in_tensorflow_eager_mode():
    tf.enable_eager_execution()
    tfe = tf.contrib.eager

    def pytorch_expr(a, b):
        return 3 * a + 4 * b * b

    x = tfpyth.eager_tensorflow_from_torch(pytorch_expr)

    assert tf.math.equal(x(tf.convert_to_tensor(1.0), tf.convert_to_tensor(3.0)), 39.0)

    dx = tfe.gradients_function(x)
    assert all(tf.math.equal(dx(tf.convert_to_tensor(1.0), tf.convert_to_tensor(3.0)), [3.0, 24.0]))
    tf.disable_eager_execution()


def test_pytorch_in_tensorflow_graph_mode():
    session = tf.Session()

    def pytorch_expr(a, b):
        return 3 * a + 4 * b * b

    a = tf.placeholder(tf.float32, name="a")
    b = tf.placeholder(tf.float32, name="b")
    c = tfpyth.tensorflow_from_torch(pytorch_expr, [a, b], tf.float32)
    c_grad = tf.gradients([c], [a, b], unconnected_gradients="zero")

    assert np.allclose(session.run([c, c_grad[0], c_grad[1]], {a: 1.0, b: 3.0}), [39.0, 3.0, 24.0])


def test_tensorflow_in_pytorch():
    session = tf.Session()

    def get_tf_function():
        a = tf.placeholder(tf.float32, name="a")
        b = tf.placeholder(tf.float32, name="b")
        c = 3 * a + 4 * b * b

        f = tfpyth.torch_from_tensorflow(session, [a, b], c).apply
        return f

    f = get_tf_function()
    a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
    b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
    x = f(a_, b_)

    assert x == 39.0

    x.backward()

    assert np.allclose((a_.grad, b_.grad), (3.0, 24.0))
