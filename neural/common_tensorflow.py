import numpy as np

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _broadcast_pad(pad, a, reps):
    """
    Broadcasts padding tensor before attaching it to a

    pad: a tensor to attach, must have 0, 1 or 2 less dimensions than a,
    a: a tensor to be padded
    reps: the number of repetitions of pad along second (timestep) axis

    pad: a broadcasted tensor
    """
    if a.shape.ndims - pad.shape.ndims == 2:
        pad, batch_tile = pad[None, None,:], tf.shape(a)[0]
    elif a.shape.ndims - pad.shape.ndims == 1:
        pad, batch_tile = pad[:,None], 1
    elif pad.shape.ndims == a.shape.ndims:
        batch_tile = 1
    else:
        raise ValueError("pad.ndim must be in [a.ndim, a.ndim-1, a.ndim-2]")
    pad = tf.tile(pad, [batch_tile, reps] + [1]*(a.shape.ndims - 2))
    return pad


def batch_shifted_fill(a, h, pad, r=0, right_pad=None, flatten=False):
    """

    a: array-like or tf.tensor
    h: int, history length
    pad: array-like or tf.tensor, padding value for a elements
    flatten: boolean, default=False
        whether to flatten histories. In case or a.dim >= 3 individual elements of a are arrays,
        therefore histories are sequences of arrays. By default they are not flattened.

    :returns
    answer_: array-like or tf.tensor
    """
    a = tf.convert_to_tensor(a)
    pad = tf.convert_to_tensor(pad)
    if r > 0:
        if right_pad is None:
            raise ValueError("right_pad cannot be None in case right padding is active (right_pad > 0)")
        right_pad = tf.convert_to_tensor(right_pad)
    pad = _broadcast_pad(pad, a, h-1)
    a_padded = tf.concat([pad, a], axis=1)
    if r > 0:
        right_pad = _broadcast_pad(right_pad, a, r-1)
        a_padded = tf.concat([a, right_pad], axis=1)
    answer_shape = tf.concat([[tf.shape(a)[0], 0, h+r], tf.cast(a.shape[2:], tf.int32)], axis=0)
    i, answer = h-1+r, tf.zeros(answer_shape, dtype=a.dtype)
    cond = lambda i, ans, a, k: i < tf.shape(a_padded)[1]
    body = lambda i, ans, a, k: (
        i+1, tf.concat([ans, tf.expand_dims(a_padded[:,i-k-r+1:i+1], 1)], axis=1), a, k)
    ans_shape = [None, None, h+r] + a_padded.shape[2:].as_list()
    _, answer_, _, _ = tf.while_loop(
        cond, body, [i, answer, a_padded, h+r],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape(ans_shape),
                          a_padded.shape, tf.TensorShape([])])
    if flatten and len(ans_shape) >= 4:
        outer_shape = tf.shape(answer_)
        elem_shape = [ans_shape[2] * ans_shape[3]] + ans_shape[4:]
        new_shape = tf.concat([outer_shape[:2], elem_shape], axis=0)
        answer_ = tf.reshape(answer_, new_shape)
    return answer_


def fill_by_slices(source, start, slice_length, steps, reverse=False):
    """
    Fills a 2D array by consecutive slices of 1D array source
    """
    source = tf.expand_dims(source, 0)
    cond = (lambda i, ans: (i < steps))
    if reverse:
        body = (lambda i, ans:
                (i+1,
                 tf.concat([ans, source[:,start-i:start-i+slice_length]], axis=0)))
    else:
        body = (lambda i, ans:
                (i+1,
                 tf.concat([ans, source[:,start+i:start+i+slice_length]], axis=0)))
    _, answer = tf.while_loop(
        cond, body, [0, tf.zeros(shape=(0, slice_length), dtype=source.dtype)],
        # second dimension is None since input_dim may be an integer value
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None])])
    return answer


def expand_on_edges(a, L, R):
    """
    Expands a by L copies of a[0] on the left and R copies of a[-1] on the right
    """
    left = tf.tile(a[:1], [L] + [1] * ( - 1))
    right = tf.tile(a[-1:], [R] + [1] * (tf.keras.backend.ndim(a) - 1))
    return tf.concat([left, a, right], axis=0)


def expand_number_to_shape(a, b):
    for _ in range(tf.keras.backend.ndim(b) - 1):
        a = tf.expand_dims(a, axis=-1)
    return tf.tile(a, tf.shape(b))


def batch_add_offset_bias(x, q, bias, transpose_bias=True):
    # performs x_{rij} = x_{rij} + dot(q_{ri}, bias_{clip(j-i, -k, k)}),
    # where clip(a, l, r) = max(l, min(a, r)),
    B, T = tf.shape(x)[0], tf.shape(x)[1]
    k = tf.shape(bias)[0] // 2
    bias = expand_on_edges(bias, T, T)
    cond = lambda i, ans: i < T
    def func(x_, q_, b_):
        b_ = tf.tile(tf.expand_dims(b_, 0), [B, 1, 1])  # b_.shape = (B, T, d)
        # q_.shape = (B, 1, d)
        z = tf.matmul(q_, b_, transpose_b=transpose_bias)
        return x_ + z
    body = lambda i, ans: (i+1, tf.concat([ans, func(x[:,i:i+1], q[:,i:i+1], bias[T+k-i:2*T+k-i])], axis=1))
    _, answer = tf.while_loop(cond, body, [0, tf.zeros_like(x[:,:0,:], dtype=x.dtype)],
                              shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None, None])])
    return answer

def generate_future_mask(input_dim):
    ones = tf.ones(shape=(input_dim,), dtype=tf.bool)
    zeros = tf.zeros(shape=(input_dim,), dtype=tf.bool)
    a = tf.concat([ones, zeros], axis=0)
    answer = fill_by_slices(a, input_dim-1, input_dim, input_dim, reverse=True)
    return answer


def test_mask():
    input_dim = 5
    sess = tf.Session()
    with sess.as_default():
        mask = generate_future_mask(input_dim)
        answer = sess.run([mask])
    print(answer)


if __name__ == "__main__":
    test_mask()