from keras.models import Model
from keras.engine import Layer

import keras.layers as kl
import keras.backend as kb
from keras.engine.topology import InputSpec


if kb.backend() == "theano":
    from neural.cells_theano import make_history_theano
elif kb.backend() == "tensorflow":
    from neural.common_tensorflow import batch_shifted_fill


def MultiConv1D(inputs, layers, windows, filters, dropout=0.0, activation=None):
    if isinstance(windows, int):
        windows = [windows]
    if isinstance(filters, int):
        filters = [filters]
    if len(filters) == 1 and len(windows) > 1:
        filters = [filters] * len(windows)
    outputs = []
    for window_size, filters_number in zip(windows, filters):
        curr_output = kl.Conv1D(filters_number, window_size,
                                padding="same", activation=activation)(inputs)
        for i in range(layers - 1):
            if dropout > 0.0:
                curr_output = kl.Dropout(dropout)(curr_output)
            curr_output = kl.Conv1D(filters_number, window_size,
                                    padding="same", activation=activation)(curr_output)
        outputs.append(curr_output)
    answer = outputs[0] if (len(outputs) == 1) else kl.Concatenate()(outputs)
    return answer


def calculate_history_shape(shape, h, flatten, only_last=False):
    if len(shape) == 2 or not flatten:
        shape = shape[:2] + (h,) + shape[2:]
    elif shape[2] is not None:
        shape = shape[:2] + (h * shape[2],) + shape[3:]
    else:
        shape = shape[:2] + (None,) + shape[3:]
    if only_last and shape[1] is not None:
        shape = (shape[0], 1) + shape[2:]
    return shape


def make_history(X, h, r, pad, right_pad=None,flatten=False,
                 only_last=False, calculate_keras_shape=False):
    if kb.backend() == "theano":
        answer = make_history_theano(X, h, pad, flatten=flatten)
    else:
        answer = batch_shifted_fill(X, h, pad, r=r, right_pad=right_pad, flatten=flatten)
    if only_last:
        # answer = answer[:,-1:]
        answer = answer[:,-1:]
    if calculate_keras_shape:
        if not hasattr(answer, "_keras_shape") and hasattr(X, "_keras_shape"):
            answer._keras_shape = calculate_history_shape(
                X._keras_shape, h+r, flatten, only_last=only_last)
    return answer


def History(X, h, r=0, flatten=False, only_last=False):
    """
    For each timestep collects h previous elements of the tensor, including current

    X: a Keras tensor, at least 2D, of shape (B, L, ...)
    h: int, history length
    flatten: bool, default=False,
        whether to concatenate h previous elements of the tensor (flatten=True),
        or stack then using a new dimension (flatten=False)
    """
    pad, right_pad = kb.zeros_like(X[0][0]), kb.zeros_like(X[0][0])
    arguments = {"h": h, "r": r, "pad": pad, "right_pad": right_pad,
                 "flatten": flatten, "only_last": only_last}
    output_shape = lambda x: calculate_history_shape(x, h+r, flatten, only_last=only_last)
    return kl.Lambda(make_history, arguments=arguments, output_shape=output_shape)(X)


class AttentionCell(Layer):
    """
    A layer collecting in each position a weighted sum of previous words embeddings
    where weights in the sum are calculated using attention
    """

    def __init__(self, left, input_dim, query_dim=None, output_dim=None,
                 right=0, merge="concatenate",  use_bias=False, **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = (None, None, input_dim)
        super(AttentionCell, self).__init__(**kwargs)
        self.left = left
        self.right = right
        self.merge = merge
        self.use_bias = use_bias
        self.input_dim = input_dim
        self.query_dim = query_dim or self.input_dim
        self.output_dim = output_dim or self.input_dim
        self.input_spec = InputSpec(shape=(None, None, input_dim))

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.M = self.add_weight(shape=(self.input_dim, self.query_dim),
                                 name='attention_embedding_1', dtype=self.dtype,
                                 initializer="glorot_uniform")
        self.C = self.add_weight(shape=(self.input_dim, self.query_dim),
                                 name='attention_embedding_2', dtype=self.dtype,
                                 initializer="glorot_uniform")
        if self.use_bias:
            self.T = self.add_weight(shape=(self.left, self.query_dim),
                                     name='bias', dtype=self.dtype,
                                     initializer="glorot_uniform")
        self.V = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 name='attention_embedding_2', dtype=self.dtype,
                                 initializer="glorot_uniform")
        super(AttentionCell, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # contexts.shape = (M, T, left, d)
        queries, keys = kb.dot(inputs, self.M), kb.dot(inputs, self.C)
        pad = kb.zeros_like(keys[0][0])
        keys = make_history(keys, self.left, pad=pad, r=self.right, right_pad=pad, flatten=False)
        if self.use_bias:
            keys += self.T
        values = kb.dot(inputs, self.V)
        values = make_history(values, self.left, pad=kb.zeros_like(values[0][0]),
                              r=self.right, right_pad=kb.zeros_like(values[0][0]),
                              flatten=False)
        answer = local_dot_attention(values, queries, keys)
        if self.merge == "concatenate":
            answer = kb.concatenate([inputs, answer], axis=-1)
        elif self.merge == "sum":
            answer += inputs
        return answer

    def compute_output_shape(self, input_shape):
        last_dim = (self.input_dim if self.merge == "sum" else
                    self.input_dim+self.output_dim if self.merge == "concatenate" else
                    self.output_dim)
        return input_shape[:-1] + (last_dim,)


def attention_func(inputs, only_last=False, **kwargs):
    answer = AttentionCell(**kwargs)(inputs)
    if not only_last:
        return answer
    return kl.Lambda(lambda x: x[:,-1:])(answer)



def TemporalDropout(inputs, dropout=0.0):
    """
    Drops with :dropout probability temporal steps of input 3D tensor
    """
    # TO DO: adapt for >3D tensors
    if dropout == 0.0:
        return inputs
    inputs_func = lambda x: kb.ones_like(inputs[:, :, 0:1])
    inputs_mask = kl.Lambda(inputs_func)(inputs)
    inputs_mask = kl.Dropout(dropout)(inputs_mask)
    tiling_shape = [1, 1, kb.shape(inputs)[2]] + [1] * (kb.ndim(inputs) - 3)
    inputs_mask = kl.Lambda(kb.tile, arguments={"n": tiling_shape},
                            output_shape=inputs._keras_shape[1:])(inputs_mask)
    answer = kl.Multiply()([inputs, inputs_mask])
    return answer

def local_dot_attention(values, queries, keys, normalize_logits=False):
    input_length = kb.shape(queries)[1]
    queries = kb.reshape(queries, (-1, queries.shape[2]))
    keys = kb.reshape(keys, (-1, keys.shape[-2], keys.shape[-1]))
    values = kb.reshape(values, (-1, values.shape[-2], values.shape[-1]))
    logits = kb.batch_dot(keys, queries, axes=[2, 1])
    if normalize_logits:
        logits /= kb.sqrt(kb.cast(kb.shape(keys)[-1], "float32"))
    scores = kb.softmax(logits)
    tiled_scores = kb.tile(scores[:,:,None], [1, 1, kb.shape(values)[-1]])
    answer = kb.sum(tiled_scores * values, axis=1)
    answer = kb.reshape(answer, (-1, input_length, answer.shape[-1]))
    return answer

def LocalAttention(inputs, keys_size, values_size, h, r, activation=None):
    queries = kl.Dense(keys_size, activation=activation)(inputs)
    keys = kl.Dense(keys_size, activation=activation)(inputs)
    window_keys = History(keys, h, r, flatten=False)
    values = kl.Dense(values_size, activation=activation)(inputs)
    window_values = History(values, h, r, flatten=False)
    # answer = local_dot_attention(queries, window_keys, window_values)
    # output_shape = lambda x: x[:-1] + (values_size,)
    return kl.Lambda(local_dot_attention, arguments={"queries": queries, "keys": window_keys})(window_values)
