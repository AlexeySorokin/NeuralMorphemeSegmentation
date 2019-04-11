import numpy as np
import theano
import theano.tensor as tT
import theano.ifelse as tif

import keras.backend as kb

def make_history_theano(X, h, pad, flatten=False):
    """
    A theano func transforming a 3D array X of shape r*m*d to a 4D array B of shape r*m*h*d,
    where B[s,i,j] = A[s,i-j] if j >= h-1-i and  and B[s,i,j] = pad otherwise

    """
    # B = tT.repeat(kb.zeros_like(kb.expand_dims(X, axis=2)), h, axis=2)
    B = tT.repeat(tT.zeros_like(kb.expand_dims(X, axis=2)), h, axis=2)
    # pad = pad[...,None,:]
    m = X.shape[1]
    def _shifted_fill_vertical(b, d, a, h):
        r1 = tT.set_subtensor(b[:,:d,h-1-d], pad)
        b_new = tif.ifelse(m >= d, tT.set_subtensor(r1[:,d:,h-1-d],a[:,:m-d]), r1)
        return [b_new, d-1]
    results, updates = theano.scan(
        _shifted_fill_vertical, non_sequences=[X, h], outputs_info=[B, h-1], n_steps=h)
    answer = results[0][-1]
    if flatten:
        new_shape = (answer.shape[0], answer.shape[1], -1)
        answer = tT.reshape(answer, new_shape, ndim=3)
    return answer


def make_context_theano(X, h, pad, r=0, right_pad=None):
    """
    A theano func transforming a 3D array X of shape r*m*d to a 3D array B of shape r*m*(h+r)*d,
    where B[s,i,j] = A[s,i+j-(h-1)+r] if j >= h-1-i-r abd  and B[i,j] = pad otherwise

    :param X:
    :param h:
    :param pad:
    :return:
    """
    # B = tT.repeat(kb.zeros_like(X)[:,:,None,:], h+r, axis=2)
    B = kb.repeat_elements(kb.zeros_like(X)[:,:,None,:], h+r, axis=2)
    m = X.shape[1]
    pad, right_pad = pad[...,None,:], right_pad[...,None,:]
    def _shifted_fill(b, d, a, h):
        column_start, column_end = tT.largest(d, 0), m+tT.smallest(0,d)
        start, end = tT.largest(-d,0), tT.smallest(m-d,m)
        b_new = tT.set_subtensor(b[:,:column_start,h-1-d], pad)
        b_new = tT.set_subtensor(b_new[:,column_start:column_end,h-1-d], a[:,start:end])
        b_new = tT.set_subtensor(b_new[:,column_end:,h-1-d], right_pad)
        return [b_new, d-1]
    results, updates = theano.scan(_shifted_fill, non_sequences=[X, h],
                                   outputs_info=[B, h-1], n_steps=h+r)
    return results[0][-1]


## TESTS ##

def test_memory():
    a = tT.dtensor3()
    pad, right_pad = tT.dvector(), tT.dvector()
    # theano.config.scan.debug = True
    # print(theano.config.scan)
    b = make_history_theano(a, 3, pad)
    c = make_history_theano(a, 3, pad)
    f = theano.function([a, pad], [b], on_unused_input="ignore")
    g = theano.function([a, pad], [c], on_unused_input="ignore")
    T, d = 500, 2
    A = np.random.rand(2, T, d)
    B, C = np.ones(shape=(d,), dtype=float), np.zeros(shape=(d,), dtype=float)
    answer = f(A, B)[0]
    other = g(A, B)[0]
    print(np.max(np.fabs(answer-other)))


def test_make_context():
    M, T, D = 32, 5000, 32
    A = np.reshape(np.arange(M*T*D, dtype=float), (M, T, D))
    PAD = np.zeros(shape=(D,), dtype=float)
    RIGHT_PAD = np.ones(shape=(D,), dtype=float)
    a, pad, right_pad = tT.dtensor3(), tT.dvector(), tT.dvector()
    b = make_context_theano(a, 3, pad, 2, right_pad)
    f = theano.function([a, pad, right_pad], [b])
    f(A, PAD, RIGHT_PAD)

if __name__ == "__main__":
    test_make_context()