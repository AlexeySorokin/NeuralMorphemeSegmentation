import bisect
from itertools import chain

import numpy as np
import keras.backend as kb


EPS = 1e-10
BOW, EOW, STEP, COPY = "BEGIN", "END", "STEP", "COPY"
PAD, BEGIN, END, UNKNOWN, STEP_CODE, COPY_CODE = 0, 1, 2, 3, 4, 5
AUXILIARY = ['PAD', BOW, EOW, 'UNKNOWN']

def make_input_with_copy_symbol(source):
    prev_step_indexes = kb.concatenate(
        [kb.ones_like(source[:,:1], dtype="bool"), kb.equal(source[:, :-1], STEP_CODE)])
    prev_step_indexes = kb.cast(prev_step_indexes, "int32")
    prev_step_indexes *= kb.cast(kb.not_equal(source, STEP_CODE), "int32")
    prev_step_indexes *= kb.cast(kb.not_equal(source, END), "int32")
    copy_matrix = kb.ones_like(source) * COPY_CODE
    answer = kb.switch(prev_step_indexes, copy_matrix, source)
    return answer

def to_one_hot(indices, num_classes):
    """
    :param indices: np.array, dtype=int
    :param num_classes: int, число классов
    :return: answer, np.array, shape=indices.shape+(num_classes,)
    """
    shape = indices.shape
    indices = np.ravel(indices)
    answer = np.zeros(shape=(indices.shape[0], num_classes), dtype=int)
    answer[np.arange(indices.shape[0]), indices] = 1
    return answer.reshape(shape+(num_classes,))


def make_bucket_lengths(lengths, buckets_number=None, max_bucket_length=None):
    m = len(lengths)
    lengths = sorted(lengths)
    last_bucket_length, bucket_lengths = 0, []
    if buckets_number is None:
        if max_bucket_length is not None:
            buckets_number = (m - 1) // max_bucket_length + 1
        else:
            raise ValueError("Either buckets_number or max_bucket_length must be given.")
    for i in range(buckets_number):
        # могут быть проблемы с выбросами большой длины
        level = (m * (i + 1) // buckets_number) - 1
        curr_length = lengths[level]
        if curr_length > last_bucket_length:
            bucket_lengths.append(curr_length)
            last_bucket_length = curr_length
    return bucket_lengths


def collect_buckets(lengths, buckets_number=None, max_bucket_length=-1):
    bucket_lengths = make_bucket_lengths(lengths, buckets_number, max_bucket_length)
    indexes = [[] for length in bucket_lengths]
    for i, length in enumerate(lengths):
        index = bisect.bisect_left(bucket_lengths, length)
        indexes[index].append(i)
    if max_bucket_length != -1:
        bucket_lengths = list(chain.from_iterable(
            ([L] * ((len(curr_indexes)-1) // max_bucket_length + 1))
            for L, curr_indexes in zip(bucket_lengths, indexes)
            if len(curr_indexes) > 0))
        indexes = [curr_indexes[start:start+max_bucket_length]
                   for curr_indexes in indexes
                   for start in range(0, len(curr_indexes), max_bucket_length)]
    return [(L, curr_indexes) for L, curr_indexes
            in zip(bucket_lengths, indexes) if len(curr_indexes) > 0]


def make_table(data, indexes, fill_value=None, fill_with_last=False, length=None):
    """
    Погружает строки data с номерами из indexes
    в таблицу ширины length, дополняя их справа
    либо значением fill_value, либо последним значением, увеличенным на 1

    letter_positions: list of lists of int
    length: int
    indexes: list of ints
    """
    if length is None:
        length = max(len(data[i]) for i in indexes)
    answer = np.zeros(shape=(len(indexes), length), dtype=int)
    if fill_value is not None:
        answer.fill(fill_value)
    for i, index in enumerate(indexes):
        curr = data[index]
        L = len(curr)
        answer[i,:L] = curr
        if fill_with_last:
            answer[i,L:] = curr[-1] + 1
    return answer


def generate_data(X, indexes_by_buckets, batches_indexes, batch_size,
                  symbols_number, auxiliary_symbols_number=None,
                  inputs_number=None, shuffle=True,
                  weights=None, weight_indexes=None, nepochs=None):
    inputs_number = inputs_number or (len(X[0]) - 1)
    auxiliary_symbols_number = auxiliary_symbols_number or []
    nsteps = 0
    while nepochs is None or nsteps < nepochs:
        if shuffle:
            for elem in indexes_by_buckets:
                np.random.shuffle(elem)
            np.random.shuffle(batches_indexes)
        for i, start in batches_indexes:
            curr_bucket, bucket_size = X[i], len(X[i][0])
            end = min(bucket_size, start + batch_size)
            curr_indexes = indexes_by_buckets[i][start:end]
            to_yield = [elem[curr_indexes] for elem in curr_bucket[:inputs_number]]
            y_to_yield = [elem[curr_indexes] for elem in curr_bucket[inputs_number:]]
            # веса объектов
            # преобразуем y_to_yield в бинарный формат
            if y_to_yield[0].ndim == 2:
                y_to_yield[0] = to_one_hot(y_to_yield[0], symbols_number)
            for i, value in auxiliary_symbols_number:
                y_to_yield[i] = to_one_hot(y_to_yield[i], value)
            for i, elem in enumerate(y_to_yield[1:], 1):
                if elem.ndim == 2:
                    y_to_yield[i] = y_to_yield[i][:,:,None]
            # yield (to_yield, y_to_yield, weights_to_yield)
            if weights is None:
                yield (to_yield, y_to_yield)
            else:
                if callable(weights):
                    weights_to_yield = weights(*(curr_bucket[j][curr_indexes] for j in weight_indexes))
                else:
                    weights_to_yield = weights[i][curr_indexes]
                yield (to_yield, y_to_yield, weights_to_yield)
        nsteps += 1