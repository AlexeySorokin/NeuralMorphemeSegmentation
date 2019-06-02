"""
Эксперименты с языковыми моделями
"""
import sys
import os
import getopt
from collections import defaultdict
import bisect
import copy
import itertools
import inspect
import json
# import ujson as json

import numpy as np
np.set_printoptions(precision=3)

import keras
import keras.optimizers as ko
import keras.backend as kb
import keras.layers as kl
from keras.models import Model
import keras.callbacks as kcall
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .common import *
from .common import generate_data
from .common_neural import make_useful_symbols_mask, PerplexitywithNegatives, ce_with_negatives
from .vocabulary import Vocabulary, vocabulary_from_json
from .cells import History, AttentionCell, attention_func

def to_one_hot(indices, num_classes):
    """
    Theano implementation for numpy arrays

    :param indices: np.array, dtype=int
    :param num_classes: int, число классов
    :return: answer, np.array, shape=indices.shape+(num_classes,)
    """
    shape = indices.shape
    indices = np.ravel(indices)
    answer = np.zeros(shape=(indices.shape[0], num_classes), dtype=int)
    answer[np.arange(indices.shape[0]), indices] = 1
    return answer.reshape(shape+(num_classes,))

def read_input(infile, label_field=None, max_num=-1):
    answer = []
    feats_column = 2 if label_field is None else 1
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            if i == max_num:
                break
            line = line.strip()
            if line == "":
                continue
            splitted = line.split()
            curr_elem = [splitted[2]]
            feats = splitted[feats_column] if len(splitted) > feats_column else ""
            feats = [x.split("=") for x in feats.split(",")]
            feats = {x[0]: x[1] for x in feats}
            if label_field is not None:
                label = feats.pop(label_field, None)
            else:
                label = splitted[1] if len(splitted) > 1 else None
            if label is not None:
                curr_elem.append(label)
                curr_elem.append(feats)
            answer.append(curr_elem)
    return answer


def make_bucket_indexes(lengths, buckets_number=None,
                        bucket_size=None, join_buckets=True):
    if buckets_number is None and bucket_size is None:
        raise ValueError("Either buckets_number or bucket_size should be given")
    indexes = np.argsort(lengths)
    lengths = sorted(lengths)
    m = len(lengths)
    if buckets_number is not None:
        level_indexes = [m * (i+1) // buckets_number for i in range(buckets_number)]
    else:
        level_indexes = [min(start+bucket_size, m) for start in range(0, m, bucket_size)]
    if join_buckets:
        new_level_indexes = []
        for i, index in enumerate(level_indexes[:-1]):
            if lengths[index-1] < lengths[level_indexes[i+1]-1]:
                new_level_indexes.append(index)
        level_indexes = new_level_indexes + [m]
    bucket_indexes =  [indexes[start:end] for start, end in
                       zip([0] + level_indexes[:-1], level_indexes)]
    bucket_lengths = [lengths[i-1] for i in level_indexes]
    return bucket_indexes, bucket_lengths


def decode_label(label):
    if isinstance(label, tuple):
        return (label[0], dict(label[1]))
    return label.split("_")


class NeuralLM:

    def __init__(self, reverse=False, min_symbol_count=1,
                 batch_size=32, nepochs=20, validation_split=0.2,
                 use_bigram_loss=False, bigram_loss_weight=0.5,
                 bigram_loss_threshold=0.005,
                 use_label=False, use_feats=False, use_full_tags=False,
                 history=1, use_attention=False, attention_activation="concatenate",
                 use_attention_bias=False,
                 use_embeddings=False, embeddings_size=16,
                 feature_embedding_layers=False, feature_embeddings_size=32,
                 rnn="lstm", encoder_rnn_size=64,
                 decoder_rnn_size=64, decoder_layers=1, dense_output_size=32,
                 embeddings_dropout=0.0, encoder_dropout=0.0, decoder_dropout=0.0,
                 random_state=187, verbose=1, callbacks=None,
                 # use_custom_callback=False
                 ):
        self.reverse = reverse
        self.min_symbol_count = min_symbol_count
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.use_bigram_loss = use_bigram_loss
        self.bigram_loss_weight = bigram_loss_weight
        self.bigram_loss_threshold = bigram_loss_threshold
        self.use_label = use_label
        self.use_feats = use_feats
        self.use_full_tags = use_full_tags
        self.history = history
        self.use_attention = use_attention
        self.attention_activation = attention_activation
        self.use_attention_bias = use_attention_bias
        self.use_embeddings = use_embeddings
        self.embeddings_size = embeddings_size
        self.feature_embedding_layers = feature_embedding_layers
        self.feature_embeddings_size = feature_embeddings_size
        # self.rnn = rnn
        self.encoder_rnn_size = encoder_rnn_size
        self.decoder_rnn_size = decoder_rnn_size
        self.decoder_layers = decoder_layers
        self.dense_output_size = dense_output_size
        # self.dropout = dropout
        self.embeddings_dropout = embeddings_dropout
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.random_state = random_state
        self.verbose = verbose
        # if self.use_custom_callback:
        #     self.callbacks.append(CustomCallback())
        self.initialize(callbacks)

    def initialize(self, callbacks=None):
        # if isinstance(self.rnn, str):
        #     self.rnn = getattr(kl, self.rnn.upper())
        # if self.rnn not in [kl.GRU, kl.LSTM]:
        #     raise ValueError("Unknown recurrent network: {}".format(self.rnn))
        callbacks = callbacks or dict()
        self.callbacks = [getattr(kcall, key)(**params) for key, params in callbacks.items()]

    def to_json(self, outfile, model_file):
        info = dict()
        # model_file = os.path.abspath(model_file)
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(NeuralLM, attr, None), property) or
                    isinstance(val, Vocabulary) or attr.isupper() or
                    attr in ["callbacks", "model_", "tag_codes_"] or
                    attr.endswith("func_")):
                info[attr] = val
            elif isinstance(val, Vocabulary):
                info[attr] = val.jsonize()
            elif attr == "model_":
                info["dump_file"] = model_file
                self.model_.save_weights(model_file)
            elif attr == "callbacks":
                callback_info = dict()
                for callback in val:
                    if isinstance(callback, EarlyStopping):
                        callback_info["EarlyStopping"] = \
                            {key: getattr(callback, key) for key in ["patience", "monitor", "min_delta"]}
                info["callbacks"] = callback_info
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    def make_features(self, X):
        """
        :param X: list of lists,
            создаёт словарь на корпусе X = [x_1, ..., x_m]
            x_i = [w_i, (c_i, (feats_i)) ], где
                w: str, строка из корпуса
                c: str(optional), класс строки (например, часть речи)
                feats: dict(optional), feats = {f_1: v_1, ..., f_k: v_k},
                    набор пар признак-значение (например, значения грамматических категорий)
        :return: self, обученная модель
        """
        # первый проход: определяем длины строк, извлекаем словарь символов и признаков
        labels, features, tags = set(), set(), set()
        for elem in X:
            # print(elem)            
            if len(elem) > 1:
                if isinstance(elem[1], (list, tuple)):
                    label, feats, tag = elem[1][0], elem[1][1:], tuple(elem[1])
                else:
                    label = elem[1]
                    if len(elem) > 1:
                        feats, tag = elem[2], (label,) + tuple(elem[2])
                    else:
                        feats, tag = None, elem[1]
                labels.add(label)
                if self.use_feats and feats is not None:
                    if isinstance(feats, dict):
                        for feature, value in feats.items():
                            features.add("_".join([label, feature, value]))
                    else:
                        features.update({"{}_{}".format(label, elem) for elem in feats})
                tags.add(tuple(tag))
        # создаём словари нужного размера
        self.labels_ = AUXILIARY + sorted(labels) + sorted(features)
        self.label_codes_ = {x: i for i, x in enumerate(self.labels_)}
        if self.use_full_tags:
            self.tags_ = sorted(tags)
            self.tag_codes_ = {x: i for i, x in enumerate(self.tags_)}
        return self

    @property
    def symbols_number_(self):
        return self.vocabulary_.symbols_number_

    @property
    def labels_number_(self):
        if self.use_label:
            answer = len(self.labels_)
            if self.use_full_tags:
                answer += len(self.tags_)
            return answer
        else:
            return 0

    def toidx(self, x):
        return self.vocabulary_.toidx(x)

    def _make_word_vector(self, word, bucket_length=None, symbols_has_features=False):
        """
        :param word:
        :param pad:
        :return:
        """
        m = len(word)
        if bucket_length is None:
            bucket_length = m + 2
        answer = np.full(shape=(bucket_length,), fill_value=PAD, dtype="int32")
        answer[0], answer[m+1] = BEGIN, END
        for i, x in enumerate(word, 1):
            answer[i] = self.vocabulary_.toidx(x)
        return answer

    def _make_feature_vector(self, label, feats=None):
        if isinstance(label, list) or isinstance(label, tuple):
            label, feats = label[0], label[1:]
            if len(feats) == 1 and isinstance(feats[0], dict):
                feats = feats[0]
        answer = np.zeros(shape=(self.labels_number_,))
        label_code = self.label_codes_.get(label, UNKNOWN)
        answer[label_code] = 1
        if label_code != UNKNOWN:
            if isinstance(feats, dict):
                feats = ["{}_{}_{}".format(label, *elem) for elem in feats.items()]
            else:
                feats = ["{}_{}".format(label, elem) for elem in feats]
            for feature in feats:
                feature_code = self.label_codes_.get(feature)
                if self.use_feats and feature_code is not None:
                    answer[feature_code] = 1
        return answer

    def _get_symbol_features_codes(self, symbol, feats):
        symbol_code = self.vocabulary_.get_feature_code(symbol)
        answer = [symbol_code]
        if symbol_code == UNKNOWN:
            return answer
        for feature, value in feats:
            if feature != "token":
                feature_repr = "{}_{}_{}".format(symbol, feature, value)
                symbol_code = self.vocabulary_.get_feature_code(feature_repr)
            else:
                symbol_code = self.vocabulary_.get_token_code(value)
            if symbol_code is not None:
                answer.append(symbol_code)
        return answer

    def _make_bucket_data(self, lemmas, bucket_length, bucket_indexes):
        bucket_size = len(bucket_indexes)
        bucket_data = np.full(shape=(bucket_size, bucket_length), fill_value=PAD, dtype=int)
        # заполняем закодированными символами
        bucket_data[:,0] = BEGIN
        for j, i in enumerate(bucket_indexes):
            lemma = lemmas[i]
            bucket_data[j,1:1+len(lemma)] = [self.vocabulary_.toidx(x) for x in lemma]
            bucket_data[j,1+len(lemma)] = END
        return bucket_data

    def transform(self, X, return_indexes=True, buckets_number=10, max_bucket_length=-1):
        lengths = [len(x[0])+2 for x in X]
        buckets_with_indexes = collect_buckets(lengths, buckets_number=buckets_number,
                                               max_bucket_length=max_bucket_length)
        data = [elem[0] for elem in X]
        if self.reverse:
            data = [elem[::-1] for elem in data]
        # print(data[:10])
        data_by_buckets = [[self._make_bucket_data(data, length, indexes)]
                           for length, indexes in buckets_with_indexes]
        if self.use_label:
            features = np.array([self._make_feature_vector(*elem[1:]) for elem in X])
            features_by_buckets = [features[indexes] for _, indexes in buckets_with_indexes]
            for i, elem in enumerate(features_by_buckets):
                data_by_buckets[i].append(elem)
        for i, elem in enumerate(data_by_buckets):
            curr_answer = np.concatenate([elem[0][:,1:], PAD*np.ones_like(elem[0][:,-1:])], axis=1)
            elem.append(curr_answer)
        if return_indexes:
            return data_by_buckets, [elem[1] for elem in buckets_with_indexes]
        else:
            return data_by_buckets

    def train(self, X, X_dev=None, model_file=None, save_file=None):
        np.random.seed(self.random_state)  # initialize the random number generator
        self.vocabulary_ = Vocabulary(self.min_symbol_count).train([elem[0] for elem in X])
        if self.use_label:
            self.make_features(X)
        else:
            self.labels_, self.label_codes_ = None, None
        X_train, indexes_by_buckets = self.transform(X)
        if self.use_bigram_loss:
            self._make_bigrams(X_train)
            for i, elem in enumerate(X_train):
                elem[-1] = to_one_hot(elem[-1], self.symbols_number_)
                elem[-1] -= self.bigram_mask_[elem[0]]
        # self._make_statistics(X)
        if X_dev is not None:
            X_dev, dev_indexes_by_buckets = self.transform(X_dev, max_bucket_length=256)
        else:
            X_dev, dev_indexes_by_buckets = None, None
        self.build()
        if save_file is not None:
            if model_file is None:
                pos = save_file.rfind(".")
                if pos == -1:
                    model_file = save_file + ".hdf5"
                else:
                    model_file = save_file[:pos] + ".hdf5"
            self.to_json(save_file, model_file)
        self.train_model(X_train, X_dev, model_file=model_file)
        return self

    def _make_bigrams(self, X):
        bigram_mask = np.zeros(shape=(self.symbols_number_, self.symbols_number_), dtype=int)
        for bucket in X:
            for word in bucket[0]:
                for i, symbol in enumerate(word[:-1]):
                    bigram_mask[symbol, word[i+1]] = 1
        self.bigram_mask_ = 1 - bigram_mask
        return self

    def _make_statistics(self, X):
        self.max_length_ = max(10, max(len(elem[0]) for elem in X))
        if self.use_label:
            label_counts = defaultdict(int)
            if self.use_feats:
                for _, label, feats in X:
                    label = (label, tuple(feats.items()))
                    label_counts[label] += 1
            else:
                for _, label in X:
                    if not isinstance(label, str):
                        label_counts["_".join(label)] += 1
            self.label_counts_ = dict()
            total_count = sum(label_counts.values())
            for key, value in label_counts.items():
                self.label_counts_[key] = value / total_count
        return label_counts

    def build(self, test=False, verbose=1):
        symbol_inputs = kl.Input(shape=(None,), dtype='int32')
        symbol_embeddings, mask = self._build_symbol_layer(symbol_inputs)
        memory, initial_encoder_states, final_encoder_states =\
            self._build_history(symbol_embeddings, only_last=test)
        if self.labels_ is not None:
            feature_inputs = kl.Input(shape=(self.labels_number_,))
            inputs = [symbol_inputs, feature_inputs]
            tiled_feature_embeddings =\
                self._build_feature_network(feature_inputs, kb.shape(memory)[1])
            to_decoder = kl.Concatenate()([memory, tiled_feature_embeddings])
        else:
            inputs, to_decoder = [symbol_inputs], memory
        # lstm_outputs = kl.LSTM(self.rnn_size, return_sequences=True, dropout=self.dropout)(to_decoder)
        outputs, initial_decoder_states,\
            final_decoder_states, lstm_outputs, pre_outputs = self._build_output_network(to_decoder)
        if self.use_bigram_loss:
            loss = PerplexitywithNegatives(self.bigram_loss_threshold, self.bigram_loss_weight)
            metrics = [ce_with_negatives]
        else:
            loss, metrics  = "categorical_crossentropy", []
        compile_args = {"optimizer": ko.nadam(clipnorm=5.0), "loss": loss, "metrics": metrics}
        self.model_ = Model(inputs, outputs)
        self.model_.compile(**compile_args)
        if verbose > 0 and self.verbose > 0:
            print(self.model_.summary())
        step_func_inputs = inputs + initial_decoder_states + initial_encoder_states
        step_func_outputs = [lstm_outputs] + final_decoder_states + final_encoder_states
        logit_func_outputs = [pre_outputs] + final_decoder_states + final_encoder_states
        self._step_func_ = kb.Function(step_func_inputs + [kb.learning_phase()], step_func_outputs)
        self._output_step_func_ = kb.Function(step_func_inputs + [kb.learning_phase()],
                                             [outputs] + final_decoder_states + final_encoder_states)
        self._logit_step_func_ = kb.Function(step_func_inputs + [kb.learning_phase()], logit_func_outputs)
        self._state_func_ = kb.Function(inputs + [kb.learning_phase()], [lstm_outputs])
        self._logit_func_ = kb.Function(inputs + [kb.learning_phase()], [pre_outputs])
        self._head_func = kb.Function([lstm_outputs], [outputs])
        self.built_ = "test" if test else "train"
        return self

    def _build_symbol_layer(self, symbol_inputs):
        useful_symbols_mask = make_useful_symbols_mask(symbol_inputs, dtype="float32")
        if self.use_embeddings:
            answer = kl.Embedding(self.symbols_number_, self.embeddings_size)(symbol_inputs)
            if self.embeddings_dropout > 0.0:
                answer = kl.Dropout(self.embeddings_dropout)(answer)
        else:
            answer = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number_},
                               output_shape=(None, self.symbols_number_))(symbol_inputs)
        answer = kl.Lambda(lambda x, y: x*y[...,None], arguments={"y": useful_symbols_mask})(answer)
        return answer, useful_symbols_mask

    def _build_history(self, inputs, only_last=False):
        if not self.use_attention:
            memory = History(inputs, self.history, flatten=True, only_last=only_last)
            return memory, [], []
        initial_states = [kb.zeros_like(inputs[:, 0, 0]), kb.zeros_like(inputs[:, 0, 0])]
        for i, elem in enumerate(initial_states):
            initial_states[i] = kb.tile(elem[:, None], [1, self.encoder_rnn_size])
        # TO DO: будут проблемы с историей при наличии attention
        encoder = kl.LSTM(self.encoder_rnn_size, return_sequences=True, return_state=True)
        lstm_outputs, final_c_states, final_h_states = encoder(inputs, initial_state=initial_states)
        attention_params = {"left": self.history, "input_dim": self.encoder_rnn_size,
                            "merge": self.attention_activation, "use_bias": self.use_attention_bias}
        memory = attention_func(lstm_outputs, only_last=only_last, **attention_params)
        return memory, initial_states, [final_c_states, final_h_states]

    def _build_feature_network(self, inputs, k):
        if self.feature_embedding_layers:
            inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(inputs)
            inputs = kl.Dense(self.feature_embeddings_size,
                              activation="relu", use_bias=False)(inputs)
            for _ in range(1, self.feature_embedding_layers):
                inputs = kl.Dense(self.feature_embeddings_size,
                                  input_shape=(self.feature_embeddings_size,),
                                  activation="relu", use_bias=False)(inputs)
        def tiling_func(x):
            x = kb.expand_dims(x, 1)
            return kb.tile(x, [1, k, 1])
        answer = kl.Lambda(tiling_func, output_shape=(lambda x: (None,) + x))(inputs)
        answer = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(answer)
        return answer

    def _build_output_network(self, inputs):
        initial_states = [kb.zeros_like(inputs[:, 0, 0]) for _ in range(2*self.decoder_layers)]
        for i, elem in enumerate(initial_states):
            L = None # if i < 128 else self.decoder_rnn_size
            initial_states[i] = kb.tile(elem[:, L], [1, self.encoder_rnn_size])
        lstm_inputs, lstm_outputs = [None] * self.decoder_layers, [None] * self.decoder_layers
        lstm_inputs[0] = inputs
        final_states = []
        for i in range(self.decoder_layers):
            decoder = kl.LSTM(self.decoder_rnn_size, return_sequences=True, return_state=True)
            lstm_outputs[i], final_c_states, final_h_states = decoder(lstm_inputs[i], initial_state=initial_states[2*i:2*i+2])
            final_states.extend([final_c_states, final_h_states])
            if i < self.decoder_layers - 1:
                lstm_inputs[i+1] = lstm_outputs[i]
        pre_outputs = kl.Dense(self.dense_output_size, activation="relu")(lstm_outputs[-1])
        outputs = kl.TimeDistributed(
            kl.Dense(self.symbols_number_, activation="softmax"))(pre_outputs)
        return outputs, initial_states, final_states, lstm_outputs[-1], pre_outputs

    def rebuild(self, test=True):
        if self.built_ != "test":
            weights = self.model_.get_weights()
            self.build(test=test)
            self.model_.set_weights(weights)
        return self

    def train_model(self, X, X_dev=None, model_file=None):
        train_indexes_by_buckets, dev_indexes_by_buckets = [], []
        for curr_data in X:
            curr_indexes = list(range(len(curr_data[0])))
            np.random.shuffle(curr_indexes)
            if X_dev is None:
                # отделяем в каждой корзине данные для валидации
                train_bucket_size = int((1.0 - self.validation_split) * len(curr_indexes))
                train_indexes_by_buckets.append(curr_indexes[:train_bucket_size])
                dev_indexes_by_buckets.append(curr_indexes[train_bucket_size:])
            else:
                train_indexes_by_buckets.append(curr_indexes)
        if model_file is not None:
            callback = ModelCheckpoint(model_file, save_weights_only=True,
                                       save_best_only=True, verbose=0)
            if self.callbacks is not None:
                self.callbacks.append(callback)
            else:
                self.callbacks = [callback]
        if X_dev is not None:
            for curr_data in X_dev:
                dev_indexes_by_buckets.append(list(range(len(curr_data[0]))))
        train_batches_indexes = list(chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), self.batch_size))
             for i, bucket in enumerate(train_indexes_by_buckets))))
        dev_batches_indexes = list(chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), self.batch_size))
             for i, bucket in enumerate(dev_indexes_by_buckets))))
        if X_dev is None:
            X_dev = X
        train_gen = generate_data(X, train_indexes_by_buckets, train_batches_indexes,
                                  self.batch_size, self.symbols_number_)
        val_gen = generate_data(X_dev, dev_indexes_by_buckets, dev_batches_indexes,
                                self.batch_size, self.symbols_number_, shuffle=False)
        self.model_.fit_generator(
            train_gen, len(train_batches_indexes), epochs=self.nepochs,
            callbacks=self.callbacks, verbose=self.verbose, validation_data=val_gen,
            validation_steps=len(dev_batches_indexes))
        if model_file is not None:
            self.model_.load_weights(model_file)
        return self

    def _score_batch(self, bucket, answer, lengths, batch_size=1, return_array=False):
        """
        :Arguments
         batch: list of np.arrays, [data, (features)]
            data: shape=(batch_size, length)
            features(optional): shape=(batch_size, self.feature_vector_size)
        :return:
        """
        bucket_size, length = bucket[0].shape[:2]
        answers = to_one_hot(answer, self.symbols_number_)
        # last two scores are probabilities of word end and final padding symbol
        scores = self.model_.predict(bucket, batch_size=batch_size)
        scores = np.clip(scores, EPS, 1.0 - EPS)
        losses = -np.sum(answers * np.log(scores), axis=-1)
        total = np.sum(losses, axis=1) # / np.log(2.0)
        letter_scores = scores[np.arange(bucket_size)[:,np.newaxis],
                               np.arange(length)[np.newaxis,:], answer]
        if not return_array:
            letter_scores = [elem[:length] for elem, length in zip(letter_scores, lengths)]
        if self.reverse:
            letter_scores = [elem[::-1] for elem in letter_scores]
        return letter_scores, total

    def score(self, x, **args):
        return self.predict([x], batch_size=1, **args)

    def sample(self, n, batch_size=32, max_length=None, labels=None):
        if max_length is None:
            max_length = self.max_length_ + 5
        if labels is None:
            possible_labels = list(map(decode_label, self.label_counts_))
            probs = list(self.label_counts_.values())
            probs = np.cumsum(probs)
        answer = []
        np.random.seed(self.random_state)
        while len(answer) < n:
            start = len(answer)
            batch_size = min(n - len(answer), batch_size)
            end = start + batch_size
            if self.use_label:
                if labels is None:
                    random_numbers = np.random.uniform(0, 1, batch_size)
                    indexes = [bisect.bisect_left(probs, x) for x in random_numbers]
                    batch_labels = [possible_labels[i] for i in indexes]
                else:
                    batch_labels = list(map(decode_label, labels[start:end]))
                batch_features = np.array([self._make_feature_vector(x) for x in batch_labels])
            else:
                batch_features = None
            indexes, lemmas = self._sample_batch(batch_size, max_length, batch_features)
            for index, lemma in zip(indexes, lemmas):
                if self.use_label:
                    answer.append([lemma, batch_labels[index]])
                else:
                    answer.append(lemma)
                if len(answer) == n:
                    break
        return answer

    def _sample_batch(self, batch_size, max_length, features=None, slow=False):
        if slow:
            history = np.zeros(shape=(batch_size, max_length), dtype="int")
            history[:,0] = BEGIN
        else:
            self.rebuild(test=True)
            history = np.zeros(shape=(batch_size, self.history), dtype="int")
            history[:,-1] = BEGIN
        words = ["" for _ in range(batch_size)]
        indexes_mask = np.ones(dtype=bool, shape=(batch_size,))
        if not slow:
            if self.use_attention:
                encoder_states = list(np.zeros(shape=(2, batch_size, self.encoder_rnn_size), dtype="float"))
            else:
                encoder_states = []
            decoder_states = list(np.zeros(shape=(2, batch_size, self.decoder_rnn_size), dtype="float"))
        for i in range(max_length):
            active_indexes = np.where(indexes_mask)[0]
            if len(active_indexes) == 0:
                break
            if self.use_label:
                inputs = [history[indexes_mask], features[indexes_mask]]
            else:
                inputs = [history[indexes_mask]]
            if slow:
                probs = self.model_.predict(inputs)[:,i]
            else:
                states = [elem[active_indexes] for elem in encoder_states + decoder_states]
                answer = self._output_step_func_(inputs + states + [0])
                probs, new_encoder_states, new_decoder_states = answer[0][:,0], answer[1:-2], answer[-2:]
            # probs[:, END] += np.sum(probs[:, [PAD, BEGIN, UNKNOWN]], axis=-1)
            # probs = np.where(probs < 0.1, 0.0, probs)
            probs[:, [PAD, BEGIN, UNKNOWN]] = 0.0
            cumulative_probs = np.cumsum(probs, axis=-1)
            random_scores = np.random.uniform(0, cumulative_probs[:,-1], probs.shape[0])
            letter_indexes = [bisect.bisect(scores, x) for scores, x in zip(cumulative_probs, random_scores)]
            if slow:
                if i < max_length - 1:
                    history[active_indexes, i+1] = letter_indexes
            else:
                history[active_indexes,:-1] = history[active_indexes,1:]
                history[active_indexes, -1] = letter_indexes
                if self.use_attention:
                    encoder_states[0][active_indexes] = new_encoder_states[0]
                    encoder_states[1][active_indexes] = new_encoder_states[1]
                decoder_states[0][active_indexes] = new_decoder_states[0]
                decoder_states[1][active_indexes] = new_decoder_states[1]
            for index, letter in zip(active_indexes, letter_indexes):
                if letter == END:
                    if self.reverse:
                        words[index] = words[index][::-1]
                    indexes_mask[index] = False
                else:
                    words[index] += self.vocabulary_[letter]
        indexes = np.where(np.logical_not(indexes_mask))[0]
        return indexes, [words[i] for i in indexes]


    def predict(self, X, batch_size=32, return_letter_scores=False,
                return_log_probs=False, return_exp_total=False):
        """

        answer = [answer_1, ..., answer_m]
        answer_i =
            (letter_score_i, total_score_i), если return_letter_scores = True,
            total_score, если return_letter_scores = False
        letter_score_i = [p_i0, ..., p_i(l_i-1), p_i(END)]
        p_ij --- вероятность j-ого элемента в X[i]
            (логарифм вероятности, если return_log_probs=True)


        Вычисляет логарифмические вероятности для всех слов в X,
        а также вероятности отдельных символов
        """
        fields_number = 2 if self.labels_ is not None else 1
        X_test, indexes = self.transform(X, max_bucket_length=batch_size)
        answer = [None] * len(X)
        lengths = np.array([len(x[0]) + 1 for x in X])
        for j, curr_indexes in enumerate(indexes):
            X_curr, y_curr = X_test[j][:fields_number], X_test[j][-1]
            letter_scores, total_scores = self._score_batch(
                X_curr, y_curr, lengths[curr_indexes], batch_size=batch_size)
            if return_log_probs:
                # letter_scores = -np.log2(letter_scores)
                letter_scores = [-np.log(letter_score) for letter_score in letter_scores]
            if return_exp_total:
                total_scores = np.exp(total_scores) # 2.0 ** total_scores
            for i, letter_score, total_score in zip(curr_indexes, letter_scores, total_scores):
                answer[i] = (letter_score, total_score) if return_letter_scores else total_score
        return answer

    def predict_states_batch(self, X, mode="state"):
        fields_number = 2 if self.labels_ is not None else 1
        X_test, indexes = self.transform(X, buckets_number=1, max_bucket_length=len(X))
        func = self._logit_func_  if mode == "logit" else self._state_func_
        states = func(X_test[0][:fields_number] + [0])[0]
        return states, X_test

    def predict_proba(self, X, batch_size=256):
        fields_number = 2 if self.labels_ is not None else 1
        X_test, indexes = self.transform(X, max_bucket_length=batch_size, return_indexes=True)
        answer = [None] * len(X)
        start_probs = np.zeros(shape=(1, self.symbols_number_), dtype=float)
        start_probs[0, BEGIN] = 1.0
        for j, curr_indexes in enumerate(indexes):
            X_curr = X_test[j][:fields_number]
            curr_probs = self.model_.predict(X_curr, batch_size=batch_size)
            for i, probs in zip(curr_indexes, curr_probs):
                answer[i] = np.vstack((start_probs, probs[:len(X[i][0])+1]))
        return answer


def load_lm(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key == "callbacks" or key.endswith("dump_file"))}
    # коллбэки
    args['callbacks'] = []
    for key, cls in zip(["early_stopping_callback", "reduce_LR_callback"],
                        [EarlyStopping, ReduceLROnPlateau]):
        if key in json_data:
            args['callbacks'].append(cls(**json_data[key]))
    # создаём языковую модель
    lm = NeuralLM(**args)
    # обучаемые параметры
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        if key == "vocabulary_":
            setattr(lm, key, vocabulary_from_json(value))
        else:
            setattr(lm, key, value)
    if hasattr(lm, "tags_"):
        lm.tags_ = [tuple(x) for x in lm.tags_]
        lm.tag_codes_ = {tag: i for i, tag in enumerate(lm.tags_)}
    # модель
    lm.build(verbose=0)  # не работает сохранение модели, приходится сохранять только веса
    lm.model_.load_weights(json_data['dump_file'])
    return lm


