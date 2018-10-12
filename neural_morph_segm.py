import sys
import os
import inspect
import bisect
from itertools import chain
from collections import defaultdict
import numpy as np

import json
# import ujson as json

import keras.layers as kl
import keras.backend as kb
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from read import extract_morpheme_type, read_BMES, read_splitted
from tabled_trie import make_trie


def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        config = json.load(fin)
    if "use_morpheme_types" not in config:
        config["use_morpheme_types"] = True
    return config

# вспомогательные фунцкии

def to_one_hot(data, classes_number):
    answer = np.eye(classes_number, dtype=np.uint8)
    return answer[data]

def make_model_file(name, i):
    pos = name.rfind(".")
    if pos != -1:
        return "{}-{}.{}".format(name[:pos], i, name[pos+1:])
    else:
        return "{}-{}".format(name, i)


AUXILIARY_CODES = PAD, BEGIN, END, UNKNOWN = 0, 1, 2, 3
AUXILIARY = ['PAD', 'BEGIN', 'END', 'UNKNOWN']


def _make_vocabulary(source):
    """
    Создаёт словарь символов.
    """
    symbols = {a for word in source for a in word}
    symbols = AUXILIARY + sorted(symbols)
    symbol_codes = {a: i for i, a in enumerate(symbols)}
    return symbols, symbol_codes

def make_bucket_lengths(lengths, buckets_number):
    """
    Вычисляет максимальные длины элементов в корзинах. Каждая корзина состоит из элементов примерно одинаковой длины
    """
    m = len(lengths)
    lengths = sorted(lengths)
    last_bucket_length, bucket_lengths = 0, []
    for i in range(buckets_number):
        # могут быть проблемы с выбросами большой длины
        level = (m * (i + 1) // buckets_number) - 1
        curr_length = lengths[level]
        if curr_length > last_bucket_length:
            bucket_lengths.append(curr_length)
            last_bucket_length = curr_length
    return bucket_lengths

def collect_buckets(lengths, buckets_number, max_bucket_size=-1):
    """
    Распределяет элементы по корзинам
    """
    bucket_lengths = make_bucket_lengths(lengths, buckets_number)
    indexes = [[] for _ in bucket_lengths]
    for i, length in enumerate(lengths):
        index = bisect.bisect_left(bucket_lengths, length)
        indexes[index].append(i)
    if max_bucket_size != -1:
        bucket_lengths = list(chain.from_iterable(
            ([L] * ((len(curr_indexes)-1) // max_bucket_size + 1))
            for L, curr_indexes in zip(bucket_lengths, indexes)
            if len(curr_indexes) > 0))
        indexes = [curr_indexes[start:start+max_bucket_size]
                   for curr_indexes in indexes
                   for start in range(0, len(curr_indexes), max_bucket_size)]
    return [(L, curr_indexes) for L, curr_indexes
            in zip(bucket_lengths, indexes) if len(curr_indexes) > 0]

def load_cls(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback") or key == "model_files")}
    args['callbacks'] = []
    # создаём классификатор
    inflector = Partitioner(**args)
    # обучаемые параметры
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        setattr(inflector, key, value)
    if hasattr(inflector, "morphemes_"):
        inflector._make_morpheme_tries()
    # модель
    inflector.build()  # не работает сохранение/загрузка модели, приходится перекомпилировать
    for i, (model, model_file) in enumerate(
            zip(inflector.models_, json_data['model_files'])):
        model.load_weights(model_file)
    return inflector


MORPHEME_TYPES = ["PREF", "ROOT", "LINK", "END", "POSTFIX", "HYPH"]
PREF, ROOT, LINK, SUFF, ENDING, POSTFIX, HYPH, FINAL = 0, 1, 2, 3, 4, 5, 6, 7


def get_next_morpheme_types(morpheme_type):
    """
    Определяет, какие морфемы могут идти за текущей.
    """
    if morpheme_type == "None":
        return ["None"]
    MORPHEMES = ["SUFF", "END", "LINK", "POSTFIX", "PREF", "ROOT"]
    if morpheme_type in ["ROOT", "SUFF", "HYPH"]:
        start = 0
    elif morpheme_type == "END":
        start = 2
    elif morpheme_type in ["PREF", "LINK", "BEGIN"]:
        start = 4
    else:
        start = 6
    answer = MORPHEMES[start:6]
    if len(answer) > 0 and morpheme_type != "HYPH":
        answer.append("HYPH")
    if morpheme_type == "BEGIN":
        answer.append("None")
    return answer

def get_next_morpheme(morpheme):
    """
    Строит список меток, которые могут идти за текущей
    """
    if morpheme == "BEGIN":
        morpheme = "S-BEGIN"
    morpheme_label, morpheme_type = morpheme.split("-")
    if morpheme_label in "BM":
        new_morpheme_labels = "ME"
        new_morpheme_types = [morpheme_type]
    else:
        new_morpheme_labels = "BS"
        new_morpheme_types = get_next_morpheme_types(morpheme_type)
    answer = ["{}-{}".format(x, y) for x in new_morpheme_labels for y in new_morpheme_types]
    return answer


def is_correct_morpheme_sequence(morphemes):
    """
    Проверяет список морфемных меток на корректность
    """
    if morphemes == []:
        return False
    if any("-" not in morpheme for morpheme in morphemes):
        return False
    morpheme_label, morpheme_type = morphemes[0].split("-")
    if morpheme_label not in "BS" or morpheme_type not in ["PREF", "ROOT", "None"]:
        return False
    morpheme_label, morpheme_type = morphemes[-1].split("-")
    if morpheme_label not in "ES" or morpheme_type not in ["ROOT", "SUFF", "ENDING", "POSTFIX", "None"]:
        return False
    for i, morpheme in enumerate(morphemes[:-1]):
        if morphemes[i+1] not in get_next_morpheme(morpheme):
            return False
    return True


class Partitioner:

    """
    models_number: int, default=1, число моделей
    to_memorize_morphemes: bool, default=False,
        производится ли запоминание морфемных энграмм
    min_morpheme_count: int, default=2,
        минимальное количество раз, которое должна встречаться запоминаемая морфема
    to_memorize_ngram_counts: bool, default=False,
        используются ли частоты энграмм как морфем при вычислении признаков
    min_relative_ngram_count: float, default=0.1,
        минимальное отношение частоты энграммы как морфемы к её общей частоте,
        необходимое для её запоминания
    use_embeddings: bool, default=False,
        используется ли дополнительный слой векторных представлений символов
    embeddings_size: int, default=32, размер символьного представления
    conv_layers: int, default=1, число свёрточных слоёв
    window_size: int or list of ints, список размеров окна в свёрточном слое
    filters_number: int or list of ints or list of list of ints,
        число фильтров в свёрточных слоях,
        filters_number[i,j] --- число фильтров для i-го окна j-го слоя,
        если задан список, то filters_number[j] --- число фильтров в окнах j-го слоя,
        если число --- то одно и то же число фильтров для всех слоёв и окон
    dense_output_units: int, default=0,
        число нейронов на дополнительном слое перед вычислением выходных вероятностей.
        если 0, то этот слой отсутствует
    use_lstm: bool, default=False,
        используется ли дополнительный выходной слой LSTM (ухудшает качество)
    lstm_units: int, default=64, число нейронов в LSTM-слое
    dropout: float, default=0.0
        доля выкидываемых нейронов в dropout-слое, помогает бороться с переобучением
    context_dropout: float, default=0.0,
        вероятность маскировки векторного представления контекста
    buckets_number: int, default=10,
        число корзин, в одну корзину попадают данные примерно одинаковой длины
    nepochs: int, default=10, число эпох в обучении
    validation_split: float, default=0.2, доля элементов в развивающей выборке
    batch_size: int, default=32, число элементов в одном батче
    callbacks: list of keras.callbacks or None, default=None,
        коллбэки для управления процессом обучения,
    early_stopping: int, default=None,
        число эпох, в течение которого не должно улучшаться качество
        на валидационной выборке, чтобы обучение остановилось,
        если None, то в любом случае модель обучается nepochs эпох
    """

    LEFT_MORPHEME_TYPES = ["pref", "root"]
    RIGHT_MORPHEME_TYPES = ["root", "suff", "end", "postfix"]

    def __init__(self, models_number=1, use_morpheme_types=True,
                 to_memorize_morphemes=False, min_morpheme_count=2,
                 to_memorize_ngram_counts=False, min_relative_ngram_count=0.1,
                 use_embeddings=False, embeddings_size=32,
                 conv_layers=1, window_size=5, filters_number=64,
                 dense_output_units=0, use_lstm=False, lstm_units=64,
                 dropout=0.0, context_dropout=0.0,
                 buckets_number=10, nepochs=10,
                 validation_split=0.2, batch_size=32,
                 callbacks=None, early_stopping=None):
        self.models_number = models_number
        self.use_morpheme_types = use_morpheme_types
        self.to_memorize_morphemes = to_memorize_morphemes
        self.min_morpheme_count = min_morpheme_count
        self.to_memorize_ngram_counts = to_memorize_ngram_counts
        self.min_relative_ngram_count = min_relative_ngram_count
        self.use_embeddings = use_embeddings
        self.embeddings_size = embeddings_size
        self.conv_layers = conv_layers
        self.window_size = window_size
        self.filters_number = filters_number
        self.dense_output_units = dense_output_units
        self.use_lstm = use_lstm
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.context_dropout = context_dropout
        self.buckets_number = buckets_number
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.early_stopping = early_stopping
        self.check_params()

    def check_params(self):
        if isinstance(self.window_size, int):
            # если было только одно окно в свёрточных слоях
            self.window_size = [self.window_size]
        # приводим фильтры к двумерному виду
        self.filters_number = np.atleast_2d(self.filters_number)
        if self.filters_number.shape[0] == 1:
            self.filters_number = np.repeat(self.filters_number, len(self.window_size), axis=0)
        if self.filters_number.shape[0] != len(self.window_size):
            raise ValueError("Filters array should have shape (len(window_size), conv_layers)")
        if self.filters_number.shape[1] == 1:
            self.filters_number = np.repeat(self.filters_number, self.conv_layers, axis=1)
        if self.filters_number.shape[1] != self.conv_layers:
            raise ValueError("Filters array should have shape (len(window_size), conv_layers)")
        # переводим в список из int, а не np.int32, чтобы не было проблем при сохранении
        self.filters_number = list([list(map(int, x)) for x in self.filters_number])
        if self.callbacks is None:
            self.callbacks = []
        if (self.early_stopping is not None and
                not any(isinstance(x, EarlyStopping) for x in self.callbacks)):
            self.callbacks.append(EarlyStopping(patience=self.early_stopping, monitor="val_acc"))
        if self.use_morpheme_types:
            self._morpheme_memo_func = self._make_morpheme_data
        else:
            self._morpheme_memo_func = self._make_morpheme_data_simple

    def to_json(self, outfile, model_file=None):
        info = dict()
        if model_file is None:
            pos = outfile.rfind(".")
            model_file = outfile[:pos] + ("-model.hdf5" if pos != -1 else "-model")
        model_files = [make_model_file(model_file, i+1) for i in range(self.models_number)]
        for i in range(self.models_number):
            # при сохранении нужен абсолютный путь, а не от текущей директории
            model_files[i] = os.path.abspath(model_files[i])
        for (attr, val) in inspect.getmembers(self):
            # перебираем поля класса и сохраняем только задаваемые при инициализации
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(Partitioner, attr, None), property) or
                    attr.isupper() or attr in [
                        "callbacks", "models_", "left_morphemes_", "right_morphemes_", "morpheme_trie_"]):
                info[attr] = val
            elif attr == "models_":
                # для каждой модели сохраняем веса
                info["model_files"] = model_files
                for model, curr_model_file in zip(self.models_, model_files):
                    model.save_weights(curr_model_file)
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    # property --- функция, прикидывающаяся переменной; декоратор метода (превращает метод класса в атрибут класса)
    @property 
    def symbols_number_(self):
        return len(self.symbols_)

    @property
    def target_symbols_number_(self):
        return len(self.target_symbols_)

    @property
    def memory_dim(self):
        return 15 if self.use_morpheme_types else 3

    def _preprocess(self, data, targets=None):
        # к каждому слову добавляются символы начала и конца строки
        lengths = [len(x) + 2 for x in data]
        # разбиваем данные на корзины
        buckets_with_indexes = collect_buckets(lengths, self.buckets_number)
        # преобразуем данные в матрицы в каждой корзине
        data_by_buckets = [self._make_bucket_data(data, length, indexes)
                           for length, indexes in buckets_with_indexes]
        # targets=None --- предсказание, иначе --- обучение
        if targets is not None:
            targets_by_buckets = [self._make_bucket_data(targets, length, indexes, is_target=True)
                                  for length, indexes in buckets_with_indexes]
            return data_by_buckets, targets_by_buckets, buckets_with_indexes
        else:
            return data_by_buckets, buckets_with_indexes

    def _make_bucket_data(self, data, bucket_length, bucket_indexes, is_target=False):
        """
        data: list of lists, исходные данные
        bucket_length: int, максимальная длина элемента в корзине
        bucket_indexes: list of ints, индексы элементов в корзине
        is_target: boolean, default=False,
            являются ли данные исходными или ответами

        answer = [symbols, (classes)],
            symbols: array of shape (len(data), bucket_length)
                элементы data, дополненные символом PAD справа до bucket_length
            classes: array of shape (len(data), classes_number)
        """
        bucket_data = [data[i] for i in bucket_indexes]
        if is_target:
            return self._recode_bucket_data(bucket_data, bucket_length, self.target_symbol_codes_)
        else:
            answer = [self._recode_bucket_data(bucket_data, bucket_length, self.symbol_codes_)]
            if self.to_memorize_morphemes:
                print("Processing morphemes for bucket length", bucket_length)
                answer.append(self._morpheme_memo_func(bucket_data, bucket_length))
                print("Processing morphemes for bucket length", bucket_length, "finished")
            return answer

    def _recode_bucket_data(self, data, bucket_length, encoding):
        answer = np.full(shape=(len(data), bucket_length), fill_value=PAD, dtype=int)
        answer[:,0] = BEGIN
        for j, word in enumerate(data):
            answer[j,1:1+len(word)] = [encoding.get(x, UNKNOWN) for x in word]
            answer[j,1+len(word)] = END
        return answer

    def _make_morpheme_data(self, data, bucket_length):
        """
        строит для каждой позиции во входных словах вектор, кодирующий энграммы в контексте

        data: list of strs, список исходных слов
        bucket_length: int, максимальная длина слова в корзине

        answer: np.array[float] of shape (len(data), bucket_length, 15)
        """
        answer = np.zeros(shape=(len(data), bucket_length, 15), dtype=float)
        for j, word in enumerate(data):
            m = len(word)
            curr_answer = np.zeros(shape=(bucket_length, 15), dtype=int)
            root_starts = [0]
            ending_ends = [m]
            prefixes = self.left_morphemes_["pref"].descend_by_prefixes(word[:-1])
            for end in prefixes:
                score = self._get_ngram_score(word[:end], "pref")
                if end == 1:
                    curr_answer[1,10] = max(score, curr_answer[1,10])
                else:
                    curr_answer[1,0] = max(score, curr_answer[1,0])
                    curr_answer[end, 5] = max(score, curr_answer[end, 5])
            root_starts += prefixes
            postfix_lengths = self.right_morphemes_["postfix"].descend_by_prefixes(word[:0:-1])
            for k in postfix_lengths:
                score = self._get_ngram_score(word[-k:], "postfix")
                if k == 1:
                    curr_answer[m, 14] = max(score, curr_answer[m, 14])
                else:
                    curr_answer[m, 9] = max(score, curr_answer[m, 9])
                    curr_answer[m-k+1,4] = max(score, curr_answer[m-k+1,4])
                ending_ends.append(m-k)
            suffix_ends = set(ending_ends)
            for end in ending_ends[::-1]:
                ending_lengths = self.right_morphemes_["end"].descend_by_prefixes(word[end-1:0:-1])
                for k in ending_lengths:
                    score = self._get_ngram_score(word[end-k:end], "end")
                    if k == 1:
                        curr_answer[end, 13] = max(score, curr_answer[end, 13])
                    else:
                        curr_answer[end-k+1, 3] = max(score, curr_answer[end-k+1, 3])
                        curr_answer[end, 8] = max(score, curr_answer[end, 8])
                    suffix_ends.add(end-k)
            suffixes = self.right_morphemes_["suff"].descend_by_prefixes(
                word[::-1], start_pos=[m-k for k in suffix_ends], max_count=3, return_pairs=True)
            suffix_starts = suffix_ends
            for first, last in suffixes:
                score = self._get_ngram_score(word[m-last:m-first], "suff")
                if last == first + 1:
                    curr_answer[m-first, 12] = max(score, curr_answer[m-first, 12])
                else:
                    curr_answer[m-last+1, 2] = max(score, curr_answer[m-last+1, 2])
                    curr_answer[m-first, 7] = max(score, curr_answer[m-first, 7])
                suffix_starts.add(m-last)
            for start in root_starts:
                root_ends = self.left_morphemes_["root"].descend_by_prefixes(word[start:])
                for end in root_ends:
                    score = self._get_ngram_score(word[start:end], "root")
                    if end == start+1:
                        curr_answer[start + 1, 11] = max(score, curr_answer[start + 1, 11])
                    else:
                        curr_answer[start + 1, 1] = max(score, curr_answer[start + 1, 1])
                        curr_answer[end, 6] = max(score, curr_answer[end, 6])
            for end in suffix_starts:
                root_lengths = self.right_morphemes_["root"].descend_by_prefixes(word[end-1:-1:-1])
                for k in root_lengths:
                    score = self._get_ngram_score(word[end-k:end], 'root')
                    if k == 1:
                        curr_answer[end, 11] = max(curr_answer[end, 11], score)
                    else:
                        curr_answer[end-k+1, 1] = max(curr_answer[end-k+1, 1], score)
                        curr_answer[end, 6] = max(curr_answer[end, 6], score)
            answer[j] = curr_answer
        return answer

    def _make_morpheme_data_simple(self, data, bucket_length):
        answer = np.zeros(shape=(len(data), bucket_length, 3), dtype=float)
        for j, word in enumerate(data):
            m = len(word)
            curr_answer = np.zeros(shape=(bucket_length, 3), dtype=int)
            positions = self.morpheme_trie_.find_substrings(word, return_positions=True)
            for starts, end in positions:
                for start in starts:
                    score = self._get_ngram_score(word[start:end])
                    if end == start+1:
                        curr_answer[start+1, 2] = max(curr_answer[start+1, 2], score)
                    else:
                        curr_answer[start+1, 0] = max(curr_answer[start+0, 2], score)
                        curr_answer[end, 1] = max(curr_answer[end, 1], score)
            answer[j] = curr_answer
        return answer

    def _get_ngram_score(self, ngram, mode="None"):
        if self.to_memorize_ngram_counts:
            return self.morpheme_counts_[mode].get(ngram, 0)
        else:
            return 1.0

    def train(self, source, targets, dev=None, dev_targets=None, model_file=None):
        """

        source: list of strs, список слов для морфемоделения
        targets: list of strs, метки морфемоделения в формате BMES
        model_file: str or None, default=None, файл для сохранения моделей

        Возвращает:
        -------------
        self, обученный морфемоделитель
        """
        self.symbols_, self.symbol_codes_ = _make_vocabulary(source)
        self.target_symbols_, self.target_symbol_codes_ = _make_vocabulary(targets)
        if self.to_memorize_morphemes:
            self._memorize_morphemes(source, targets)

        data_by_buckets, targets_by_buckets, _ = self._preprocess(source, targets)
        if dev is not None:
            dev_data_by_buckets, dev_targets_by_buckets, _ = self._preprocess(dev, dev_targets)
        else:
            dev_data_by_buckets, dev_targets_by_buckets = None, None
        self.build()
        self._train_models(data_by_buckets, targets_by_buckets,  dev_data_by_buckets,
                           dev_targets_by_buckets, model_file=model_file)
        return self

    def build(self):
        """
        Создаёт нейронные модели
        """
        self.models_ = [self.build_model() for _ in range(self.models_number)]
        print(self.models_[0].summary())
        return self

    def build_model(self):
        """
        Функция, задающая архитектуру нейронной сети
        """
        # symbol_inputs: array, 1D-массив длины m
        symbol_inputs = kl.Input(shape=(None,), dtype='uint8', name="symbol_inputs")
        # symbol_embeddings: array, 2D-массив размера m*self.symbols_number
        if self.use_embeddings:
            symbol_embeddings = kl.Embedding(self.symbols_number_, self.embeddings_size,
                                             name="symbol_embeddings")(symbol_inputs)
        else:
            symbol_embeddings = kl.Lambda(kb.one_hot, output_shape=(None, self.symbols_number_),
                                          arguments={"num_classes": self.symbols_number_},
                                          name="symbol_embeddings")(symbol_inputs)
        inputs = [symbol_inputs]
        if self.to_memorize_morphemes:
            # context_inputs: array, 2D-массив размера m*15
            context_inputs = kl.Input(shape=(None, self.memory_dim), dtype='float32', name="context_inputs")
            inputs.append(context_inputs)
            if self.context_dropout > 0.0:
                context_inputs = kl.Dropout(self.context_dropout)(context_inputs)
            # представление контекста подклеивается к представлению символа
            symbol_embeddings = kl.Concatenate()([symbol_embeddings, context_inputs])
        conv_inputs = symbol_embeddings
        conv_outputs = []
        for window_size, curr_filters_numbers in zip(self.window_size, self.filters_number):
            # свёрточный слой отдельно для каждой ширины окна
            curr_conv_input = conv_inputs
            for j, filters_number in enumerate(curr_filters_numbers[:-1]):
                # все слои свёртки, кроме финального (после них возможен dropout)
                curr_conv_input = kl.Conv1D(filters_number, window_size,
                                            activation="relu", padding="same")(curr_conv_input)
                if self.dropout > 0.0:
                    # между однотипными слоями рекомендуется вставить dropout
                    curr_conv_input = kl.Dropout(self.dropout)(curr_conv_input)
            if not self.use_lstm:
                curr_conv_output = kl.Conv1D(curr_filters_numbers[-1], window_size,
                                             activation="relu", padding="same")(curr_conv_input)
            else:
                curr_conv_output = curr_conv_input
            conv_outputs.append(curr_conv_output)
        # соединяем выходы всех свёрточных слоёв в один вектор
        if len(conv_outputs) == 1:
            conv_output = conv_outputs[0]
        else:
            conv_output = kl.Concatenate(name="conv_output")(conv_outputs)
        if self.use_lstm:
            conv_output = kl.Bidirectional(
                kl.LSTM(self.lstm_units, return_sequences=True))(conv_output)
        if self.dense_output_units:
            pre_last_output = kl.TimeDistributed(
                kl.Dense(self.dense_output_units, activation="relu"),
                name="pre_output")(conv_output)
        else:
            pre_last_output = conv_output
        # финальный слой с softmax-активацией, чтобы получить распределение вероятностей
        output = kl.TimeDistributed(
            kl.Dense(self.target_symbols_number_, activation="softmax"), name="output")(pre_last_output)
        model = Model(inputs, [output])
        model.compile(optimizer=adam(clipnorm=5.0),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def _train_models(self, data_by_buckets, targets_by_buckets,
                      dev_data_by_buckets=None, dev_targets_by_buckets=None, model_file=None):
        """
        data_by_buckets: list of lists of np.arrays,
            data_by_buckets[i] = [..., bucket_i, ...],
            bucket = [input_1, ..., input_k],
            input_j --- j-ый вход нейронной сети, вычисленный для текущей корзины
        targets_by_buckets: list of np.arrays,
            targets_by_buckets[i] --- закодированные ответы для i-ой корзины
        model_file: str or None, путь к файлу для сохранения модели
        """
        train_indexes_by_buckets, dev_indexes_by_buckets = [], []
        if dev_data_by_buckets is not None:
            train_indexes_by_buckets = [list(range(len(bucket[0]))) for bucket in data_by_buckets]
            for elem in train_indexes_by_buckets:
                np.random.shuffle(elem)
            dev_indexes_by_buckets = [list(range(len(bucket[0]))) for bucket in dev_data_by_buckets]
            train_data, dev_data = data_by_buckets, dev_data_by_buckets
            train_targets, dev_targets = targets_by_buckets, dev_targets_by_buckets
        else:
            for bucket in data_by_buckets:
                # разбиваем каждую корзину на обучающую и валидационную выборку
                L = len(bucket[0])
                indexes_for_bucket = list(range(L))
                np.random.shuffle(indexes_for_bucket)
                train_bucket_length = int(L*(1.0 - self.validation_split))
                train_indexes_by_buckets.append(indexes_for_bucket[:train_bucket_length])
                dev_indexes_by_buckets.append(indexes_for_bucket[train_bucket_length:])
            train_data, dev_data = data_by_buckets, data_by_buckets
            train_targets, dev_targets = targets_by_buckets, targets_by_buckets
        # разбиваем на батчи обучающую и валидационную выборку
        # (для валидационной этого можно не делать, а подавать сразу корзины)
        train_batches_indexes = list(chain.from_iterable(
            [[(i, elem[j:j+self.batch_size]) for j in range(0, len(elem), self.batch_size)]
             for i, elem in enumerate(train_indexes_by_buckets)]))
        dev_batches_indexes = list(chain.from_iterable(
            [[(i, elem[j:j+self.batch_size]) for j in range(0, len(elem), self.batch_size)]
             for i, elem in enumerate(dev_indexes_by_buckets)]))
        # поскольку функции fit_generator нужен генератор, порождающий batch за batch'ем,
        # то приходится заводить генераторы для обеих выборок
        train_gen = generate_data(train_data, train_targets, train_batches_indexes,
                                  classes_number=self.target_symbols_number_, shuffle=True)
        val_gen = generate_data(dev_data, dev_targets, dev_batches_indexes,
                                classes_number=self.target_symbols_number_, shuffle=False)
        for i, model in enumerate(self.models_):
            if model_file is not None:
                curr_model_file = make_model_file(model_file, i+1)
                # для сохранения модели с наилучшим результатом на валидационной выборке
                save_callback = ModelCheckpoint(curr_model_file, save_weights_only=True, save_best_only=True)
                curr_callbacks = self.callbacks + [save_callback]
            else:
                curr_callbacks = self.callbacks
            model.fit_generator(train_gen, len(train_batches_indexes),
                                epochs=self.nepochs, callbacks=curr_callbacks,
                                validation_data=val_gen, validation_steps=len(dev_batches_indexes))
            if model_file is not None:
                model.load_weights(curr_model_file)
        return self

    def _memorize_morphemes(self, words, targets):
        """
        запоминает морфемы. встречающиеся в словах обучающей выборки
        """
        morphemes = defaultdict(lambda: defaultdict(int))
        for word, target in zip(words, targets):
            start = None
            for i, (symbol, label) in enumerate(zip(word, target)):
                if label.startswith("B-"):
                    start = i
                elif label.startswith("E-"):
                    dest = extract_morpheme_type(label)
                    morphemes[dest][word[start:i+1]] += 1
                elif label.startswith("S-"):
                    dest = extract_morpheme_type(label)
                    morphemes[dest][word[i]] += 1
                elif label == END:
                    break
        self.morphemes_ = dict()
        for key, counts in morphemes.items():
            self.morphemes_[key] = [x for x, count in counts.items() if count >= self.min_morpheme_count]
        self._make_morpheme_tries()
        if self.to_memorize_ngram_counts:
            self._memorize_ngram_counts(words, morphemes)
        return self

    def _memorize_ngram_counts(self, words, counts):
        """
        запоминает частоты морфем, встречающихся в словах обучающей выборки
        """
        prefix_counts, suffix_counts, ngram_counts  = defaultdict(int), defaultdict(int), defaultdict(int)
        for i, word in enumerate(words, 1):
            if i % 5000 == 0:
                print("{} words processed".format(i))
            positions = self.morpheme_trie_.find_substrings(word, return_positions=True)
            for starts, end in positions:
                for start in starts:
                    segment = word[start:end]
                    ngram_counts[segment] += 1
                    if start == 0:
                        prefix_counts[segment] += 1
                    if end == len(word):
                        suffix_counts[segment] += 1
        self.morpheme_counts_ = dict()
        for key, curr_counts in counts.items():
            curr_relative_counts = dict()
            curr_ngram_counts = (prefix_counts if key == "pref" else
                                 suffix_counts if key in ["end", "postfix"] else ngram_counts)
            for ngram, count in curr_counts.items():
                if count < self.min_morpheme_count or ngram not in curr_ngram_counts:
                    continue
                relative_count = min(count / curr_ngram_counts[ngram], 1.0)
                if relative_count >= self.min_relative_ngram_count:
                    curr_relative_counts[ngram] = relative_count
            self.morpheme_counts_[key] = curr_relative_counts
        return self

    def _make_morpheme_tries(self):
        """
        строит префиксный бор для морфем для более быстрого их поиска
        """
        self.left_morphemes_, self.right_morphemes_ = dict(), dict()
        if self.use_morpheme_types:
            for key in self.LEFT_MORPHEME_TYPES:
                self.left_morphemes_[key] = make_trie(list(self.morphemes_[key]))
            for key in self.RIGHT_MORPHEME_TYPES:
                self.right_morphemes_[key] = make_trie([x[::-1] for x in self.morphemes_[key]])
        if not self.use_morpheme_types or self.to_memorize_ngram_counts:
            morphemes = {x for elem in self.morphemes_.values() for x in elem}
            self.morpheme_trie_ = make_trie(list(morphemes))
        return self

    def _predict_probs(self, words):
        """
        data = [word_1, ..., word_m]

        Возвращает:
        -------------
        answer = [probs_1, ..., probs_m]
        probs_i = [p_1, ..., p_k], k = len(word_i)
        p_j = [p_j1, ..., p_jr], r --- число классов
        (len(AUXILIARY) + 4 * 4 (BMES; PREF, ROOT, SUFF, END) + 3 (BME; POSTFIX) + 2 * 1 (S; LINK, HYPHEN) = 23)
        """
        data_by_buckets, indexes_by_buckets = self._preprocess(words)
        word_probs = [None] * len(words)
        for r, (bucket_data, (_, bucket_indexes)) in\
                enumerate(zip(data_by_buckets, indexes_by_buckets), 1):
            print("Bucket {} predicting".format(r))
            bucket_probs = np.mean([model.predict(bucket_data) for model in self.models_], axis=0)
            for i, elem in zip(bucket_indexes, bucket_probs):
                word_probs[i] = elem
        answer = [None] * len(words)
        for i, (elem, word) in enumerate(zip(word_probs, words)):
            if i % 1000 == 0 and i > 0:
                print("{} words decoded".format(i))
            answer[i] = self._decode_best(elem, len(word))
        return answer

    def labels_to_morphemes(self, word, labels, probs=None, return_probs=False, return_types=False):
        """

        Преобразует ответ из формата BMES в список морфем
        Дополнительно может возвращать список вероятностей морфем

        word: str, текущее слово,
        labels: list of strs, предсказанные метки в формате BMES,
        probs: list of floats or None, предсказанные вероятности меток

        answer = [morphemes, (morpheme_probs), (morpheme_types)]
            morphemes: list of strs, список морфем
            morpheme_probs: list of floats, список вероятностей морфем
            morpheme_types: list of strs, список типов морфем
        """
        morphemes, curr_morpheme, morpheme_types = [], "", []
        if self.use_morpheme_types:
            end_labels = ['E-ROOT', 'E-PREF', 'E-SUFF', 'E-END', 'E-POSTFIX', 'S-ROOT',
                          'S-PREF', 'S-SUFF', 'S-END', 'S-LINK', 'S-HYPH']
        else:
            end_labels = ['E-None', 'S-None']
        for letter, label in zip(word, labels):
            curr_morpheme += letter
            if label in end_labels:
                morphemes.append(curr_morpheme)
                curr_morpheme = ""
                morpheme_types.append(label.split("-")[-1])
        if return_probs:
            if probs is None:
                Warning("Для вычисления вероятностей морфем нужно передать вероятности меток")
                return_probs = False
        if return_probs:
            morpheme_probs, curr_morpheme_prob = [], 1.0
            for label, prob in zip(labels, probs):
                curr_morpheme_prob *= prob[self.target_symbol_codes_[label]]
                if label in end_labels:
                    morpheme_probs.append(curr_morpheme_prob)
                    curr_morpheme_prob = 1.0
            answer = [morphemes, morpheme_probs]
        else:
            answer = [morphemes]
        if return_types:
            answer.append(morpheme_types)
        return answer

    def predict(self, words, return_probs=False):
        labels_with_probs = self._predict_probs(words)
        return [self.labels_to_morphemes(word, elem[0], elem[1], return_probs=return_probs)
                for elem, word in zip(labels_with_probs, words)]

    def _decode_best(self, probs, length):
        """
        Поддерживаем в каждой позиции наилучшие гипотезы для каждого состояния
        Состояние --- последняя предсказанняя метка
        """
        # вначале нужно проверить заведомо наилучший вариант на корректность
        best_states = np.argmax(probs[:1+length], axis=1)
        best_labels = [self.target_symbols_[state_index] for state_index in best_states]
        if not is_correct_morpheme_sequence(best_labels[1:]):
            # наилучший вариант оказался некорректным
            initial_costs = [np.inf] * self.target_symbols_number_
            initial_states = [None] * self.target_symbols_number_
            initial_costs[BEGIN], initial_states[BEGIN] = -np.log(probs[0, BEGIN]), BEGIN
            costs, states = [initial_costs], [initial_states]
            for i in range(length):
                # состояний мало, поэтому можно сортировать на каждом шаге
                state_order = np.argsort(costs[-1])
                curr_costs = [np.inf] * self.target_symbols_number_
                prev_states = [None] * self.target_symbols_number_
                inf_count = self.target_symbols_number_
                for prev_state in state_order:
                    if np.isinf(costs[-1][prev_state]):
                        break
                    elif prev_state in AUXILIARY_CODES and i != 0:
                        continue
                    possible_states = self.get_possible_next_states(prev_state)
                    for state in possible_states:
                        if np.isinf(curr_costs[state]):
                            # поскольку новая вероятность не зависит от state,
                            # а старые перебираются по возрастанию штрафа,
                            # то оптимальное значение будет при первом обновлении
                            curr_costs[state] = costs[-1][prev_state] - np.log(probs[i+1,state])
                            prev_states[state] = prev_state
                            inf_count -= 1
                    if inf_count == len(AUXILIARY_CODES):
                        # все вероятности уже посчитаны
                        break
                costs.append(curr_costs)
                states.append(prev_states)
            # последнее состояние --- обязательно конец морфемы
            possible_states = [self.target_symbol_codes_["{}-{}".format(x, y)]
                               for x in "ES" for y in ["ROOT", "SUFF", "END", "POSTFIX", "None"]
                               if "{}-{}".format(x, y) in self.target_symbol_codes_]
            best_states = [min(possible_states, key=(lambda x: costs[-1][x]))]
            for j in range(length, 0, -1):
                # предыдущее состояние
                best_states.append(states[j][best_states[-1]])
            best_states = best_states[::-1]
        probs_to_return = np.zeros(shape=(length, self.target_symbols_number_), dtype=np.float32)
        # убираем невозможные состояния
        for j, state in enumerate(best_states[:-1]):
            possible_states = self.get_possible_next_states(state)
            # оставляем только возможные состояния.
            probs_to_return[j,possible_states] = probs[j+1,possible_states]
        return [self.target_symbols_[i] for i in best_states[1:]], probs_to_return

    def get_possible_next_states(self, state_index):
        state = self.target_symbols_[state_index]
        next_states = get_next_morpheme(state)
        return [self.target_symbol_codes_[x] for x in next_states if x in self.target_symbol_codes_]


def generate_data(data, targets, indexes, classes_number, shuffle=False, nepochs=None):
    """

    data: list of lists of arrays,
        data = [bucket_1, ..., bucket_m],
        bucket = [input_1, ..., input_k], k --- число входов в графе вычислений
    targets: list of arrays,
        targets[i,j] --- код j-ой метки при морфемоделении i-го слова
    indexes: list of pairs,
        indexes = [(i_1, bucket_indexes_1), ...]
        i_j --- номер корзины, откуда берутся элементы j-го батча
        bucket_indexes_j --- номера элементов j-го батча в соответствующей корзине
    shuffle: boolean, default=False, нужно ли перемешивать порядок батчей
    nepochs: int or None, default=None,
        число эпох, в течение которых генератор выдаёт батчи, в случае None генератор бесконечен
    :return:
    """
    nsteps = 0
    while nepochs is None or nsteps < nepochs:
        if shuffle:
            np.random.shuffle(indexes)
        for i, bucket_indexes in indexes:
            curr_bucket, curr_targets = data[i], targets[i]
            data_to_yield = [elem[bucket_indexes] for elem in curr_bucket]
            targets_to_yield = to_one_hot(curr_targets[bucket_indexes], classes_number)
            yield data_to_yield, targets_to_yield
        nsteps += 1


def measure_quality(targets, predicted_targets, english_metrics=False, measure_last=True):
    """

    targets: метки корректных ответов
    predicted_targets: метки предсказанных ответов

    Возвращает словарь со значениями основных метрик
    """
    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    SE = ['{}-{}'.format(x, y) for x in "SE" for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "None"]]
    # SE = ['S-ROOT', 'S-PREF', 'S-SUFF', 'S-END', 'S-LINK', 'E-ROOT', 'E-PREF', 'E-SUFF', 'E-END']
    corr_words = 0
    for corr, pred in zip(targets, predicted_targets):
        corr_len = len(corr) + int(measure_last) - 1
        pred_len = len(pred) + int(measure_last) - 1
        boundaries = [i for i in range(corr_len) if corr[i] in SE]
        pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
        common = [x for x in boundaries if x in pred_boundaries]
        TP += len(common)
        FN += len(boundaries) - len(common)
        FP += len(pred_boundaries) - len(common)
        equal += sum(int(x==y) for x, y in zip(corr, pred))
        total += len(corr)
        corr_words += (corr == pred)
    metrics = ["Точность", "Полнота", "F1-мера", "Корректность", "Точность по словам"]
    if english_metrics:
        metrics = ["Precision", "Recall", "F1", "Accuracy", "Word accuracy"]
    results = [TP / (TP+FP), TP / (TP+FN), TP / (TP + 0.5*(FP+FN)),
               equal / total, corr_words / len(targets)]
    answer = list(zip(metrics, results))
    return answer


SHORT_ARGS = "a:"

if __name__ == "__main__":
    np.random.seed(261) # для воспроизводимости
    if len(sys.argv) < 2:
        sys.exit("Pass config file")
    config_file = sys.argv[1]
    params = read_config(config_file)
    use_morpheme_types = params["use_morpheme_types"]
    read_func = read_BMES if use_morpheme_types else read_splitted
    if "train_file" in params:
        n = params.get("n_train") # число слов в обучающей+развивающей выборке
        inputs, targets = read_func(params["train_file"], n=n)
        if "dev_file" in params:
            n = params.get("n_dev")  # число слов в обучающей+развивающей выборке
            dev_inputs, dev_targets = read_func(params["dev_file"], n=n)
        else:
            dev_inputs, dev_targets = None, None
        # inputs, targets = read_input(params["train_file"], n=n)
    else:
        inputs, targets, dev_inputs, dev_targets = None, None, None, None
    if not "load_file" in params:
        partitioner_params = params.get("model_params", dict())
        partitioner_params["use_morpheme_types"] = use_morpheme_types
        cls = Partitioner(**partitioner_params)
    else:
        cls = load_cls(params["load_file"])
    if inputs is not None:
        cls.train(inputs, targets, dev_inputs, dev_targets, model_file=params.get("model_file"))
    if "save_file" in params:
        model_file = params.get("model_file")
        cls.to_json(params["save_file"], model_file)
    if "test_file" in params:
        inputs, targets = read_func(params["test_file"], shuffle=False)
        # inputs, targets = read_input(params["test_file"])
        predicted_targets = cls._predict_probs(inputs)
        measure_last = params.get("measure_last", use_morpheme_types)
        quality = measure_quality(targets, [elem[0] for elem in predicted_targets],
                                  english_metrics=params.get("english_metrics", False),
                                  measure_last=measure_last)
        for key, value in sorted(quality):
            print("{}={:.2f}".format(key, 100*value))
        if "outfile" in params:
            outfile = params["outfile"]
            output_probs = params.get("output_probs", True)
            format_string = "{}\t{}\t{}\n" if output_probs else "{}\t{}\n"
            output_morpheme_types = params.get("output_morpheme_types", True)
            morph_format_string = "{}\t{}" if output_morpheme_types else "{}"
            with open(outfile, "w", encoding="utf8") as fout:
                for word, (labels, probs) in zip(inputs, predicted_targets):
                    morphemes, morpheme_probs, morpheme_types = cls.labels_to_morphemes(
                        word, labels, probs, return_probs=True, return_types=True)
                    fout.write(format_string.format(
                        word, "/".join(morph_format_string.format(*elem)
                                       for elem in zip(morphemes, morpheme_types)),
                        " ".join("{:.2f}".format(100*x) for x in morpheme_probs)))
                    # fout.write(format_string.format(
                    #     word, "#".join(morphemes), "-".join(
                    #         "{:.2f}/{}".format(100*x, y) for x, y in zip(morpheme_probs, morpheme_types))))
