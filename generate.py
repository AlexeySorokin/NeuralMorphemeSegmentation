import numpy as np

from keras import Model
from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, CallbackList, BaseLogger
from keras.engine.training_generator import evaluate_generator

# вспомогательные фунцкии

def to_one_hot(data, classes_number):
    answer = np.eye(classes_number, dtype=np.uint8)
    return answer[data]


class DataGenerator:

    def __init__(self, data,  targets, indexes, classes_number, shuffle=False, nepochs=None):
        self.data = data
        self.targets = targets
        self.indexes = indexes
        self.classes_number = classes_number
        self.shuffle = shuffle
        self.nepochs = nepochs
        self._initialize()

    def _initialize(self):
        self.step = 0
        self.epoch = 0

    @property
    def steps_per_epoch(self):
        return len(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch == self.nepochs:
            raise StopIteration()
        if self.shuffle and self.step == 0:
            np.random.shuffle(self.indexes)
        i, bucket_indexes = self.indexes[self.step]
        curr_bucket, curr_targets = self.data[i], self.targets[i]
        data_to_yield = [elem[bucket_indexes] for elem in curr_bucket]
        targets_to_yield = to_one_hot(curr_targets[bucket_indexes], self.classes_number)
        self.step += 1
        if self.step == self.steps_per_epoch:
            self.step = 0
            self.epoch += 1
        return data_to_yield, targets_to_yield


class SimpleDataGenerator:

    def __init__(self, data, targets=None, data_vocabulary_size=None,
                 target_vocabulary_size=None, batch_size=16, epochs=None,
                 yield_indexes=False, shuffle=True, seed=179):
        self.data = data
        self.targets = targets
        self.data_vocabulary_size = data_vocabulary_size or [None] * len(self.data)
        if targets is not None:
            self.target_vocabulary_size = target_vocabulary_size or [None] * len(self.targets)
        else:
            self.target_vocabulary_size = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.yield_indexes = yield_indexes
        self.shuffle = shuffle
        self.seed = seed
        self._initialize()

    def __len__(self):
        return len(self.data[0])

    @property
    def steps_per_epoch(self):
        return len(self.indexes)

    def _initialize(self):
        self.indexes = []
        lengths = [len(elem) for elem in self.data[0]]
        ordered_indexes = np.argsort(lengths)
        for i, start in enumerate(range(0, len(self.data[0]), self.batch_size)):
            curr_indexes = ordered_indexes[start:start + self.batch_size]
            self.indexes.append(curr_indexes)
        self.step = 0
        self.epoch = 0
        if self.shuffle:
            np.random.seed(self.seed)
        return self

    def __iter__(self):
        return self

    def _make_batch(self, data, indexes, vocabulary_shapes):
        answer = []
        inputs_number = len(data)
        for i, curr_data in enumerate(data):
            shape = [len(indexes)] + list(np.shape(curr_data[0]))
            if len(shape) > 1:
                max_length = max(len(curr_data[index]) for index in indexes)
                shape[1] = max_length
            curr_answer = np.zeros(shape=shape, dtype=int)
            for j, index in enumerate(indexes):
                if len(shape) > 1:
                    curr_answer[j, :len(curr_data[index])] = curr_data[index]
                else:
                    curr_answer[j] = curr_data[index]
            vocabulary_size = vocabulary_shapes[i]
            if vocabulary_size is not None:
                curr_answer = to_one_hot(curr_answer, vocabulary_size)
            answer.append(curr_answer)
        return answer

    def __next__(self):
        if self.step == 0:
            if self.epochs is not None and self.epoch >= self.epochs:
                raise StopIteration()
            if self.shuffle:
                np.random.shuffle(self.indexes)
        indexes = self.indexes[self.step]
        batch = self._make_batch(self.data, indexes, self.data_vocabulary_size)
        if self.targets is not None:
            batch_targets = self._make_batch(self.targets, indexes, self.target_vocabulary_size)
        self.step += 1
        if self.step == len(self.indexes):
            self.step, self.epoch = 0, self.epoch + 1
        to_return = [batch]
        if self.targets is not None:
            to_return.append(batch_targets)
        if self.yield_indexes:
            to_return.append(indexes)
        return tuple(to_return)

class MultitaskDataGenerator:

    def __init__(self, data, targets, lm_data=None, lm_shifts=None,
                 data_vocabulary_size=None, target_vocabulary_size=None,
                 batch_size=16, epochs=None, start_epochs=0,
                 shuffle=True, seed=179):
        self.data = data
        self.targets = targets
        self.lm_data = lm_data or []
        self.lm_shifts = lm_shifts or []
        self.batch_size = batch_size
        self.epochs = epochs
        self.start_epochs = start_epochs
        self.shuffle = shuffle
        self.seed = seed
        self._initialize_vocabulary_shape(data_vocabulary_size, "data")
        self._initialize_vocabulary_shape(target_vocabulary_size, "targets")
        self._initialize()

    def _initialize_vocabulary_shape(self, shapes, key):
        data = self.targets if key == "targets" else self.data + self.lm_data
        if shapes is None:
            answer = [None] * len(data)
        elif isinstance(shapes, dict):
            answer = [None] * len(data)
            for i, value in shapes.items():
                answer[i] = value
        else:
            answer = shapes
        for i, elem in enumerate(answer):
            if not isinstance(elem, (list, tuple, dict)):
                answer[i] = [elem] * len(data[i])
            elif isinstance(elem, dict):
                answer[i] = [None] * len(data[i])
                for j, value in elem.items():
                    answer[i][j] = value
        if key == "targets":
            self.target_vocabulary_shapes = answer
        else:
            self.data_vocabulary_shapes = answer
        return

    def _initialize(self):
        if isinstance(self.epochs, int) or self.epochs is None:
            self.epochs = [self.epochs] * (len(self.data) + len(self.lm_data))
        if isinstance(self.start_epochs, int) or self.start_epochs is None:
            self.start_epochs = [self.start_epochs] * (len(self.data) + len(self.lm_data))
        self.indexes = []
        for dataset_index, curr_data in enumerate(self.data + self.lm_data):
            lengths = [len(elem) for elem in curr_data[0]]
            ordered_indexes = np.argsort(lengths)
            for i, start in enumerate(range(0, len(curr_data[0]), self.batch_size)):
                curr_indexes = ordered_indexes[start:start+self.batch_size]
                self.indexes.append((dataset_index, curr_indexes))
        self.step = 0
        self.epoch = 0
        if self.shuffle:
            np.random.seed(self.seed)
        return self

    def __iter__(self):
        return self

    def steps_per_epoch(self, dataset_index=None):
        if dataset_index is None:
            return len(self._epoch_indexes)
        return len([x for x in self._epoch_indexes if x[0] == dataset_index])

    def _make_batch(self, data, indexes, vocabulary_shapes, shifts=None):
        shifts = shifts or [0] * len(data)
        answer = []
        for i, curr_data in enumerate(data):
            shape = [len(indexes)] + list(np.shape(curr_data[0]))
            if len(shape) > 1:
                max_length = max(len(curr_data[index]) for index in indexes)
                shape[1] = max_length
            curr_answer = np.zeros(shape=shape, dtype=int)
            for j, index in enumerate(indexes):
                # print(index, len(curr_data))
                curr_answer[j,:len(curr_data[index])] = curr_data[index]
            vocabulary_size = vocabulary_shapes[i]
            shift = shifts[i]
            if shift > 0:
                pad = np.zeros_like(curr_answer[:,:shift])
                curr_answer = np.concatenate([curr_answer[:,shift:], pad], axis=1)
            elif shift < 0:
                pad = np.zeros_like(curr_answer[:, shift:])
                curr_answer = np.concatenate([pad, curr_answer[:, :shift]], axis=1)
            if vocabulary_size is not None:
                curr_answer = to_one_hot(curr_answer, vocabulary_size)
            answer.append(curr_answer)
        return answer

    def __next__(self):
        if self.step == 0:
            self._epoch_indexes = [(i, elem) for i, elem in self.indexes
                                   if self.epoch < self.epochs[i] and self.epoch >= self.start_epochs[i]]
            if len(self._epoch_indexes) == 0:
                raise StopIteration()
            if self.shuffle:
                np.random.shuffle(self._epoch_indexes)
        dataset_index, indexes = self._epoch_indexes[self.step]
        vocabulary_size = self.data_vocabulary_shapes[dataset_index]
        target_vocabulary_size = self.target_vocabulary_shapes[dataset_index]
        if dataset_index >= len(self.data):
            lm_dataset_index = dataset_index - len(self.data)
            data = self.lm_data[lm_dataset_index]
            targets = [self.lm_data[lm_dataset_index][0] for _ in self.lm_shifts[lm_dataset_index]]
            shifts = self.lm_shifts[lm_dataset_index]
        else:
            vocabulary_size = self.data_vocabulary_shapes[dataset_index]
            target_vocabulary_size = self.target_vocabulary_shapes[dataset_index]
            data, targets = self.data[dataset_index], self.targets[dataset_index]
            shifts = None
        batch = self._make_batch(data, indexes, vocabulary_size)
        batch_targets = self._make_batch(targets, indexes, target_vocabulary_size, shifts=shifts)
        self.step += 1
        if self.step == len(self._epoch_indexes):
            self.step, self.epoch = 0, self.epoch + 1
        return dataset_index, batch, batch_targets


class MultimodelTrainer:

    def __init__(self, models, epochs, progbar_model_index=None, dev_model_index=None,
                 metrics=None, early_stopping=None, monitor="val_acc"):
        self.models = models
        self.epochs = epochs
        self.progbar_model_index = progbar_model_index or dev_model_index
        self.dev_model_index = dev_model_index
        self.metrics = metrics
        self.early_stopping = early_stopping
        self.monitor = monitor
        self._initialize()

    def _initialize(self):
        self.dev_model = self.models[self.dev_model_index] if self.dev_model_index is not None else None
        metrics = list(self.metrics or self.dev_model.metrics_names)
        metrics += ["val_" + metric for metric in metrics]
        callback_params = {"epochs": self.epochs[self.dev_model_index],
                           "do_validation": True,
                           "verbose": 1}
        self.callbacks = CallbackList()
        self._loggers = []
        for i, model in enumerate(self.models):
            logger = BaseLogger()
            logger.set_model(model)
            curr_callback_params = callback_params.copy()
            curr_callback_params["metrics"] = ["model_{}_{}".format(i, metric) for metric in model.metrics_names]
            logger.set_params(curr_callback_params)
            self.callbacks.append(logger)
            self._loggers.append(logger)
        if self.progbar_model_index is not None:
            self.progbar_callback = ProgbarLogger(count_mode="steps")
            self.progbar_callback.set_model(self.models[self.progbar_model_index])
            curr_callback_params = callback_params.copy()
            curr_callback_params["metrics"] = metrics
            for i, logger in enumerate(self._loggers):
                if i != self.progbar_model_index:
                    curr_callback_params["metrics"].extend(logger.params["metrics"])
            self.progbar_callback.set_params(curr_callback_params)
            self.callbacks.append(self.progbar_callback)
        else:
            self.progbar_callback = None
        if self.early_stopping is not None and self.dev_model is not None:
            self.early_stopping_callback = EarlyStopping(
                monitor=self.monitor, patience=self.early_stopping, restore_best_weights=True)
            self.early_stopping_callback.set_model(self.dev_model)
            self.early_stopping_callback.set_params(callback_params)
            self.callbacks.append(self.early_stopping_callback)
        else:
            self.early_stopping_callback = None


    def train(self, data: MultitaskDataGenerator, dev_data: SimpleDataGenerator):
        self.callbacks.on_train_begin()
        for dataset_index, x, y in data:
            if data.step == 1:
                if self.progbar_callback is not None:
                    self.progbar_callback.params["steps"] =\
                        data.steps_per_epoch(dataset_index=self.progbar_model_index)
                self.callbacks.on_epoch_begin(data.epoch)
            model: Model = self.models[dataset_index]
            batch_logs = {"batch": data.step, "size": data.batch_size}
            batch_logs_for_logger = batch_logs.copy()
            if dataset_index == self.progbar_model_index:
                self.progbar_callback.on_batch_begin(data.step, batch_logs)
            self._loggers[dataset_index].on_batch_begin(batch_logs)
            outs = model.train_on_batch(x, y)
            for metric, value in zip(model.metrics_names, outs):
                batch_logs[metric] = value
                batch_logs_for_logger["model_{}_{}".format(dataset_index, metric)] = value
            if dataset_index == self.progbar_model_index:
                self.progbar_callback.on_batch_end(data.step-1, batch_logs)
            self._loggers[dataset_index].on_batch_end(data.step-1, batch_logs_for_logger)
            if data.step == 0:
                outs = evaluate_generator(self.dev_model, dev_data, steps=dev_data.steps_per_epoch)
                epoch_logs = dict()
                for metric, value in zip(self.dev_model.metrics_names, outs):
                    epoch_logs["val_" + metric] = value
                self.callbacks.on_epoch_end(data.epoch-1, epoch_logs)
                if data.epoch == self.epochs:
                    break
            if self.dev_model is not None and getattr(self.dev_model, "stop_training", False):
                break
        self.callbacks.on_train_end()
        return self






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
        # if shuffle:
        #     print("")
        #     print(indexes[0][1])
        for i, bucket_indexes in indexes:
            curr_bucket, curr_targets = data[i], targets[i]
            data_to_yield = [elem[bucket_indexes] for elem in curr_bucket]
            targets_to_yield = to_one_hot(curr_targets[bucket_indexes], classes_number)
            yield data_to_yield, targets_to_yield
        nsteps += 1