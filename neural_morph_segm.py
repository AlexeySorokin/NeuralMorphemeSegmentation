import sys
import numpy as np

import json
# import ujson as json

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from evaluate import measure_quality
from network import Partitioner, MultitaskPartitioner
from read import read_BMES, read_splitted, read_lowresource_format, read_words


def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        config = json.load(fin)
    if "save_params" not in config and ("model_file" in config):
        config["save_params"] = {"model_file": config.pop("model_file")}
    if "input_format" not in config:
        config["input_format"] = "types" if config.get("use_morpheme_types", True) else "splitted"
    config["use_morpheme_types"] = config.get("use_morpheme_types", config["input_format"] != "splitted")
    if config.get("use_lm", False):
        config["use_oneside"] = True
    return config


def load_cls(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback") or key.endswith("model_files"))}
    args['callbacks'] = []
    cls = eval(args.pop("cls", "Partitioner"))
    # создаём классификатор
    inflector = cls(**args)
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
    if "language_model_files" in json_data:
        for i, (model, model_file) in enumerate(
                zip(inflector.language_models_, json_data['language_model_files'])):
            model.load_weights(model_file)

    return inflector


SHORT_ARGS = "a:"

def make_state_filename(filename, state):
    answer = ("{}-{}".format(filename, state) if "." not in filename else
              "{0}-{2}.{1}".format(*(filename.split(".", maxsplit=1)), state))
    return answer

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    kbt.set_session(tf.Session(config=config))
    if len(sys.argv) < 2:
        sys.exit("Pass config file")
    config_file = sys.argv[1]
    params = read_config(config_file)
    random_state = params.get("random_state", [261])
    if isinstance(random_state, str):
        random_state = list(map(int, random_state.split(",")))
        # if "train_file" not in params:
        #     random_state = random_state[:1]
    elif isinstance(random_state, int):
        random_state = [random_state]
    input_format = params["input_format"]
    read_func = (read_BMES if input_format == "types" else
                 read_lowresource_format if input_format == "low_resource" else
                 read_splitted)
    use_morpheme_types = params["use_morpheme_types"]
    # read_func = read_BMES if use_morpheme_types else read_splitted
    read_params = params.get("read_params", dict())
    if "train_file" in params and params.get("to_train", True):
        n = params.get("n_train")  # число слов в обучающей+развивающей выборке
        if input_format == "low_resource":
            sents, inputs, targets = read_func(params["train_file"], n=n, **read_params)
        else:
            inputs, targets = read_func(params["train_file"], n=n, **read_params)
        if "dev_file" in params:
            n = params.get("n_dev")  # число слов в обучающей+развивающей выборке
            if input_format == "low_resource":
                _, dev_inputs, dev_targets = read_func(params["dev_file"], n=n)
            else:
                dev_inputs, dev_targets = read_func(params["dev_file"], n=n, **read_params)
        else:
            dev_inputs, dev_targets = None, None
        if "lm_train_file" in params:
            train_params = {"lm_data": read_words(params["lm_train_file"], min_length=5, n=10000)}
        else:
            train_params = dict()
    else:
        inputs, targets, dev_inputs, dev_targets = None, None, None, None
        train_params = dict()
    qualities = dict()
    for curr_random_state in random_state:
        np.random.seed(curr_random_state)  # для воспроизводимости
        if not "load_file" in params:
            partitioner_params = params.get("model_params", dict())
            if params.get("use_oneside", False) and "use_lm" not in partitioner_params:
                partitioner_params["use_lm"] = params.get("use_lm", False)
            # partitioner_params["use_morpheme_types"] = use_morpheme_types
            partitioner = MultitaskPartitioner if params.get("use_oneside", False) else Partitioner
            cls = partitioner(**partitioner_params)
            # cls = MultitaskPartitioner(**partitioner_params)
        else:
            load_file = params["load_file"]
            if len(random_state) > 1:
                load_file = make_state_filename(load_file, curr_random_state)
            cls = load_cls(load_file)
        if "save_file" in params:
            save_file, model_file = params["save_file"], params.get("model_file")
            if len(random_state) > 1:
                save_file = make_state_filename(save_file, curr_random_state)
                if model_file is not None:
                    model_file = make_state_filename(model_file, curr_random_state)
        else:
            save_file, model_file = None, None
        if inputs is not None:
            cls.train(inputs, targets, dev_inputs, dev_targets, **train_params,
                      model_file=model_file, verbose=(len(random_state) == 1))
        if "save_file" in params:
            cls.to_json(save_file, model_file)
        if "test_file" in params:
            test_file = params["test_file"]
            if input_format == "low_resource":
                test_sents, test_inputs, test_targets = read_func(test_file, shuffle=False)
            else:
                test_inputs, test_targets = read_func(test_file, shuffle=False, **read_params)
                test_sents = None
            # inputs, targets = read_input(params["test_file"])
            predicted_targets = cls._predict_probs(test_inputs)
            if params.get("measure_quality", True):
                measure_last = params.get("measure_last", use_morpheme_types)
                quality = measure_quality(test_targets, [elem[0] for elem in predicted_targets],
                                          english_metrics=params.get("english_metrics", False),
                                          measure_last=measure_last)
                for key, value in sorted(quality):
                    print("{}={:.2f}".format(key, 100*value))
                    if key not in qualities:
                        qualities[key] = [value]
                    else:
                        qualities[key].append(value)
            if "outfile" in params:
                outfile = params["outfile"]
                if len(random_state) > 1:
                    outfile = make_state_filename(outfile, curr_random_state)
                output_probs = params.get("output_probs", True)
                format_string = "{}\t{}\t{}" if output_probs else "{}\t{}"
                output_morpheme_types = params.get("output_morpheme_types", True)
                morph_format_string = "{}-{}" if output_morpheme_types else "{}"
                with open(outfile, "w", encoding="utf8") as fout:
                    morphs = [cls.labels_to_morphemes(word, labels, return_types=True) for word, labels in zip(inputs, targets)]
                    corr_segmentation_probs = cls.prob(inputs, morphs)
                    for r, (word, corr_labels, (labels, probs)) in enumerate(zip(inputs, targets, predicted_targets)):
                        morphemes, morpheme_probs, morpheme_types = cls.labels_to_morphemes(
                            word, labels, probs, return_probs=True, return_types=True)
                        s = format_string.format(
                            word, "-".join(morph_format_string.format(*elem)
                                           for elem in zip(morphemes, morpheme_types)),
                            " ".join("{:.2f}".format(100*x) for x in morpheme_probs))
                        fout.write(s)
                        if params.get("output_errors", True) and corr_labels != labels:
                            fout.write("\tERROR")
                            fout.write("\n")
                            fout.write(" ".join("{}-{}".format(*elem) for elem in zip(*morphs[r])) + "\t")
                            fout.write(" ".join("{:.2f}".format(100 * x) for x in corr_segmentation_probs[r]))
                        fout.write("\n")
            if "predictions_file" in params:
                predictions_file = params["predictions_file"]
                if len(random_state) > 1:
                    predictions_file = make_state_filename(predictions_file, curr_random_state)
                with open(predictions_file, "w", encoding="utf8") as fout:
                    sent_index = 0
                    if test_sents is not None:
                        sent_lengths = np.cumsum([len(sent) for sent in test_sents])
                    for r, (word, corr_labels, (labels, probs)) in enumerate(zip(test_inputs, test_targets, predicted_targets), 1):
                        morphemes, morpheme_probs, morpheme_types = cls.labels_to_morphemes(
                            word, labels, probs, return_probs=True, return_types=True)
                        fout.write(word + "\t" + " ".join(morphemes) + "\n")
                        if test_sents is not None and (sent_index < len(sent_lengths)) and r == sent_lengths[sent_index]:
                            sent_index += 1
                            fout.write("\n")
                        # fout.write(format_string.format(
                        #     word, "#".join(morphemes), "-".join(
                        #         "{:.2f}/{}".format(100*x, y) for x, y in zip(morpheme_probs, morpheme_types))))
    if len(qualities) > 0:
        for key, values in sorted(qualities.items()):
            print("{}: {:.2f}({:.2f})".format(key, 100 * np.mean(values), 100 * np.std(values)), end="")
            print("\t" + ",".join("{:.2f}".format(100*x) for x in values))