# чтение и разметка данных
import numpy as np


def generate_BMES(morphs, morph_types):
    answer = []
    for morph, morph_type in zip(morphs, morph_types):
        if len(morph) == 1:
            answer.append("S-" + morph_type)
        else:
            answer.append("B-" + morph_type)
            answer.extend(["M-" + morph_type] * (len(morph) - 2))
            answer.append("E-" + morph_type)
    return answer


def read_splitted(infile, transform_to_BMES=True, n=None, morph_sep="/", shuffle=True):
    source, targets = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                break
            word, analysis = line.split("\t")
            morphs = analysis.split(morph_sep)
            morph_types = ["None"] * len(morphs)
            if transform_to_BMES:
                target = generate_BMES(morphs, morph_types)
            else:
                target = morph_types
            source.append(word)
            targets.append(target)
    indexes = list(range(len(source)))
    if shuffle:
        np.random.shuffle(indexes)
    if n is not None:
        indexes = indexes[:n]
    source = [source[i] for i in indexes]
    targets = [targets[i] for i in indexes]
    return source, targets


def read_BMES(infile, transform_to_BMES=True, n=None,
              morph_sep="/" ,sep=":", shuffle=True):
    source, targets = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                break
            word, analysis = line.split("\t")
            analysis = [x.split(sep) for x in analysis.split(morph_sep)]
            morphs, morph_types = [elem[0] for elem in analysis], [elem[1] for elem in analysis]
            target = generate_BMES(morphs, morph_types) if transform_to_BMES else morphs
            source.append(word)
            targets.append(target)
    indexes = list(range(len(source)))
    if shuffle:
        np.random.shuffle(indexes)
    if n is not None:
        indexes = indexes[:n]
    source = [source[i] for i in indexes]
    targets = [targets[i] for i in indexes]
    return source, targets


def partition_to_BMES(s1, s2):
    morphemes = s1.split("/")
    labels = s2.split(" , ")
    answer = []
    for l, m in zip(labels, morphemes):
        length = len(m)
        if l.startswith("Корень"):
            if m.startswith("-"):
                    answer.append("S-HYPH")
                    length -= 1
            if length == 1:
                answer.append("S-ROOT")
            else:
                answer.append("B-ROOT")
                for i in range(length-2):
                    answer.append("M-ROOT")
                answer.append("E-ROOT")

        elif l.startswith("Приставка"):
            if m.startswith("-"):
                    answer.append("S-HYPH")
                    length -= 1
            if length == 1:
                answer.append("S-PREF")
            else:
                answer.append("B-PREF")
                for i in range(length-2):
                    answer.append("M-PREF")
                answer.append("E-PREF")

        elif l.startswith("Суффикс"):
            if length == 1:
                answer.append("S-SUFF")
            else:
                answer.append("B-SUFF")
                for i in range(length-2):
                    answer.append("M-SUFF")
                answer.append("E-SUFF")

        elif l.startswith("Соединительная гласная") is True:
            answer.append("S-LINK")

        elif l.startswith("Окончание") is True:
            if length == 1:
                answer.append("S-END")
            else:
                answer.append("B-END")
                for i in range(length-2):
                    answer.append("M-END")
                answer.append("E-END")

        #elif l.startswith("Нулевое окончание") is True:
            #answer.append("S-NULL_END")

        elif l.startswith("Постфикс") is True:
            if m.startswith("-") is True:
                answer.append("HYPH")
                length -= 1
            answer.append("B-POSTFIX")
            for i in range(length-2):
                answer.append("M-POSTFIX")
            answer.append("E-POSTFIX")

    return answer


def extract_morpheme_type(x):
    return x[2:].lower()


def read_input(infile, transform_to_BMES=True, n=None, shuffle=True):
    source, targets = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                break
            word, morphs, analysis = line.split(" | ")
            target = partition_to_BMES(morphs, analysis) if transform_to_BMES else morphs
            source.append(word)
            targets.append(target)
    if n is not None:
        indexes = list(range(len(source)))
        if shuffle:
            np.random.shuffle(indexes)
        indexes = indexes[:n]
        source = [source[i] for i in indexes]
        targets = [targets[i] for i in indexes]
    return source, targets