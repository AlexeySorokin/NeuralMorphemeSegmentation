# чтение и разметка данных
from collections import defaultdict
import numpy as np


def normalize_morph_types(morphs, aliases=None):
    if aliases is None:
        return morphs
    aliases = dict(aliases)
    morphs = [aliases.get(x, x) for x in morphs]
    return morphs

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


def read_splitted(infile, transform_to_BMES=True, make_morph_types=None,
                  n=None, morph_sep="/", shuffle=True, morph_aliases=None):
    source, targets = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                break
            word, analysis = line.split("\t")
            morphs = analysis.split(morph_sep)
            if make_morph_types is None:
                morph_types = ["None"] * len(morphs)
            elif make_morph_types == "suff":
                morph_types = ["ROOT"] + ["SUFF"] * (len(morphs) - 1)
            morph_types = normalize_morph_types(morph_types, morph_aliases)
            if transform_to_BMES:
                target = generate_BMES(morphs, morph_types)
            else:
                target = morphs
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


def read_lowresource_format(infiles, remove_frequent=False, min_count=5, n=None, shuffle=False):
    sents, source, targets = [], [], []
    if isinstance(infiles, str):
        infiles = [infiles]
    for infile in infiles:
        with open(infile, "r", encoding="utf8") as fin:
            curr_sent = []
            for line in fin:
                line = line.strip()
                if line == "":
                    if len(curr_sent) > 0:
                        sents.append(curr_sent)
                        if n is not None and len(source) >= n:
                            break
                    curr_sent = []
                    continue
                if "\t" in line:
                    word, morphs = line.split("\t")
                    word = "".join(x for x in word if x != " ")
                    morphs = morphs.split()
                    morphs = [morphs[0]] + [elem.split("_")[0] for elem in morphs[1:]]
                    morph_types = ["ROOT"] + ["SUFF"] * (len(morphs) - 1)
                    target = generate_BMES(morphs, morph_types)
                else:
                    word, morphs, morph_types = line, None, None
                    target = None
                source.append(word)
                targets.append(target)
                if target is not None and len(word) != len(target):
                    print(word, target)
                curr_sent.append(word)
    if remove_frequent:
        counts = defaultdict(int)
        for word, target in zip(source, targets):
            counts[(word, tuple(target))] += 1
        source, targets = [], []
        for (word, target), count in counts.items():
            if count >= min_count:
                count = min_count + int(np.log2(1 + count - min_count))
            source += [word] * count
            targets += [target] * count
        indexes = np.random.permutation(len(source))
        source = [source[i] for i in indexes]
        targets = [targets[i] for i in indexes]
    return sents, source, targets


def read_BMES(infile, transform_to_BMES=True, n=None,
              morph_sep="/" ,sep=":", shuffle=True, morph_aliases=None):
    source, targets = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                break
            word, analysis = line.split("\t")
            word = word.strip()
            analysis = [x.split(sep) for x in analysis.split(morph_sep)]
            morphs, morph_types = [elem[0] for elem in analysis], [elem[1] for elem in analysis]
            morph_types = normalize_morph_types(morph_types, morph_aliases)
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


def read_words(infile, min_length=-1, n=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if "\t" in line and line.split("\t")[0].isdigit():
                line = line.split("\t")[1]
            if line != "" and len(line) >= min_length and all(not x.isdigit() for x in line) and any(x.isalpha() for x in line):
                answer.append(line)
    np.random.shuffle(answer)
    if n is not None and n >= 0:
        answer = answer[:n]
    return answer