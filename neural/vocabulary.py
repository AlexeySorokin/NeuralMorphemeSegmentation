from collections import defaultdict
import inspect


from .common import *


def vocabulary_from_json(info):
    info_to_initialize = dict(elem for elem in info.items() if elem[0][-1] != "_")
    vocab = Vocabulary(**info_to_initialize)
    args = dict()
    for attr, val in info.items():
        if attr[-1] == "_" and not isinstance(getattr(Vocabulary, attr, None), property):
            setattr(vocab, attr, val)
    if hasattr(vocab, "symbols_") and not hasattr(vocab, "symbol_codes_"):
        vocab.symbol_codes_ = {x: i for i, x in enumerate(vocab.symbols_)}
    return vocab

class Vocabulary:

    def __init__(self, character=False, min_count=1):
        self.character = character
        self.min_count = min_count

    def train(self, text):
        symbols = defaultdict(int)
        for elem in text:
            if self.character:
                curr_symbols = [symbol for x in elem for symbol in x]
            else:
                curr_symbols = elem
            for x in curr_symbols:
                symbols[x] += 1
        symbols = [x for x, count in symbols.items() if count >= self.min_count]
        self.symbols_ = AUXILIARY + sorted(symbols)
        self.symbol_codes_ = {x: i for i, x in enumerate(self.symbols_)}
        return self

    def toidx(self, x):
        return self.symbol_codes_.get(remove_token_field(x), UNKNOWN)

    def __getitem__(self, item):
        return self.symbols_[item]

    @property
    def symbols_number_(self):
        return len(self.symbols_)

    def jsonize(self):
        info = {attr: val for attr, val in inspect.getmembers(self)
                if (not(attr.startswith("__") or inspect.ismethod(val))
                    and (attr[-1] != "_" or attr in ["symbols_", "symbol_codes_", "tokens_"]))}
        return info


def remove_token_field(x):
    splitted = x.split("token=")
    return splitted[0][:-1] if (len(splitted) > 1) else splitted[0]


def remove_token_fields(text):
    new_text = []
    for sent in text:
        new_text.append([remove_token_field(x) for x in sent])
    return new_text




