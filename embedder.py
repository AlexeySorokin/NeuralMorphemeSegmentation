import numpy as np

from read import read_splitted
from neural.neural_LM import load_lm

class Embedder:

    def __init__(self, model_file, reverse_model_file):
        self.model = load_lm(model_file)
        self.reverse_model = load_lm(reverse_model_file)


    def transform(self, data):
        first_states, _ = self.model.predict_states_batch(data)
        second_states, _ = self.reverse_model.predict_states_batch(data)
        answer = []
        for i, elem in enumerate(data):
            L = len(elem[0])
            # берём левые состояния уже после прочтения текущей буквы
            curr_first_states = first_states[i,1:L+1]
            curr_second_states = second_states[i,L-1::-1]
            states = np.concatenate([curr_first_states, curr_second_states], axis=-1)
            answer.append(states)
        return answer


if __name__ == "__main__":
    model_file = "models/sigmorphon/lm/sami-small.json"
    reverse_model_file = "models/sigmorphon/lm/sami-small-reverse.json"
    embedder = Embedder(model_file, reverse_model_file)
    data, _ = read_splitted("/home/alexeysorokin/data/Data/Morpheme/North Sami/sme_al_annotations_v2/train.full", transform_to_BMES=False)
    data = [[elem] for elem in data]
    embeddings = embedder.transform(data)