import numpy as np

from read import read_splitted
from neural.neural_LM import load_lm, END

class Embedder:

    def __init__(self, model_file, reverse_model_file, pad=True):
        self.model = load_lm(model_file)
        self.reverse_model = load_lm(reverse_model_file)
        self.pad = pad

    def transform(self, data):
        first_states, _ = self.model.predict_states_batch(data)
        first_probs = self.model._head_func([first_states])[0]
        second_states, _ = self.reverse_model.predict_states_batch(data)
        second_probs = self.reverse_model._head_func([second_states])[0]
        answer, probs_answer = [], []
        for i, elem in enumerate(data):
            L = len(elem[0])
            if self.pad:
                start, end = 0, L+2
                right_start, right_end = L, None
            else:
                start, end = 1, L+1
                right_start, right_end = L-1, None
            curr_first_states = first_states[i,start:end]
            curr_first_probs = first_probs[i,start:end]
            curr_second_states = second_states[i, right_start:right_end:-1]
            curr_second_probs = second_probs[i, right_start:right_end:-1]
            if self.pad:
                pad_state = np.zeros_like(curr_second_states[0])[None,:]
                pad_probs = np.zeros_like(curr_second_probs[0])[None,:]
                pad_probs[0,END] = 1
                curr_second_states = np.concatenate([curr_second_states, pad_state], axis=0)
                curr_second_probs = np.concatenate([curr_second_probs, pad_probs], axis=0)
            states = np.concatenate([curr_first_states, curr_second_states], axis=-1)
            # states = curr_second_states
            answer.append(states)
            probs_answer.append([curr_first_probs, curr_second_probs])
        return answer


if __name__ == "__main__":
    model_file = "models/sigmorphon/lm/sami-small.json"
    reverse_model_file = "models/sigmorphon/lm/sami-small-reverse.json"
    embedder = Embedder(model_file, reverse_model_file)
    data, _ = read_splitted("/home/alexeysorokin/data/Data/Morpheme/North Sami/sme_al_annotations_v2/train.full", transform_to_BMES=False)
    data = [[elem] for elem in data]
    embeddings = embedder.transform(data)