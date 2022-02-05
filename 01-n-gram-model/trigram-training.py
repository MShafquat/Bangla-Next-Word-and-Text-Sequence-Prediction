import os
import pickle
from collections import Counter, defaultdict
from nltk import ngrams
from nltk.probability import ConditionalFreqDist

class NgramModel():
    def __init__(self, filepath=None, n=0, train_test_split=0.8):
        self.model = dict()
        self.filepath = filepath
        if filepath:
            self.num_lines = sum(1 for _ in open(self.filepath))
        self.n = n
        self.train_test_split = 0.8

    def train_model(self):
        num_lines_to_read = int(self.num_lines * self.train_test_split)
        with open(self.filepath) as datafile:
            for i, line in enumerate(datafile):
                for bag in ngrams(line.split(), self.n):
                    prev_words = bag[:-1]
                    cur_word = bag[-1]
                    self.model[prev_words] = self.model.get(prev_words, {})
                    self.model[prev_words][cur_word] = self.model.get(
                        cur_word, 0)
                    self.model[prev_words][cur_word] += 1
                if (i + 1) == num_lines_to_read:
                    break
                if (i + 1) % 10000 == 0:
                    print(f"Trained {i+1} lines")
                    break
        for prev_words in self.model:
            total_count = sum(self.model[prev_words].values())
            for cur_word in self.model[prev_words]:
                self.model[prev_words][cur_word] /= total_count
        return self.model

    def predict(self, sentence):
        words = sentence.split()
        bag = words[-self.n+1:]
        return self.model.get(tuple(bag), {})

    def train_perplexity(self):
        res = 1
        for bag in self.model:
            for key in self.model[bag]:
                val = self.model[bag][key]
                if val == 0:
                    res *= 100
                    res = res ** (1/self.n)
                else:
                    res *= (1 / val)
                    res = res ** (1/self.n)
        return res

    def test_perplexity(self):
        num_lines_to_skip = int(self.num_lines * self.train_test_split)
        res = 1
        with open(self.filepath) as datafile:
            for i, line in enumerate(datafile):
                if i > num_lines_to_skip:
                    for bag in ngrams(line.split(), self.n):
                        prev_words = bag[:-1]
                        cur_word = bag[:-1]
                        try:
                            val = model[prev_words][cur_word]
                            if val == 0:
                                res *= 100
                                res = res ** (1/self.n)
                            else:
                                res *= (1 / val)
                                res = res ** (1/self.n)
                        except:
                            res *= 100
                            res = res ** (1/self.n)
        return res

    def save(self, model_path):
        # save the model
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(model_path, "wb") as pkl_handle:
            pickle.dump(self.model, pkl_handle)

    def load(self, model_path, n):
        with open(model_path, "rb") as model_file:
            self.model = pickle.load(model_file)
            print(len(self.model))
        self.n = n        


if __name__ == '__main__':
    trigram = NgramModel("../processed_data/processed_data.txt", n=3)
    trigram.train_model()
    trigram.save("../model/trigram-model.pkl")
    print(f"Train perplexity: {trigram.train_perplexity()}")
    print(f"Test perplexity: {trigram.test_perplexity()}")

    while True:
        try:
            print(trigram.predict(input()))
        except KeyboardInterrupt:
            break

