from collections import defaultdict
import dill
import os
import pickle
import random
import numpy as np
from bnlp import NLTKTokenizer
from nltk import ngrams

class NgramModel():
    def __init__(self, filepath=None, n=0, num_sentences=0,
                 train_test_split=0.8):
        self.n = n
        self.model = defaultdict(lambda: defaultdict(int))
        self.filepath = filepath
        self.bnltk = NLTKTokenizer()
        self.data = None
        self.num_sentences = num_sentences
        self.train_test_split = 0.8
        self.train_size = int(self.num_sentences * self.train_test_split)
        self.train_data = None
        self.test_data = None
        if filepath:
            self.data = open(filepath).readlines()[:num_sentences]
            random.shuffle(self.data)
            self.data = [self.bnltk.word_tokenize(line) for line in self.data]
            self.train_data = self.data[:self.train_size]
            self.test_data = self.data[self.train_size:]

    def train_model(self):
        for line in self.train_data:
            for words in ngrams(line, self.n):
                self.model[words[:-1]][words[-1]] += 1

        for pre in self.model:
            total_count = sum(self.model[pre].values())
            for cur in self.model[pre]:
                self.model[pre][cur] /= total_count

    def predict(self, sentence):
        sentence = self.bnltk.word_tokenize(sentence)[-self.n+1:]
        return self.model[sentence]

    def train_perplexity(self):
        pass
    
    def test_perplexity(self):
        pass

    def save(self, model_path):
        # save the model
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(model_path, "wb") as pkl_handle:
            dill.dump(self.model, pkl_handle)

    def load(self, model_path, n):
        with open(model_path, "rb") as model_file:
            self.model = dill.load(model_file)
        self.n = n        


if __name__ == '__main__':
    trigram = NgramModel("../processed_data/processed_data.txt",
                         num_sentences=100_000, n=3)
    print("Model loaded")
    trigram.train_model()
    trigram.save("../model/trigram-model.pkl")
    trigram.load("../model/trigram-model.pkl", n=3)
    print(f"Train perplexity: {trigram.train_perplexity()}")
    print(f"Test perplexity: {trigram.test_perplexity()}")

    while True:
        try:
            print(trigram.predict(input(">>> ")))
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            break
