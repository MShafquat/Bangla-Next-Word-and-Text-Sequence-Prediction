import os
import pickle
from collections import Counter, defaultdict
from nltk import trigrams

def dd():
    return defaultdict(int)
model = defaultdict(dd)

# count frequencies
print("Counting frequencies")
with open('../processed_data/processed_data.txt') as data:
    for i, line in enumerate(data):
        i += 1
        for w1, w2, w3 in trigrams(line, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1
        if i % 10000 == 0:
            print(f"Processed {i} lines")
            break

# convert to probabilities
print("Converting to probabilities")
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
print(f"Model length: {len(model)}")

# save the model
os.makedirs("../model", exist_ok=True)
with open("../model/trigram-model.pkl", "wb") as pkl_handle:
    pickle.dump(model, pkl_handle)
