import os
from bnlp import NLTKTokenizer
from bnlp.corpus import letters, digits

def pure_sentence(str):
    """
    Check if a str is a pure bengali sentence containing
    only bengali letters and digits
    """
    valids = letters + digits
    return all(thing in set(sentence_token) for thing in valids)


bnltk = NLTKTokenizer()

raw_data_filepath = "../data/bn_dedup.txt"
processed_data_dir = "../processed_data/"
processed_data_filepath = processed_data_dir + "processed_data.txt"

os.makedirs(processed_data_dir, exist_ok=True)
if os.path.exists(processed_data_filepath):
    os.remove(processed_data_filepath)
processed_data_file = open(processed_data_filepath, "a+")

print(f"Processing {raw_data_filepath}")
with open(raw_data_filepath) as raw_data_file:
    for i, line in enumerate(raw_data_file):
        sentence_tokens = bnltk.sentence_tokenize(line)
        for sentence_token in sentence_tokens:
            if pure_sentence(sentence_token):
                processed_data_file.write(sentence_token)
                processed_data_file.write("\n")
        if i % 1000 == 0:
            print(f"Processed {i} lines")

processed_data_file.close()
