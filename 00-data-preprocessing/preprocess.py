import os
import regex
from bnlp import NLTKTokenizer

def pure_sentence(str):
    """
    Check if str is a pure bengali sentence containing
    only bengali letters, digits, and punctuations
    """
    return bool(regex.fullmatch(
        r'\P{L}*\p{Bengali}+(?:\P{L}+\p{Bengali}+)*\P{L}*', str))

def process_sentence(str):
    """
    Remove punctuations from a sentence
    """
    for symbol in punctuations:
        str = str.replace(symbol, ' ')
    str = " ".join(str.split())
    return str


def preprocess_file(raw_data_filepath, processed_data_dir, processed_filename):
    """
    Reads from raw_data_filepath, and writes processed sentences from
    the file into processed_data_dir/processed_filename one sentence per line.
    """
    bnltk = NLTKTokenizer()
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
            if i % 10000 == 0:
                print(f"Processed {i} lines")

    processed_data_file.close()


if __name__ == '__main__':
    raw_data_filepath = "../data/bn_dedup.txt"
    processed_data_dir = "../processed_data/"
    processed_data_filepath = processed_data_dir + "processed_data.txt"
    preprocess_file(raw_data_filepath,
                    processed_data_dir,
                    processed_data_filepath)
