from tokenizers import ByteLevelBPETokenizer
import os
from pathlib import Path

tokenizer = ByteLevelBPETokenizer()

# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')]

# train the tokenizer
tokenizer.train(files=files,
                vocab_size=30_000,
                min_frequency=10,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

# create a model directory if does not exist and save the tokenizer
os.makedirs(project_root / 'models/bn-roberta/', exist_ok=True)
tokenizer.save_model(str(project_root / 'models/bn-roberta/'))
