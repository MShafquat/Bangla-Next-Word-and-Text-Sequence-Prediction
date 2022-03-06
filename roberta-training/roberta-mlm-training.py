from transformers import RobertaTokenizer
from pathlib import Path

# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')]

# initialize the tokenizer using the tokenizer we initialized and saved to file
tokenizer = RobertaTokenizer.from_pretrained(project_root / 'models/bn-roberta', max_len=512) 
