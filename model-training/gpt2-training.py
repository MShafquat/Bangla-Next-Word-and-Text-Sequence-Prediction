from calendar import c
import math
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFKC
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')][:2]

# training the tokenizer
# tokenizer = ByteLevelBPETokenizer()
# normalizer = normalizers.Sequence([NFKC()])
# tokenizer.normalizer = normalizer
# tokenizer.train(files=files,
#                 vocab_size=30_000,
#                 min_frequency=10,
#                 special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

# # create a model directory if does not exist and save the tokenizer
# os.makedirs(project_root / 'models/bn-gpt2/', exist_ok=True)
# tokenizer.save_model(str(project_root / 'models/bn-gpt2/'))
tokenizer = GPT2Tokenizer.from_pretrained(str(project_root / 'models/bn-gpt2/'))
tokenizer.add_special_tokens({"pad_token": "<pad>"})

# creating dataset
print("Creating dataset")
dataset = load_dataset('text', data_files={'train': files[:-1], 'test': files[-1]})
max_seq_length = 512
num_proc = 4


def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [line for line in examples["text"]
                        if len(line) > 0 and not line.isspace()]
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=["text"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)
print("Dataset created")

# creating the configurations from which the model can be made
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# creating the model
model = GPT2LMHeadModel(config).to("cuda")

training_args = TrainingArguments(
    output_dir=str(project_root / 'models/bn-gpt2/'),
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=True,
    optim="adafactor",
    eval_steps=500,
    save_steps=1000,
    warmup_steps=500,
    evaluation_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)

trainer.train()
trainer.save_model()
