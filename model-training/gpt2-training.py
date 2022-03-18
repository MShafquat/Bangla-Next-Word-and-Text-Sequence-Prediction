from pathlib import Path
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')]

tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-bengali")
tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})
model = GPT2LMHeadModel.from_pretrained("flax-community/gpt2-bengali")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

dataset = load_dataset('text', data_files={'train': files[:1], 'eval': files[1:2]})
dataset = dataset.map(lambda example: tokenizer(example['text']), batched=True, num_proc=4, remove_columns=['text'])

training_arguments = TrainingArguments(
    output_dir=str(project_root / 'models/bn-gpt2'),
    overwrite_output_dir=True,
    num_train_epochs=3,
    save_total_limit=2,
    save_steps=1000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    learning_rate=1e-5,
    fp16=True,
    fp16_opt_level='O1',
    max_grad_norm=1.0,
    resume_from_checkpoint=str(project_root / 'models/bn-gpt2'),
    logging_dir=str(project_root / 'logs/bn-gpt2'),
    logging_first_step=False,
    logging_steps=500,
    # do_eval=True,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    # eval_dataset=dataset['eval'],
)

trainer.train()
trainer.save_model()

history = trainer.logs()
print(history)

with open(str(project_root / 'models/bn_gpt2/history'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
