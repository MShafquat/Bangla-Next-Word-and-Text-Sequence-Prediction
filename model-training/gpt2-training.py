from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, TFGPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import tensorflow as tf

# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')][:1]

tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-bengali")

with tf.distribute.get_strategy().scope():
    model = TFGPT2LMHeadModel.from_pretrained(
        "flax-community/gpt2-bengali", from_pt=True)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=model.compute_loss)

train_dataset = load_dataset('text', data_files=files)
train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text']), batched=True, num_proc=4, remove_columns=['text'])
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors="tf"
)

train_dataset = train_dataset.to_tf_dataset(
    batch_size=4, collate_fn=data_collator
)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, mode='auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint(str(
    project_root / 'models/bn_gpt2/bn_gpt2_{epoch:02d}_{val_loss:.4f}.h5'),
    monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(train_dataset, epochs=100, batch_size=128,
                    callbacks=[checkpoint])
