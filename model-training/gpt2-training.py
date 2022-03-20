import os
from pathlib import Path
import pickle
from transformers import AutoTokenizer, TFGPT2LMHeadModel
from transformers import WEIGHTS_NAME, CONFIG_NAME
import tensorflow as tf
import matplotlib.pyplot as plt

# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')]
model_dir = project_root / 'models/bn_gpt2/'
os.makedirs(model_dir, exist_ok=True)

# create tokenizer and model from pretrained model
tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-bengali")
tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})
model = TFGPT2LMHeadModel.from_pretrained(
    "flax-community/gpt2-bengali", from_pt=True)

# create a string from files text
with open(files[0], 'r') as f:
    text = f.read()[:5_000]

# tokenize the text
string_tokenized = tokenizer.encode(text)

# create a list of block size tokens
examples = []
block_size = 100
BATCH_SIZE = 12
BUFFER_SIZE = 1000
for i in range(0, len(string_tokenized) - block_size + 1, block_size):
    examples.append(string_tokenized[i:i + block_size])

# create inputs and labels
inputs, labels = [], []
for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# create model parameters
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(loss=[loss, *[None] * model.config.n_layer], optimizer=adam, metrics=[metric])
checkpoint = tf.keras.callbacks.ModelCheckpoint(str(model_dir), monitor='loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(dataset, epochs=3, callbacks=[checkpoint])

print(history.history)
with open(str(model_dir / 'history'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# save model
model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
output_config_file = os.path.join(model_dir, CONFIG_NAME)
# save model and model configs
model.save_pretrained(model_dir)
model_to_save.config.to_json_file(output_config_file)
# save tokenizer
tokenizer.save_pretrained(model_dir)

fig = plt.figure(figsize=(10, 6))
plt.plot(history.history['logits_accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
fig.savefig(str(model_dir / 'accuracy.png'), dpi=fig.dpi)

fig = plt.figure(figsize=(10, 6))
plt.plot(history.history['logits_loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
fig.savefig(str(model_dir / 'loss.png'), dpi=fig.dpi)
