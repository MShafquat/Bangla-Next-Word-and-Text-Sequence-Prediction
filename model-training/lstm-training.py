import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from transformers import AutoTokenizer
from matplotlib import pyplot as plt


class Attention(tf.keras.layers.Layer):
    def init(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(
            input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name='att_bias', shape=(
            input_shape[-2], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        return K.sum(output, axis=1)


def create_lstm_model(vocab_size, block_size, with_attention=False):
    # model training without attention
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(
        vocab_size, 128, input_length=block_size))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    if with_attention:
        model.add(Attention())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(64))
    # model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model


def append_history(total_history, new_history):
    for key in total_history.keys():
        total_history[key] += new_history[key]
    return total_history


def create_plot(x, y, title, xlabel, ylabel, save_path):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, np.range(1, len(x) + 1))
    plt.plot(y, np.range(1, len(y) + 1))
    plt.title(title)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig(save_path, dpi=fig.dpi)


# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')][:3]

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-bengali")
tokenizer.add_special_tokens(
    {'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})
vocab_size = tokenizer.vocab_size

block_size = 512
BATCH_SIZE = 12
BUFFER_SIZE = 1000
validation_split = 0.2

lstm_model = create_lstm_model(
    vocab_size=vocab_size, block_size=block_size, with_attention=False)
lstm_model_with_attention = create_lstm_model(
    vocab_size=vocab_size, block_size=block_size, with_attention=True)
lstm_model_dir = project_root / 'models/bn_lstm/'
lstm_model_with_attention_dir = project_root / 'models/bn_lstm_attention/'

lstm_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(lstm_model_dir), monitor='loss', verbose=1, save_best_only=True, mode='min')
lstm_model_with_attention_checkpoint = tf.keras.callbacks.ModelCheckpoint(str(
    lstm_model_with_attention_dir), monitor='loss', verbose=1, save_best_only=True, mode='min')

lstm_model_history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': []
}

lstm_model_with_attention_history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': []
}

for file in files:
    with open(file, 'r') as f:
        text = f.read()
        tokenized_text = tokenizer.encode(text, return_tensors='tf')
        print(tokenized_text)
        inputs = []
        labels = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            block = tokenized_text[i:i + block_size]
            inputs.append(block[:-1])
            labels.append(block[-1])
        validation_size = int(len(inputs) * validation_split)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs[:validation_size], labels[:validation_size]))
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (inputs[validation_size:], labels[validation_size:]))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(
            BATCH_SIZE, drop_remainder=True)
        history = lstm_model.fit(dataset, epochs=3, callbacks=[
                                 lstm_model_checkpoint])
        history_with_attention = lstm_model_with_attention.fit(
            dataset, epochs=3, callbacks=[lstm_model_with_attention_checkpoint])

        append_history(lstm_model_history, history.history)
        append_history(lstm_model_with_attention_history,
                       history_with_attention.history)

lstm_model.save(str(lstm_model_dir / 'model.h5'), save_format='h5')
lstm_model_with_attention.save(
    str(lstm_model_with_attention_dir / 'model.h5'), save_format='h5')

with open(str(lstm_model_dir / 'history'), 'wb') as file_pi:
    pickle.dump(lstm_model_history, file_pi)

with open(str(lstm_model_with_attention_dir / 'history'), 'wb') as file_pi:
    pickle.dump(lstm_model_with_attention_history, file_pi)

create_plot(lstm_model_history['loss'], lstm_model_history['val_loss'],
            'LSTM Model Loss', 'Loss', 'Epochs', str(lstm_model_dir / 'loss.png'))
create_plot(lstm_model_history['acc'], lstm_model_history['val_acc'],
            'LSTM Model Accuracy', 'Accuracy', 'Epochs', str(lstm_model_dir / 'acc.png'))
