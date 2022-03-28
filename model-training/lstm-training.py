import os
import pickle
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from matplotlib import pyplot as plt


def create_tokenizer(max_words, files, model_dir):
    tokenizer = Tokenizer(num_words=max_words)
    for file in tqdm(files):
        with open(file, 'r') as f:
            for line in f:
                tokenizer.fit_on_texts([line])
    save_path = model_dir / 'tokenizer.pickle'
    with open(save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def create_dataset(tokenizer, files, max_words, max_sequence_length):
    input_sequences = []

    for file in tqdm(files):
        with open(file, 'r') as f:
            for line in f:
                token_list = tokenizer.texts_to_sequences([line])[0]
                for i in range(1, len(token_list)):
                    # keep only sequences of length <= max_sequence_length
                    n_gram_sequence = token_list[max(
                        0, i+1-max_sequence_length):i+1]
                    input_sequences.append(n_gram_sequence)

    # pad sequences with 0s so that all sequences have same length
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        input_sequences, maxlen=max_sequence_length, padding='pre')
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = tf.keras.utils.to_categorical(label, num_classes=max_words+1)
    return predictors, label


def create_lstm_model(max_words, max_sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            max_words+1, max_sequence_length-1, input_length=max_sequence_length-1),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(max_words+1, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def create_plot(history, model_dir, type='train'):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    fig.savefig(str(model_dir / f'{type}_accuracy.png'), dpi=fig.dpi)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    fig.savefig(str(model_dir / f'{type}_loss.png'), dpi=fig.dpi)


def train_lstm_model(file, model, epochs=10):
    X_train, y_train = create_dataset(tokenizer, [file], max_words, max_sequence_length)
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_split=0.2, verbose=2)
    return history

def append_history(history, new_history):
    history['accuracy'] += new_history.history['accuracy']
    history['val_accuracy'] += new_history.history['val_accuracy']
    history['loss'] += new_history.history['loss']
    history['val_loss'] += new_history.history['val_loss']
    return history


# get data and model directories
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'data/'
model_dir = project_root / 'models/bn_lstm'
os.makedirs(model_dir, exist_ok=True)

files = [str(file) for file in Path(data_dir).glob('**/*.txt')]
sorted(files, key=os.path.getsize)
files = files[:10] # train on only 50 files

# initialize train and test files
train_test_split = 0.2
train_files = files[:-int(len(files) * train_test_split)]
test_files = files[-int(len(files) * train_test_split):]

# create tokenizer
max_words = 10000
max_sequence_length = 100

print('Creating tokenizer...')
tokenizer = create_tokenizer(max_words, train_files, model_dir)
total_words = len(tokenizer.word_counts)+1
print(f'Created a tokenizer of total words: {total_words}')


# create, train and save model
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, mode='auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(model_dir), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model = create_lstm_model(max_words, max_sequence_length)
print('Training model...')

history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

for file in train_files:
    print(file)
    new_history = train_lstm_model(file, model, epochs=1)
    history = append_history(history, new_history)

model.save(str(model_dir / 'bn_lstm.h5'), save_format='h5')
# test_history = model.evaluate(X_test, y_test, batch_size=16)
print('Trained model saved to disk.')

# save history and plot accuracy and loss
with open(str(model_dir / 'history'), 'wb') as file_pi:
    pickle.dump(history, file_pi)
# with open(str(model_dir / 'test_history'), 'wb') as file_pi:
#     pickle.dump(test_history, file_pi)

create_plot(history, model_dir)
