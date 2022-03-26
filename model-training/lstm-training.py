import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt


def create_tokenizer(max_words, files, model_dir):
    tokenizer = Tokenizer(num_words=max_words)
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                tokenizer.fit_on_texts([line])
    save_path = model_dir / f'tokenizer{len(tokenizer.word_counts) + 1}.pickle'
    with open(save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def create_dataset(tokenizer, files, max_words, max_sequence_length):
    input_sequences = []

    for file in files:
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


def create_plot(history, model_dir):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    fig.savefig(str(model_dir / 'accuracy.png'), dpi=fig.dpi)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    fig.savefig(str(model_dir / 'loss.png'), dpi=fig.dpi)


# get data and model directories
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
model_dir = project_root / 'models/bn_lstm'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')]

# initialize train and test files
train_test_split = 0.5
train_files = files[:-int(len(files) * train_test_split)]
test_files = files[-int(len(files) * train_test_split):]

# create tokenizer
max_words = 30000
max_sequence_length = 200
tokenizer = create_tokenizer(max_words, files, model_dir)
total_words = len(tokenizer.word_counts)+1

# create dataset
X_train, y_train = create_dataset(
    tokenizer, train_files, total_words, max_sequence_length)
X_test, y_test = create_dataset(
    tokenizer, test_files, total_words, max_sequence_length)

# create, train and save model
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, mode='auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint(str(
    project_root / 'models/bn_lstm'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model = create_lstm_model(total_words, max_sequence_length)
history = model.fit(X_train, y_train, epochs=2, validation_split=0.2,
                    callbacks=[earlystop, checkpoint])
model.save(str(project_root / 'models/bn_lstm/bn_lstm.h5'), save_format='h5')

# save history and plot accuracy and loss
with open(str(project_root / 'models/bn_lstm/history'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
create_plot(history, model_dir)
