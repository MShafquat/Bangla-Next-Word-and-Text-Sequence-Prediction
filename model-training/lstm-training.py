import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
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


# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
file = [str(file) for file in Path(data_dir).glob('**/*.txt')][0]

# create tokenizer
tokenizer = Tokenizer()
with open(file, 'r') as f:
    text = f.read()[:100000]

corpus = text.split('\n')
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_counts)+1
tokenizer_path = str(
    project_root / f'models/bn_lstm/tokenizer_len{str(total_words)}.pickle')
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_sequence_len, dtype='int32', padding='pre',
    truncating='pre', value=0
)

X = input_sequences[:, :-1]
labels = input_sequences[:, -1]
Y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(
    total_words, max_sequence_len-1, input_length=max_sequence_len-1))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(64))
# model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam, metrics=['accuracy'])
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=25, mode='auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint(str(
    project_root / 'models/bn_lstm'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(X, Y, epochs=500, validation_split=0.2,
                    callbacks=[earlystop, checkpoint])
model.save(str(project_root / 'models/bn_lstm/bn_lstm.h5'), save_format='h5')

with open(str(project_root / 'models/bn_lstm/history'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

fig = plt.figure(figsize=(3, 6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
fig.savefig(str(project_root / 'models/bn_lstm/accuracy.png'), dpi=fig.dpi)

fig = plt.figure(figsize=(3, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
fig.savefig(str(project_root / 'models/bn_lstm/loss.png'), dpi=fig.dpi)
