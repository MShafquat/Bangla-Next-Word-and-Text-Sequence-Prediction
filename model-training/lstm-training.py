from pathlib import Path
import numpy as np
import tensorflow as tf

word_to_id = {}
id_to_word = {}

def generate_data_from_directory(path, windowsize, batchsize):
    inputs = []
    labels = []
    batch_count = 0
    while True:
        files = Path(path).glob('**/*.txt')
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip().split()
                    for word in line:
                        if word not in word_to_id:
                            word_to_id[word] = len(word_to_id)
                            id_to_word[len(id_to_word)] = word
                    line = [word_to_id[word] for word in line]
                    if len(line) > 0:
                        for window in range(0, len(line) - windowsize, windowsize):
                            inputs.append(line[window:window + windowsize])
                            labels.append(line[window + windowsize])
                            batch_count += 1
                            if batch_count > batchsize:
                                batch_count = 0
                                inputs = np.array(inputs)
                                labels = np.array(labels)
                                yield (inputs, labels)
                                inputs = []
                                labels = []
# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim=5, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# for inputs, labels in generate_data_from_directory(data_dir, windowsize=2, batchsize=32):
#     labels = [id_to_word[label] for label in labels]
#     print(labels)
# Fit data to model
model.fit(generate_data_from_directory(data_dir, windowsize=5, batchsize=32), epochs=10, steps_per_epoch = 130)

