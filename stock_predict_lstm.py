from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def batch_generator(all_data, batch_size, shuffle=True):
    """
    :param all_data : all_data整个数据集
    :param batch_size: batch_size表示每个batch的大小
    :param shuffle: 每次是否打乱顺序
    :return:
    """
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    print("data_size: ", data_size)
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]


file_path = "C:/Users/neniu/PycharmProjects/TensorflowTutorials/text_classification/data/_SP500.csv"
input_size = 5
num_steps = 3
test_ratio = 0.1
BATCH_SIZE = 64
BUFFER_SIZE = 1000
lstm_size = 128

info = pd.read_csv(file_path)
print(info)
price = np.array(info['Close'].tolist())
print(price)
seq = []
for i in range(len(price) // input_size):
    seq.append(np.array(price[i * input_size: (i + 1) * input_size]))
print("seq1:", seq)
# normalize and use ratio
for i, curr in enumerate(seq[1:3]):
    print("i:", i, "enu:", curr, "value:", seq[i][-1])
seq = [seq[0] / seq[0][0] - 1.0] + [
    curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]
print("seq2:", seq)
# split into groups of num_steps
x = []
y = []
for i in range(len(seq) - num_steps):
    x.append(np.array(seq[i: i + num_steps]))
    y.append(np.array(seq[i + num_steps]))
print("seq_x0:", x[0:3])
print("seq_y0:", y[0:3])
x = np.array(x)
y = np.array(y)
print("seq_x:", x[0:3])
print("seq_y:", y[0:3])

# split into training and test data
train_size = int(len(x) * (1.0 - test_ratio))
train_x, test_x = x[:train_size], x[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(input_size, activation='sigmoid')
])

# this model below is too big to have high accuracy, underfitting case
# model = tf.keras.Sequential([
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size//2)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(input_size, activation='sigmoid')
# ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='mse',
              metrics=['mse'])

history = model.fit(train_x, train_y, epochs=2, batch_size=BATCH_SIZE,
                    validation_split=0.1, validation_steps=20)
predict = model.predict(test_x[0:1]).flatten()
print("predict:", predict)
print("real:", test_y[0])

x_asix = []
for i in range(input_size):
    x_asix.append("day_" + str(i + 1))
plt.plot(x_asix, predict, color='red', label="predict")
plt.plot(x_asix, test_y[0], color='blue', label="real")
plt.xticks(x_asix)
plt.legend(loc='upper right')
plt.show()
