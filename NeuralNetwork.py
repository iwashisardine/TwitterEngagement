import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

dataset_path = './data/Rpat2_osusume.csv'
pattern = 2

raw_dataset = pd.read_csv(dataset_path)

dataset = raw_dataset.copy()

dataset.isna().sum()
dataset = dataset.dropna()

if pattern == 2 or pattern == 4:
  day = dataset.pop('day')
  dataset['Mon'] = (day == 1)*1.0
  dataset['Tue'] = (day == 2)*1.0
  dataset['Wed'] = (day == 3)*1.0
  dataset['Thu'] = (day == 4)*1.0
  dataset['Fri'] = (day == 5)*1.0
  dataset['Sat'] = (day == 6)*1.0
  dataset['Sun'] = (day == 7)*1.0

  if pattern == 2:
    dataset.drop(columns='minute')

  hour = dataset.pop('hour')
  dataset['am0'] = (hour == 0)*1.0
  dataset['am1'] = (hour == 1)*1.0
  dataset['am2'] = (hour == 2)*1.0
  dataset['am3'] = (hour == 3)*1.0
  dataset['am4'] = (hour == 4)*1.0
  dataset['am5'] = (hour == 5)*1.0
  dataset['am6'] = (hour == 6)*1.0
  dataset['am7'] = (hour == 7)*1.0
  dataset['am8'] = (hour == 8)*1.0
  dataset['am9'] = (hour == 9)*1.0
  dataset['am10'] = (hour == 10)*1.0
  dataset['am11'] = (hour == 11)*1.0
  dataset['am12'] = (hour == 12)*1.0
  dataset['pm1'] = (hour == 13)*1.0
  dataset['pm2'] = (hour == 14)*1.0
  dataset['pm3'] = (hour == 15)*1.0
  dataset['pm4'] = (hour == 16)*1.0
  dataset['pm5'] = (hour == 17)*1.0
  dataset['pm6'] = (hour == 18)*1.0
  dataset['pm7'] = (hour == 19)*1.0
  dataset['pm8'] = (hour == 20)*1.0
  dataset['pm9'] = (hour == 21)*1.0
  dataset['pm10'] = (hour == 22)*1.0
  dataset['pm11'] = (hour == 23)*1.0

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("engagement")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('engagement')
test_labels = test_dataset.pop('engagement')

# データの正規化
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# モデルの構築
def build_model():
  model = keras.Sequential([
    layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

#model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)

# エポックが終わるごとにドットを一つ出力することで進捗を表示
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def plot_history(history, path = ''):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  fig1 = plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [engagement]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0, 150])
  plt.legend()

  fig2 = plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$engagement^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,2500000])
  plt.legend()
  plt.show()

  if not path == '':
    fig1.savefig(path + '_mae.png')
    fig2.savefig(path + '_mse.png')

#plot_history(history)

output_path = '/content/drive/Shareddrives/NLP/outputs/Rpat2_osusume_10-34' #.pngはつけない

EPOCHS = 100

model = build_model()

# patience は改善が見られるかを監視するエポック数を表すパラメーター
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
#                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[PrintDot()])

plot_history(history, output_path)
#plot_history(history)

import math

output_path = '/content/drive/Shareddrives/NLP/outputs/Rpat2_osusume_10-32.txt'
loss, mae, mse = model.evaluate(normed_test_data,test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} engagement".format(mae))
print("Testing set Mean Squared Error: {:5.2f} engagement".format(mse))
print("Testing set Root Mean Squared Error: {:5.2f} engagement".format(math.sqrt(mse)))

with open(output_path, 'a', newline='') as f:
    f.write("Testing set Mean Abs Error: {:5.2f} engagement".format(mae))
    f.write("Testing set Mean Squared Error: {:5.2f} engagement".format(mse))
    f.write("Testing set Root Mean Squared Error: {:5.2f} engagement".format(math.sqrt(mse)))