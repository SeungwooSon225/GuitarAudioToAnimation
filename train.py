from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Lambda, TimeDistributed
import keras.backend as K
import keras
from sklearn.preprocessing import MinMaxScaler
import json
import numpy as np
from tqdm import tqdm
from keras.callbacks import TensorBoard
from time import time
import matplotlib.pyplot as plt
from keras import optimizers

time_delay = 30  # 0
look_back = 30  # 50
n_epoch = 50
tbCallback = TensorBoard(log_dir="logs/{}".format(time()))

# Load the files
with open('processed data/concatenated_audio_data.json', 'r') as file:
    audio_data = json.load(file)

with open('processed data/concatenated_video_data.json', 'r') as file:
    video_data = json.load(file)

audio_data = np.array(audio_data['0'])
video_data = np.array(video_data['0'])

X, y = [], []
for i in range(time_delay, len(audio_data) - look_back):
    a = np.array(audio_data[i:i + look_back])
    # v = np.array(video_data[i + look_back - time_delay]).reshape((1, -1))

    # tt = np.append(video_data[i], video_data[i])
    v = np.array([np.append(np.append(video_data[i - time_delay],  video_data[i]), video_data[i+time_delay]).flatten()])
    # v = np.array([np.array(video_data[i - time_delay:i]).flatten()])
    X.append(a)
    y.append(v)

X = np.array(X)
y = np.array(y)
shapeX = X.shape
shapey = y.shape
print('Shapes', shapeX, shapey)
X = X.reshape(-1, X.shape[2])
y = y.reshape(-1, y.shape[2])
print('Shapes:', X.shape, y.shape)

scalerX = MinMaxScaler(feature_range=(0, 1))
# scalery = MinMaxScaler(feature_range=(0, 1))

X = scalerX.fit_transform(X)
# y = scalery.fit_transform(y)

X = X.reshape(shapeX)
y = y.reshape(shapey[0], shapey[2])

# plt.plot(X)
# plt.show()
plt.plot(y)
plt.show()

print('Shapes:', X.shape, y.shape)
print('X mean:', np.mean(X), 'X var:', np.var(X))
print('y mean:', np.mean(y), 'y var:', np.var(y))

split1 = int(0.8 * X.shape[0])
split2 = int(0.9 * X.shape[0])

train_X = X[0:split1]
train_y = y[0:split1]
val_X = X[split1:split2]
val_y = y[split1:split2]
test_X = X[split2:]
test_y = y[split2:]

# Initialize the model

model = Sequential()
model.add(LSTM(500, input_shape=(look_back, 26)))  # 25, input_shape=(look_back, 26)))
model.add(Dropout(0.2)) # 0.25
model.add(Dense(192))
adam = optimizers.Adam(learning_rate=0.00008, beta_1=0.9, beta_2=0.98, epsilon=0.000000001)
model.compile(loss='mean_squared_error', optimizer=adam)
print(model.summary())
# model = load_model('my_model.h5')

# train LSTM with validation data
# for i in tqdm(range(n_epoch)):
print('Epoch', (i + 1), '/', n_epoch, ' - ', int(100 * (i + 1) / n_epoch))
model.fit(train_X, train_y, epochs=250, batch_size=64,
          verbose=1, shuffle=True, callbacks=[tbCallback], validation_data=(val_X, val_y))
# model.reset_states()
test_error = np.mean(np.square(test_y - model.predict(test_X)))
# model.reset_states()
print('Test Error: ', test_error)

# Save the model
model.save('my_model.h5')
model.save_weights('my_model_weights.h5')
print('Saved Model.')
