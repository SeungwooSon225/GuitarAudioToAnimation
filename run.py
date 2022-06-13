from keras.models import load_model
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


look_back = 30  # 50

model = load_model('my_model_1.h5')

# Load the files
with open('test/test3_audio.json', 'r') as file:
    audio_data = json.load(file)

audio_data = np.array(audio_data['0'])

model.summary()

plt.plot(audio_data)
plt.show()

X = []
for i in range(0, len(audio_data) - look_back):
    a = np.array(audio_data[i:i + look_back])
    X.append(a)

X = np.array(X)
shapeX = X.shape
print('Shapes', shapeX)
X = X.reshape(-1, X.shape[2])
print('Shapes:', X.shape)

scalerX = MinMaxScaler(feature_range=(0, 1))

X = scalerX.fit_transform(X)

X = X.reshape(shapeX)

y_pred = model.predict(X)
print('y_pred.shape:', y_pred.shape)

y_pred = y_pred.tolist()

result = {
    "0": y_pred
}

with open('runs/test3_video.json', 'w', encoding='utf-8') as file:
    json.dump(result, file)

plt.plot(y_pred)
plt.show()


with open('processed data/training7_video.json', 'r') as file:
    video_data = json.load(file)
plt.plot(video_data['0'])
plt.show()
