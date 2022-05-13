import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_path = 'BrainTumorDataset/'

kategoriler = os.listdir(data_path)
labels = [i for i in range(len(kategoriler))]

label_dict = dict(zip(kategoriler, labels))
# print(label_dict)
# print(kategoriler)
# print(labels)

img_size = 256
data = []
label = []

for kategori in kategoriler:
    folder_path = os.path.join(data_path, kategori)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            label.append(label_dict[kategori])

        except Exception as e:
            print('Exception:', e)


data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
label = np.array(label)
new_label = to_categorical(label)

# print(new_label.shape)
# print(data.shape)
# print(data.shape[1:])

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(data, new_label, test_size=0.1)

# plt.figure(figsize=(10, 10))

# for i in range(20):
#    plt.subplot(5, 5, i + 1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(np.squeeze(x_test[i]))
#    plt.xlabel(kategoriler[np.argmax(y_test[i])])
# plt.show()

model_ = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

model.save('BrainTrainingModel.h5')

vaL_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("test loss:", vaL_loss, '%')
print("test accuracy:", val_accuracy, "%")