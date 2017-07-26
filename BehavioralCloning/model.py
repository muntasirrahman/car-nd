import csv
import matplotlib.image as mpimg
import numpy as np

#Loading lines from log file
lines = []
with open('training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:

    measurement = float(line[3])
    #To read image from center, left, and right camera
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'training/IMG/' + filename

        image = mpimg.imread(current_path)
        images.append(image)

        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
#Based on NVIDIA model, with slight modification of Dropout layer
model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.4)) #Dropout rate 0.4
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()
