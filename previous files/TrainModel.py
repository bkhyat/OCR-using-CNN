import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D


def train():
    start_time = time.time()
    print("Starting The Training Process")
    print("Loading The Preprocessed Images Data set")
    datasets = np.load('preprocessed_train_data.npy', allow_pickle=True) #Preprocessed Train Data
    print("Preprocessed Data ser Loaded Successfully.")
    print("Shuffling The Data set")
    np.random.shuffle(datasets)
    print(datasets.shape)
    pixels = []
    labels = []
    for pixel, label in datasets:
        pixels.append(pixel)
        labels.append(label)
    pixels = np.array(pixels).reshape(-1, 32, 32, 1)
    # print('number of inputs:')
    # Building The Model:
    print("creating The Sequential Model.")
    model = Sequential()
    print("Adding First Convolution Layer To The Model.")
    model.add(Conv2D(16, (5, 5), padding="same", input_shape=(32, 32, 1), activation="relu"))
    print("Adding Pooling Layer To The First Convolution Layer.")
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Adding The Second Convolution Layer To The Model.")
    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
    print("Adding Pooling Layer To The Second Convolution Layer.")
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Flattening The Output From Second Convolution Layer.")
    model.add(Flatten())
    print("Adding The Hidden Dense Layer With 1000 Neurons.")
    model.add(Dense(1000))
    print("Adding The Output Layer With 36 Neurons For 36 Classes.")
    model.add(Dense(36))
    print("Adding The Softmax Activation At The Output Layer.")
    model.add(Activation("softmax"))
    print("Using The Cross Entropy As The Loss Function With Adam Optimizer")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Fitting The Model.")
    print("Training Begins...........")
    history = model.fit(pixels, labels, batch_size=256, validation_split=0.15, epochs=30)
    end_time = time.time()
    total_time = round(end_time-start_time)
    time_msg = "Training Completed Successfully in {Time}".format(Time=str(datetime.timedelta(seconds=total_time)))
    print(time_msg)
    print("Saving the Model in Hard Drive For Later Use")
    # serialize model to JSON
    model_json = model.to_json()
    with open("Trained_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Weights_model.h5")
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# train()
