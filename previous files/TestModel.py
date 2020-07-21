import tensorflow as tf
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from GetClassLabelFromIndex import get_class_label_from_index as get_label


def test():
    print("Testing The Model")
    print("Loading The Test Data Set")
    data = np.load('preprocessed_test_data.npy', allow_pickle=True) #Preprocessed Test Data
    pixels = []
    labels = []
    for pixel, label in data:
        pixels.append(pixel)
        labels.append(label)
    labels = np.array(labels)
    pixels = np.array(pixels).reshape(-1, 32, 32, 1)
    print("Loading The Trained Model.")
    json_model = open('Trained_Model.json', 'r')
    loaded_model_json = json_model.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    print("Loading The Trained Weights.")
    loaded_model.load_weights("Weights_model.h5")
    print("Compiling The Model")
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model Compiled Successfully.")
    print("Feeding The Test Data Set To The Model.")
    print("Starting The Test.")
    test_score, test_accuracy = loaded_model.evaluate(pixels, labels)
    print("Test Accuracy : ", round(test_accuracy, 4))
    print("Plotting The Confusion Matrix")
    y_prob = loaded_model.predict(pixels)
    y_classes = y_prob.argmax(axis=-1)
    truly_classified = list()
    truly_classified_label = list()
    miss_classified = list()
    miss_classified_label = list()
    miss_classified_label_true = list()
    for i in range(len(y_classes)):
        if y_classes[i] == labels[i]:
            truly_classified.append(pixels[i])
            truly_classified_label.append(labels[i])
            if len(truly_classified) >= 12:
                break
    for i in range(len(y_classes)):
        if y_classes[i] != labels[i]:
            miss_classified.append(pixels[i])
            miss_classified_label.append(y_classes[i])
            miss_classified_label_true.append((labels[i]))
            if len(miss_classified) >= 12:
                break
    fig, axes = plt.subplots(4, 3)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Some of truly classified test samples", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(truly_classified[i]).reshape(32, 32), cmap='binary')
        x_label = "Class Label: {} Predicted Label: {}".format(get_label(truly_classified_label[i]), get_label(truly_classified_label[i]))
        ax.set_xlabel(x_label)
    plt.show()
    fig, axes = plt.subplots(4, 3)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Some of miss-classified test samples", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(miss_classified[i]).reshape(32, 32), cmap='binary')
        x_label = "Class Label: {} Predicted Label: {}".format(get_label(miss_classified_label_true[i]), get_label(miss_classified_label[i]))
        ax.set_xlabel(x_label)
    plt.show()

    cm = confusion_matrix(labels, y_classes)
    df_cm = pd.DataFrame(cm, index = [i for i in "0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ"],
                  columns = [i for i in "0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ"])
    plt.figure(figsize=(25, 25))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}, cmap='binary')
    plt.show()


#test()
