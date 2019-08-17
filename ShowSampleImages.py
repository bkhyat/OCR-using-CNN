import numpy as np
from matplotlib import pyplot as plt
from GetClassLabelFromIndex import get_class_label_from_index as get_class


def pick_sample_test_data():
    data = np.load("preprocessed_test_data.npy", allow_pickle=True) #Preprocessed Test Data
    pixels = list()
    labels = list()
    for x in data:
        pixels.append(np.array(x[0]))
        labels.append(get_class(x[1]))
        if len(pixels) >= 16:
            break
    pixels = np.array(pixels)
    before = [np.array(x)*255 for x in pixels]
    after = [255-np.array(x)*255 for x in pixels]
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Some sample test images before preprocessing", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(before[i], cmap='binary')
        x_label = "Class Label: {}".format(labels[i])
        ax.set_xlabel(x_label)
    plt.show()
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Some sample test images after preprocessing", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(after[i], cmap='binary')
        x_label = "Class Label: {}".format(labels[i])
        ax.set_xlabel(x_label)
    plt.show()


def pick_sample_train_data():
    data = np.load("preprocessed_train_data.npy", allow_pickle=True) #Preprocessed Train Data
    pixels = list()
    labels = list()
    for x in data:
        pixels.append(np.array(x[0]))
        labels.append(get_class(x[1]))
        if len(pixels) >= 16:
            break
    pixels = np.array(pixels)
    before = [np.array(x)*255 for x in pixels]
    after = [255-np.array(x)*255 for x in pixels]
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Some sample train images before preprocessing", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(before[i], cmap='binary')
        x_label = "Class Label: {}".format(labels[i])
        ax.set_xlabel(x_label)
    plt.show()
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Some sample train images after preprocessing", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(after[i], cmap='binary')
        x_label = "Class Label: {}".format(labels[i])
        ax.set_xlabel(x_label)
    plt.show()


def showsampleimages():
    pick_sample_train_data()
    pick_sample_train_data()
