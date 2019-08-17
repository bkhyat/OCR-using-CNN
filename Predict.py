import tensorflow as tf
from PIL import Image as Img
from matplotlib import pyplot as plt
import numpy as np
from GetClassLabelFromIndex import  get_class_label_from_index as get_class


def predict(path):
    path = path
    print("Reading Input")
    im = Img.open(path)
    images = list()
    images.append(im)
    im = im.resize((32, 32), Img.ANTIALIAS)
    im.save("resized_sample.jpg")
    im = Img.open("resized_sample.jpg")
    pixel = np.array(list(im.convert('L').getdata()))
    pixel = pixel.reshape(32, 32, 1)
    pixel = 255-pixel
    images.append(pixel.reshape(32, 32))
    fig, axes = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig_title = "\n\nOriginal Image                             " +\
                "                             Preprocessed Image"
    fig.suptitle(fig_title, fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap ='binary')
    lst1 = list()
    pixel = pixel.reshape(32, 32, 1)
    lst1.append(pixel)
    pixel = np.array(lst1)
    print("Loading The Trained Model.")
    json_model = open('Trained_Model.json', 'r')
    loaded_model_json = json_model.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    print("Loading The Trained Weights.")
    loaded_model.load_weights("Weights_model.h5")
    print("Compiling The Model")
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predicted = loaded_model.predict_classes(pixel)
    print(predicted)
    for i, ax in enumerate(axes.flat):
        if i == 1:
            ax.set_xlabel("Predicted Class: {}".format(get_class(predicted)), fontsize=16)
    plt.show()


# predict("sample3.jpg")
