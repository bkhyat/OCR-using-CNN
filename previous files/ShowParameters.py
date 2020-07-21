import h5py
import numpy as np
from matplotlib import pyplot as plt

layer1_filters = []
layer2_filters = []
f = h5py.File('Weights_model.h5', 'r')
for each in f['conv2d']['conv2d']['kernel:0']:
    layer1_filters.append(np.array(each))
layer1_filters = np.array(layer1_filters)
layer1_filters = layer1_filters.transpose([3, 0, 1, 2])
for each in f['conv2d_1']['conv2d_1']['kernel:0']:
    layer2_filters.append(np.array(each))
layer2_filters = np.array(layer2_filters)
layer2_filters = layer2_filters.transpose([2, 3, 0, 1])
layer2_filter_first = layer2_filters[0]
layer2_filter_first = layer2_filter_first.reshape(64, 5, 5)
for layer2_filter in layer2_filters:
    print(layer2_filter.shape)
    print((layer2_filter.reshape(64, 5, 5)).shape)
    break


def show_layer1_filters():
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Layer1 Filters ( 16 Filters of shape (5,5) )", fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(layer1_filters[i].reshape(5, 5), cmap='binary')
    plt.show()


def show_layer2_filters():
    filter_counter = 0
    for layer2_filter in layer2_filters:
        filter_counter += 1
        fig, axes = plt.subplots(8, 8)
        fig.subplots_adjust(hspace=0.45, wspace=0.3)
        fig.suptitle("Layer2 Filters ( 64 Filters of shape (5,5) for output channel {} from Layer1 )".format(filter_counter), fontsize=18)
        for i, ax in enumerate(axes.flat):
            ax.imshow(layer2_filter[i].reshape(5, 5), cmap='binary')
        plt.show()


def show_number_of_learned_parameters():
    cnn1_biases = f['conv2d']['conv2d']['bias:0']
    cnn1_kernels = f['conv2d']['conv2d']['kernel:0']
    cnn2_biases = f['conv2d_1']['conv2d_1']['kernel:0']
    cnn2_kernels = f['conv2d_1']['conv2d_1']['bias:0']
    dense1_biases = f['dense']['dense']['bias:0']
    dense1_kernels = f['dense']['dense']['kernel:0']
    dense2_biases = f['dense_1']['dense_1']['bias:0']
    dense2_kernels = f['dense_1']['dense_1']['kernel:0']
    print("Number Of Learned Parameters:")
    print("CNN Layer1 Biases : ", cnn1_biases.shape)
    print("CNN Layer1 Kernels : ", cnn1_biases.shape)
    print(cnn1_biases.size + cnn1_kernels.size)
    print("CNN Layer2 Biases : ", cnn2_biases.shape)
    print("CNN Layer2 Kernels : ", cnn2_kernels.shape)
    print(cnn2_biases.size + cnn2_kernels.size)
    print("Hidden Dense Layer Biases : ", dense1_biases.shape)
    print("Hidden Dense Layer Weights : ", dense1_kernels.shape)
    print(dense1_biases.size + dense1_kernels.size)
    print("Output Layer Biases: ", dense2_biases.shape)
    print("Output Layer Weights : ", dense2_kernels.shape)
    print(dense2_biases.size + dense2_kernels.size)
    print("Number Of Total Parameters Read : ")
    total = cnn1_biases.size + cnn1_kernels.size + \
        cnn2_biases.size + cnn2_kernels.size + \
        dense1_biases.size + dense1_kernels.size + \
        dense2_biases.size + dense2_kernels.size
    print(total)


def show_parameters():
    show_number_of_learned_parameters()
    show_layer1_filters()
    show_layer2_filters()



