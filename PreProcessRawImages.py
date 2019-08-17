import numpy as np
import glob
from PIL import Image as Img
import random                   # To Shuffle The Data set

CATEGORIES = list()
pix_val = list()
global_counter = 0


def read_inputs(c):
    global global_counter
    CATEGORIES.append(c)
    index = CATEGORIES.index(c)
    path = 'E:/MajorProject/DataSet/Train/'+c+'/*.jpg'  #Path to the Dataset
    i = 0
    print("Reading Raw Images For: ", c)
    for filename in glob.glob(path):
        im = Img.open(filename)
        temp = np.array(list(im.getdata()))  # Getting Pixel Value in 1D array
        temp = (255 - temp)/255              # Normalizing The Data
        temp = np.around(temp, 4)            # Rounding up to 4 decimal places
        pix_val.append([temp, index])        # Appending 2D array along with the category index value
        i += 1
    global_counter += i
    print("Number Of Raw Images Read: ", i)


def preprocess():
    print("Starting the Pre-processing of the Raw Images....")
    read_inputs("0")
    read_inputs("1")
    read_inputs("2")
    read_inputs("3")
    read_inputs("4")
    read_inputs("5")
    read_inputs("6")
    read_inputs("7")
    read_inputs("8")
    read_inputs("9")
    read_inputs("A")
    read_inputs("B")
    read_inputs("C")
    read_inputs("D")
    read_inputs("E")
    read_inputs("F")
    read_inputs("G")
    read_inputs("H")
    read_inputs("I")
    read_inputs("J")
    read_inputs("K")
    read_inputs("L")
    read_inputs("M")
    read_inputs("N")
    read_inputs("P")
    read_inputs("Q")
    read_inputs("R")
    read_inputs("S")
    read_inputs("T")
    read_inputs("U")
    read_inputs("V")
    read_inputs("W")
    read_inputs("X")
    read_inputs("Y")
    read_inputs("Z")
    print("Total Raw Images Read: ", global_counter)
    print("Shuffling all the images")
    random.shuffle(pix_val)
    print("Writing The Image Data To A File")
    np.save('train_data.npy', pix_val)
    print("Writing The Image Classes And Indexes To A File")
    with open('test_category.txt', 'w') as file:
        for c in CATEGORIES:
            file.write(str(CATEGORIES.index(c))+';'+c+'\n')

