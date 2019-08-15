# OCR-using-CNN

A computer Engineering Major Project For Optical Character Recognition using Convolution Neural Network
-
Objective: To build and train a convolution neural network for the optical character recognition
-
Running The Program
-
Just Extract the Program.rar folder first.
It was developed using Pycharm Edu so you can download the program folder and open the project using PyCharm. Since it contains the virtual environment of all the required dependencies.

Files Description:
-
Preprocess.py : 
-
It is used to preprocess the training and testing dataset. Just download the Kaggle dataset from https://www.kaggle.com/vaibhao/handwritten-characters and extract it. Then specify the extracted folder location of the data set to the path varaible inside the preprocess() function of Preprocess.py

TrainModel.py: 
-
It is used to build and train the model using keras sequential model. Just make sure to provide the proper path for the preprocessed training dataset obtained from the preprocess.py Then just call the train() function in order to train the model. It automatically plots the training and validation accuracy and loss for the 30 epochs after showing the training and validation accuracy. It takes a couple of hours to train the model for 30 epochs. You can change the number of epoch as per your design. The function also displays the training time once the model is trained. The trained model and the weights are saved for the later use of the model.

TestModel.py: 
-
It contains function test() in order to test the model. Just make sure you have preprocessed the test data set using the preprocess() function of the Preprocess.py. It will take a few seconds or minutes to test the samples. After completion of the test, it shows the test accuracy and plots some of the truely calssified and miss classified samples using matplotlib. It then plots the confusion matrix for the test samples. 

Predict.py:
-
It contains the preidct() function that takes path of an jpeg image in order to predict the character inside the image. You can call it providing the path of your own sample image. But make sure that image is in jpeg format. It will plot the original image along with the preprocessed image and give the class label for the particular character inside the input jpeg image. For e.g.: you can provide sample1.jpeg, sample2.jpeg, sample3.jpeg, sample4.jpeg and so on already contained in this guthub project.

UI.py: 
-
Well, it is there just to allow you to test with your own input using GUI. Just run the UT.py then it will pop-up a window along with the button to select an image. Just click on it and make sure to select a jpeg image to test your own data sample. 

There are couple of files like ShowParamaters.py and ShowSampleImages.py just in case you want to see the plots of filters, some sample test and test images before and after preprocessing. Just call the functions inside them to see the visualization of those data.

