import tkinter as tk
from tkinter import filedialog
from PreProcessRawImages import preprocess
from TrainModel import train
from TestModel import test
from ShowParameters import show_layer1_filters, show_layer2_filters, show_number_of_learned_parameters
from ShowSampleImages import pick_sample_test_data, pick_sample_train_data
from Predict import predict


def onclick(args):
    if args == 1:
        preprocess()
    elif args == 2:
        train()
    elif args == 3:
        test()
    elif args == 4:
        show_layer1_filters()
        show_layer2_filters()
        show_number_of_learned_parameters()
    elif args == 5:
        pick_sample_test_data()
        pick_sample_train_data()
    else:
        path = tk.filedialog.askopenfile(title="Select Image File", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        print(path)
        try:
            predict(path.name)
        except:
            print("Error While Reading File")

root = tk.Tk()
root.geometry("500x300+300+200")
root.title("Optical Alpha-Numeric Character Recognition")
preprocess_button = tk.Button(root, text="Preprocess Data Sample", command=lambda: onclick(1))
#preprocess_button.pack()
train_button = tk.Button(root, text="Train Model", command=lambda: onclick(2))
#train_button.pack()
test_button = tk.Button(root, text="Test Model", command=lambda: onclick(3))
#test_button.pack()
parameters_button = tk.Button(root, text="Show Learned Parameters", command=lambda: onclick(4))
#parameters_button.pack()
sample_button = tk.Button(root, text="Show Sample Images", command=lambda: onclick(5))
#sample_button.pack()
predict_button = tk.Button(root, text="Select An Image To Make Prediction", command=lambda: onclick(6))
predict_button.pack()
#preprocess_button.pack()


root.mainloop()
