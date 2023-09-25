import numpy as np
from nn import *
# from new_nn import NN
# from new_nn import Relu, Linear, SquaredLoss, CELoss
from utils import data_loader, acc, save_plot, loadMNIST, onehot
from sklearn.decomposition import PCA
import pickle
import time


# Several passes of the training data
def train(model, training_data, learning_rate, batch_size, max_epoch):
    X_train, Y_train = training_data['X'], training_data['Y']
    for i in range(max_epoch):
        for X, Y in data_loader(X_train, Y_train, batch_size=batch_size, shuffle=True):
            training_loss, grad_Ws, grad_bs = model.compute_gradients(X, Y)
            model.update(grad_Ws, grad_bs, learning_rate)
    return model


# Test the model
def test(model, dev_data):
    X_dev, Y_dev = dev_data['X'], dev_data['Y']
    dev_acc = acc(model.predict(X_dev), Y_dev)
    return dev_acc


if __name__ == "__main__":
    # default settings
    lr = 1e-2
    max_epoch = 20
    batch_size = 128

    # Dimensions for PCA
    dimensions = [2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48]  # LEO dimensions
    # [56, 112, 168, 224, 280, 336, 392, 448, 504, 560, 616, 672, 728, 784] # WEI dimensions

    train_time = []
    tr_acc = []
    test_time = []
    dev_acc = []

    # load data
    x_train, label_train = loadMNIST('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    x_test, label_test = loadMNIST('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
    y_train = onehot(label_train)
    y_test = onehot(label_test)

    # Train the neural network after dimension reduction with PCA using different dimensions
    for i in dimensions:
        print("=== Run Training for Dim=", i, " ===")
        model = NN(Relu(), SquaredLoss(), hidden_layers=[256, 256], input_d=i, output_d=10)
        print("\t NN model loaded")
        # Create PCA objects with i components
        pca = PCA(n_components=i)
        print("\t PCA object created")

        # Fit the PCA model to the data
        pca.fit(x_train.T)
        print("\t Finished finding PCs")

        # Transform the data using the PCA model
        x_train_tmp = pca.transform(x_train.T).T
        x_test_tmp = pca.transform(x_test.T).T
        print("\t Transforming Inputs")

        # Train the neural network and record the time and accuracy
        training_data = {"X": x_train_tmp, "Y": y_train}
        testing_data = {"X": x_test_tmp, "Y": y_test}

        # Train the neural network
        start = time.time()
        model = train(model, training_data, lr, batch_size, max_epoch)
        end = time.time()

        # Training time
        elapsed = end - start
        train_time.append(elapsed)
        print("\t Training Time:", elapsed)

        # Test the neural network
        start = time.time()
        dev_acc.append(test(model, testing_data))
        end = time.time()

        # Testing time
        elapsed = end - start
        test_time.append(elapsed)
        print("\t Testing Time:", elapsed)
        print("\t Accuracy:", dev_acc[-1])

        training_data = None
        testing_data = None
        x_train_tmp = None
        x_test_tmp = None
        pca = None
        model = None

    save_plot(dimensions, train_time, "Training Time vs. Dimensions")
    save_plot(dimensions, test_time, "Testing Time vs. Dimensions")
    save_plot(dimensions, dev_acc, "Accuracy vs. Dimensions")

    save_dict = {"train_time": train_time,
                 "test_time": test_time,
                 "test_acc": dev_acc,
                 "dimensions": dimensions}

    with open('time_stuff_LEO.pickle', 'wb') as ofile:
        pickle.dump(save_dict, ofile)
