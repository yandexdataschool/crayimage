import argparse

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

import crayfis

parser = argparse.ArgumentParser(description="")
parser.add_argument('--num_train_files', metavar="num_train_files", type=int, default=10)
parser.add_argument('--num_test_files', metavar="num_test_files", type=int, default=2)
parser.add_argument('--threshold', metavar="threshold", type=int, default=50)
parser.add_argument('--num_epochs', metavar="num_epochs", type=int, default=50)
args = parser.parse_args()

num_bg_file_train = args.num_train_files
num_sg_file_train = args.num_train_files

num_bg_file_test = args.num_train_files + args.num_test_files
num_sg_file_test = args.num_train_files + args.num_test_files

n_row = 28
n_col = 28

threshold = args.threshold
num_epochs = args.num_epochs

###

input_img = Input(shape=(n_row, n_col, 1))
x = Conv2D(196, (3,3), activation="relu", padding="same")(input_img)
x = MaxPooling2D((2,2), padding="same")(x)
x = Conv2D(98, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D((2,2), padding="same")(x)
x = Conv2D(49, (3,3), activation="relu", padding="same")(x)
encoded = MaxPooling2D((2,2), padding="same")(x)

# bottle neck: 

x = Conv2D(49, (3,3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(98, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(196, (3,3), activation="relu")(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")

# input dataset
from keras.datasets import mnist
import crayfis
import numpy as np
(x_train, _), (x_test, _) = crayfis.load_data(num_bg_file_train, num_bg_file_test,
                                              num_sg_file_train, num_sg_file_test,
                                              n_row, n_col, threshold)

x_train = x_train.astype("float32") 
x_test = x_test.astype("float32") 
x_train = np.reshape(x_train, (len(x_train), n_row, n_col, 1))
x_test = np.reshape(x_test, (len(x_test), n_row, n_col, 1))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=num_epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 12
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(n_row, n_col))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(n_row, n_col))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("noise_autoencoder.png")
