import numpy as np
from keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

from layer import Dense
from activation import Tanh, Softmax, Sigmoid
from losses import mse, mse_prime
from network import stoch_train, predict

import matplotlib.pyplot as plt
import numpy as np


def prepro_data(x, y, limit):
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255

    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = prepro_data(x_train, y_train, 1000)
x_test, y_test = prepro_data(x_test, y_test, 20)


network1 = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

network2 = [
    Dense(28 * 28, 40),
    Sigmoid(),     
    Dense(40, 10),
    Tanh()
]

epochs = 200

#err_list1 = train(network1, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1)
err_list2 = stoch_train(network2, mse, mse_prime, x_train, y_train, epochs=epochs, learning_rate=0.1)

plt.figure(figsize=(10, 6))
#plt.plot(np.arange(0, epochs), err_list1, label='network 1', color='blue')
plt.plot(np.arange(0, epochs), err_list2, label='network 2', color='red')

plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs')
plt.legend()
plt.grid(True)
plt.show()


for x, y in zip(x_test, y_test):
    output = predict(network2, x)
    print("pred: ", np.argmax(output), "\ttrue: ", np.argmax(y))

