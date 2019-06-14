from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')
import argparse
import numpy as np
import CellFromIllustrator
import random
import matplotlib.pyplot as plt
import os
from Constants import SIZE, RADIUS


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b


def baseline_model():
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, SIZE, SIZE), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def model_to_json(model, name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5")
    print("Saved model to disk")


def load_json_model(name):
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + ".h5")
    print("Loaded model from disk")
    return loaded_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NN")
    parser.add_argument("-t", "--train", type=str, help="Directory to input training data")
    parser.add_argument("-m", "--model", type=str, help="Filepath of model file")
    parser.add_argument("-p", "--pretrained_model", type=str, help="Optional pretrained model")
    args = parser.parse_args()
    X = []
    Y = []
    for file in os.listdir(args.train):
        print(file)
        loaded = np.load(os.path.join(args.train, file), allow_pickle=True)
        if loaded['X'].size > 0:
            print(loaded['X'].shape)
            X.extend(loaded['X'])
            Y.extend(loaded['Y'])
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    for file in os.listdir(args.train):
        os.remove(os.path.join(args.train, file))
    np.savez_compressed(os.path.join(args.train, "00.npz"), X=X, Y=Y)
    X = X.astype(np.float32) / 255
    X, Y = shuffle_in_unison_scary(X, Y)
    test_cohort_size = X.shape[0] // 20
    test_X = np.array(X[:test_cohort_size, :, :])
    test_X = test_X.reshape(test_X.shape[0], 1, SIZE, SIZE)
    test_Y = np.array(Y[:test_cohort_size, :])
    train_X = np.array(X[test_cohort_size:, :, :])
    train_X = train_X.reshape(train_X.shape[0], 1, SIZE, SIZE)
    train_Y = np.array(Y[test_cohort_size:, :])
    if args.pretrained_model is None:
        model = baseline_model()
    else:
        model = load_json_model(args.pretrained_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=5, batch_size=40)
    model_to_json(model, args.model)
    scores = model.evaluate(test_X, test_Y, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))

    for i in range(test_X.shape[0]):
        x = test_X[i: i+1, :, :, :]
        output = model.predict(x)[0]
        plt.imshow(x[0, 0, :, :], cmap="gray")
        if output[0] > output[1]:
            plt.title("Cell")
        else:
            plt.title("Not Cell")
        plt.show()
